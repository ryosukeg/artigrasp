#!/usr/bin/env python3
"""Render evaluated trajectories with their actual URDF visual geometry.

The MP4s are recorded-state playback. No physics/policy rerun, invented motion,
or change to the quantitative result is involved. Requires the existing
text2hoi environment (pyrender, trimesh, scipy, Pillow) and ffmpeg.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from functools import lru_cache
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import pyrender
from scipy.spatial.transform import Rotation
import trimesh

LABELS={'horizontal':'水平','height':'高さ','orientation':'向き','articulation':'関節角度','all':'全て'}
FIELDS=('paper_position_error_m','paper_orientation_error_rad','paper_articulation_error_rad')


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def vec(value,default=(0,0,0)):
    return np.array(default if value is None else [float(v) for v in value.split()])


def origin(element):
    t=np.eye(4)
    if element is not None:
        t[:3,:3]=Rotation.from_euler('xyz',vec(element.get('rpy'))).as_matrix()
        t[:3,3]=vec(element.get('xyz'))
    return t


def pose(position,q):
    t=np.eye(4)
    t[:3,:3]=Rotation.from_quat(np.roll(q,-1)).as_matrix()
    t[:3,3]=position
    return t


class Model:
    def __init__(self,path,color,hand=False,ghost=False):
        tree=ET.parse(path).getroot()
        joints=tree.findall('joint')
        children={j.find('child').get('link') for j in joints}
        self.root=(('right' if path.name.startswith('rhand') else 'left')+'_wrist_rz') if hand else next(
            l.get('name') for l in tree.findall('link') if l.get('name') not in children)
        links={l.get('name'):l for l in tree.findall('link')}
        by_parent={}
        for j in joints:by_parent.setdefault(j.find('parent').get('link'),[]).append(j)
        self.joints=[];self.visuals=[];self.bounds=[];self.count=0
        material=pyrender.MetallicRoughnessMaterial(baseColorFactor=color,
                    metallicFactor=0,roughnessFactor=.8,alphaMode='BLEND' if ghost else 'OPAQUE',
                    doubleSided=True)
        def visit(name):
            for visual in links[name].findall('visual'):
                mesh=visual.find('geometry/mesh')
                if mesh is None:continue
                tm=trimesh.load(path.parent/mesh.get('filename'),force='mesh',process=False)
                tm.apply_scale(vec(mesh.get('scale'),(1,1,1)))
                tm.apply_transform(origin(visual.find('origin')))
                if not hand:
                    # A transformed local bounding box is very loose for the
                    # obliquely scanned objects. Hull vertices give the exact
                    # world bounds without processing the full visual mesh.
                    self.bounds.append((name,np.asarray(tm.convex_hull.vertices)))
                self.visuals.append((name,pyrender.Mesh.from_trimesh(tm,material=material,smooth=False)))
            for j in by_parent.get(name,[]):
                kind=j.get('type');index=None
                if kind in ('revolute','continuous','prismatic'):
                    index=self.count;self.count+=1
                elif kind!='fixed':raise ValueError('Unsupported joint '+kind)
                axis=vec(j.find('axis').get('xyz')) if j.find('axis') is not None else vec(None,(1,0,0))
                child=j.find('child').get('link')
                self.joints.append((name,child,origin(j.find('origin')),axis,kind,index))
                visit(child)
        visit(self.root)
        if self.count!=(45 if hand else 1):raise ValueError('Unexpected coordinate count for '+str(path))

    def add(self,scene):
        return [(name,scene.add(mesh)) for name,mesh in self.visuals]

    def update(self,scene,nodes,base,coordinates):
        transforms={self.root:base}
        for parent,child,offset,axis,kind,index in self.joints:
            t=np.eye(4)
            if index is not None:
                if kind=='prismatic':t[:3,3]=axis*coordinates[index]
                else:t[:3,:3]=Rotation.from_rotvec(axis*coordinates[index]).as_matrix()
            transforms[child]=transforms[parent]@offset@t
        for name,node in nodes:scene.set_pose(node,transforms[name])
        if self.count==1:
            points=np.concatenate([v@transforms[n][:3,:3].T+transforms[n][:3,3] for n,v in self.bounds])
            self.world_bounds=np.array([points.min(axis=0),points.max(axis=0)])


def look_at(center,direction,distance):
    direction=np.asarray(direction,float);direction/=np.linalg.norm(direction)
    eye=np.asarray(center)+direction*distance
    backward=direction;right=np.cross([0,0,1],backward);right/=np.linalg.norm(right)
    up=np.cross(backward,right)
    t=np.eye(4);t[:3,:3]=np.stack([right,up,backward],axis=1);t[:3,3]=eye
    return t


@lru_cache(maxsize=10)
def models(resource,obj):
    r=Path(resource)
    return (Model(r/'mano_double/rhand_mano_demo.urdf',[.9,.48,.34,1],True),
            Model(r/'mano_double/lhand_mano_demo.urdf',[.3,.57,.9,1],True),
            Model(r/f'arctic/{obj}/{obj}.urdf',[.95,.68,.16,1]),
            Model(r/f'arctic/{obj}/{obj}.urdf',[.10,.75,.48,.16],ghost=True))


def make_scene(resource,row):
    scene=pyrender.Scene(bg_color=[.94,.96,.98,1],ambient_light=[.30,.30,.30])
    ms=models(resource,row['object']);nodes=[m.add(scene) for m in ms]
    for extents,position,color in [([3,3,.1],[.75,0,.55],[.64,.69,.73,1]),
                                  ([2,1,.5],[1.25,0,.25],[.38,.43,.48,1])]:
        tm=trimesh.creation.box(extents=extents)
        tm.apply_translation(position)
        material=pyrender.MetallicRoughnessMaterial(baseColorFactor=color,metallicFactor=0,roughnessFactor=1)
        scene.add(pyrender.Mesh.from_trimesh(tm,material=material))
    camera=scene.add(pyrender.PerspectiveCamera(yfov=np.pi/4,znear=.02,zfar=100))
    scene.main_camera_node=camera
    for direction in ([1,-1,2],[-1,1,1]):
        scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=1.2),pose=look_at([.75,0,.7],direction,3))
    return scene,ms,nodes,camera


def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)


def probe(path):
    out=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
        'stream=width,height,nb_frames:format=duration','-of','json',str(path)],text=True)
    data=json.loads(out);s=data['streams'][0]
    return dict(width=s['width'],height=s['height'],frames=int(s['nb_frames']),duration=float(data['format']['duration']))


def phase(step,timing):
    g,t=timing['grasp_steps'],timing['transit_steps']
    if step<g:return 'Grasp'
    if step<g+t:return 'Lift'
    if step<g+2*t:return 'Place'
    if step<g+(2 if timing['protocol']=='ketchup' else 3)*t+100:return 'Prepare articulation'
    return 'Articulate'


def render_one(job):
    result_dir,resource,row,fps=job
    root=Path(result_dir)
    outcome='success' if row['ood_goal_success'] else 'failure'
    dest=root/'videos'/row['object']/row['profile']/outcome
    dest.mkdir(parents=True,exist_ok=True)
    video=dest/(row['episode_id']+'.mp4');metadata=video.with_suffix('.json')
    source=root/row['trajectory_path']
    if sha256(source)!=row['trajectory_sha256']:raise ValueError('Trajectory checksum mismatch')
    if video.is_file() and metadata.is_file():
        saved=json.loads(metadata.read_text())
        if saved.get('trajectory_sha256')==row['trajectory_sha256'] and saved.get('renderer_source_sha256')==sha256(__file__):
            probe(video);return saved
    with np.load(source,allow_pickle=False) as z:states=z['states'].copy()
    if not np.isfinite(states).all():raise ValueError('Nonfinite trajectory')
    dt=float(row['environment_config']['environment']['control_dt'])
    duration=(len(states)-1)*dt
    indices=np.minimum(np.rint(np.arange(int(round(duration*fps)))/(fps*dt)).astype(int),len(states)-1)
    # Force the actual terminal state into the last displayed frame.
    indices[-1]=len(states)-1
    anchors=np.concatenate([states[:,122:125],states[:,6:9],states[:,64:67],
                            np.array(row['lift_target_position_m'])[None],np.array(row['target_position_m'])[None]])
    center0=states[0,122:125]
    bounded=anchors[np.linalg.norm(anchors-center0,axis=1)<4]
    low,high=np.min(bounded,axis=0),np.max(bounded,axis=0)
    center=(low+high)/2
    distance=max(1.1,float(np.linalg.norm(high-low)))*1.65
    scene,ms,nodes,camera=make_scene(resource,row)
    renderer=pyrender.OffscreenRenderer(800,560)
    tmp=video.with_suffix('.partial.mp4')
    encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1200x720','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','1',
        '-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)],
        stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    heading,body,small=font(24),font(18),font(14)
    initial_z=float(states[0,124]);heights=states[:,124]-initial_z;target=float(row['target']['lift_m'])
    hmax=max(target+.05,float(np.quantile(heights,.99)),.2);hmin=min(-.05,float(np.quantile(heights,.01)))
    graph_indices=np.linspace(0,len(states)-1,min(500,len(states))).astype(int)
    preview=video.with_suffix('.png')
    try:
        for number,index in enumerate(indices):
            s=states[index];actual=s[122:130]
            ms[0].update(scene,nodes[0],pose(s[6:9],s[9:13]),s[19:64])
            ms[1].update(scene,nodes[1],pose(s[64:67],s[67:71]),s[77:122])
            ms[2].update(scene,nodes[2],pose(actual[:3],actual[3:7]),[actual[7]])
            goal=s[130:138]
            if np.linalg.norm(goal[3:7])<.5:
                goal=np.r_[row['target_position_m'],row['target_quaternion_wxyz'],row['target_articulation_rad']]
            ms[3].update(scene,nodes[3],pose(goal[:3],goal[3:7]),[goal[7]])
            scene.set_pose(camera,look_at(center,[1.2,-1.7,1.2],distance))
            primary,_=renderer.render(scene,flags=pyrender.RenderFlags.SKIP_CULL_FACES)
            close_center=ms[2].world_bounds.mean(axis=0) if np.linalg.norm(actual[:3]-center0)<10 else center0
            half=(ms[2].world_bounds[1]-ms[2].world_bounds[0])/2
            camera_axes=look_at(close_center,[-1.1,-1.5,1.1],1)[:3,:3]
            projected_half=np.abs(camera_axes).T@half
            close_distance=max(.25,(max(projected_half[0]/(np.tan(np.pi/8)*800/560),
                                        projected_half[1]/np.tan(np.pi/8))+projected_half[2])*1.2)
            scene.set_pose(camera,look_at(close_center,[-1.1,-1.5,1.1],close_distance))
            detail,_=renderer.render(scene,flags=pyrender.RenderFlags.SKIP_CULL_FACES)
            frame=Image.new('RGB',(1200,720),'#f2f5f9');draw=ImageDraw.Draw(frame)
            badge='SUCCESS' if row['ood_goal_success'] else 'FAILURE'
            draw.text((20,15),f"{row['object']} / {row['profile']} / {row['sample_index']:03d}    {badge}",font=heading,fill='#17324d')
            draw.text((20,51),'Evaluated-state playback  |  Right: coral  Left: blue  Object: gold  Goal: translucent green',font=small,fill='#52677a')
            frame.paste(Image.fromarray(primary),(12,88))
            frame.paste(Image.fromarray(detail).resize((360,252)),(828,88))
            draw.text((840,348),'Object detail (tracking camera)',font=small,fill='#52677a')
            pe=np.linalg.norm(actual[:3]-np.array(row['target_position_m']))
            dq=abs(float(np.dot(actual[3:7]/np.linalg.norm(actual[3:7]),np.array(row['target_quaternion_wxyz'])/np.linalg.norm(row['target_quaternion_wxyz']))))
            ae=2*math.acos(min(1,max(0,dq)));aae=abs(float(actual[7])-row['target_articulation_rad'])
            values=[f"Time: {index*dt:.2f} / {duration:.2f} s",f"Phase: {phase(index,row['timing'])}",
                    f"Lift: {heights[index]:.3f} / {target:.3f} m",f"Joint: {actual[7]:.3f} / {row['target_articulation_rad']:.3f} rad",
                    f"PE: {pe:.3f} m",f"AE: {ae:.3f} rad   AAE: {aae:.3f} rad"]
            for k,value in enumerate(values):draw.text((840,382+k*28),value,font=body,fill='#243b53')
            gx,gy,gw,gh=842,584,326,72
            def gypos(h):return gy+gh-(np.clip(h,hmin,hmax)-hmin)/(hmax-hmin)*gh
            draw.rectangle((gx,gy,gx+gw,gy+gh),outline='#a8b6c4')
            draw.line((gx,gypos(target),gx+gw,gypos(target)),fill='#239d73',width=2)
            chosen=graph_indices[graph_indices<=index]
            if len(chosen)>1:
                draw.line([(gx+i/(len(states)-1)*gw,gypos(heights[i])) for i in chosen],fill='#2873b8',width=2)
            draw.text((gx,gy+gh+4),'Lift height: blue = actual, green = requested',font=small,fill='#52677a')
            draw.text((20,665),f"Final: PE {row[FIELDS[0]]:.3f} m | AE {row[FIELDS[1]]:.3f} rad | AAE {row[FIELDS[2]]:.3f} rad | Height reached: {row['lift_target_success']}",font=body,fill='#17324d')
            draw.text((20,695),'Saved physical states + URDF visual meshes; same sequence and classification as the quantitative evaluation.',font=small,fill='#52677a')
            if number==len(indices)//2:frame.save(preview)
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close();error=encoder.stderr.read().decode();code=encoder.wait()
        if code:raise RuntimeError(error)
        tmp.replace(video)
    except BaseException:
        encoder.kill();encoder.wait();raise
    finally:renderer.delete()
    info=probe(video)
    if abs(info['duration']-duration)>1/fps+.02:raise ValueError('Video duration does not match simulation')
    entry=dict(episode_id=row['episode_id'],object=row['object'],profile=row['profile'],outcome=outcome,
               sample_index=row['sample_index'],video=str(video.relative_to(root/'videos')),
               preview=str(preview.relative_to(root/'videos')),trajectory_path=row['trajectory_path'],
               trajectory_sha256=row['trajectory_sha256'],video_sha256=sha256(video),
               SucT=row['ood_goal_success'],paper_SucT=row['paper_success'],
               PE=row[FIELDS[0]],AE=row[FIELDS[1]],AAE=row[FIELDS[2]],
               lift_target_success=row['lift_target_success'],**info,
               renderer_source_sha256=sha256(__file__),
               renderer='Recorded-state URDF visual meshes, EGL/Pyrender; no simulation rerun')
    metadata.write_text(json.dumps(entry,indent=2)+'\n')
    return entry


def write_gallery(root,entries,shortages,errors,completed,planned):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    manifest=dict(entries=entries,shortages=shortages,errors=errors,completed=completed,planned=planned,
                  renderer_source_sha256=sha256(__file__))
    (root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    data=json.dumps(manifest,ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OOD 成功・失敗動画</title>
<style>body{font:16px system-ui,sans-serif;margin:0;background:#f5f7fa;color:#213047}main{max-width:1400px;margin:auto;padding:28px}p{line-height:1.7}select{font:inherit;padding:8px;margin:6px}#grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:20px}.card{background:white;border:1px solid #d5dee9;border-radius:8px;overflow:hidden}video{width:100%;background:#eee}.info{padding:16px;line-height:1.7}a{color:#175aa9}table{border-collapse:collapse}th,td{text-align:left;padding:8px;border-bottom:1px solid #d5dee9}.bad{color:#a43d32}</style>
<main><h1>5条件の成功・失敗動画</h1><p><a href="../report.html">評価結果へ戻る</a> · <a href="manifest.json">動画と出典の記録</a></p><p>全100シーケンスの評価済み軌跡から、各物体・各条件で成功と失敗を3本ずつまで選択。Suc.Tは高さの達成も必須です。<br>保存済みの実測状態と実際のURDF形状を再生しています。追加のシミュレーションや動作補間は行っていません。</p><p id="status"></p>
<select id="obj"><option value="">全物体</option></select><select id="task"><option value="">全条件</option></select><select id="outcome"><option value="">成功・失敗</option><option value="success">成功</option><option value="failure">失敗</option></select><span id="count"></span><div id="grid"></div><h2>3本に満たない組</h2><p>成功・失敗の実数が足りない場合は、追加試行で補充せず不足数を表示します。数値異常の試行は動画から除外しています。</p><table><thead><tr><th>物体</th><th>条件</th><th>成功／失敗</th><th>得られた例</th><th>不足</th></tr></thead><tbody id="short"></tbody></table></main>
<script>const d=DATA,labels={horizontal:'水平',height:'高さ',orientation:'向き',articulation:'関節角度',all:'全て'},obj=document.getElementById('obj'),task=document.getElementById('task'),out=document.getElementById('outcome'),grid=document.getElementById('grid');const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));for(const x of [...new Set(d.entries.map(e=>e.object))].sort())obj.add(new Option(x,x));for(const [k,v]of Object.entries(labels))task.add(new Option(v,k));document.getElementById('status').textContent=`${d.completed}/${d.planned}試行完了、動画${d.entries.length}本、描画エラー${d.errors.length}件`;function render(){document.querySelectorAll('video').forEach(v=>v.pause());const es=d.entries.filter(e=>(!obj.value||obj.value===e.object)&&(!task.value||task.value===e.profile)&&(!out.value||out.value===e.outcome));document.getElementById('count').textContent=es.length+'本';grid.innerHTML=es.map(e=>`<article class="card"><video controls preload="none" poster="${esc(e.preview)}" src="${esc(e.video)}"></video><div class="info"><b>${esc(e.object)} · ${labels[e.profile]} · ${e.outcome==='success'?'成功':'失敗'}</b><br>sample ${e.sample_index} · ${e.duration.toFixed(2)}秒<br>PE ${e.PE.toFixed(3)} m ／ AE ${e.AE.toFixed(3)} rad ／ AAE ${e.AAE.toFixed(3)} rad<br><a href="${esc(e.video)}">MP4</a></div></article>`).join('');document.getElementById('short').innerHTML=d.shortages.filter(e=>(!obj.value||obj.value===e.object)&&(!task.value||task.value===e.profile)).map(e=>`<tr><td>${esc(e.object)}</td><td>${labels[e.profile]}</td><td>${e.outcome==='success'?'成功':'失敗'}</td><td>${e.available}</td><td class="bad">${e.shortage}</td></tr>`).join('')}for(const el of[obj,task,out])el.onchange=render;render();</script></html>'''
    (root/'index.html').write_text(page.replace('DATA',data))


def load_rows(path):
    if not path.exists():return []
    text=path.read_text()
    # A numerical worker may be appending the last row while following a run.
    lines=text.splitlines()
    if text and not text.endswith('\n'):lines=lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,required=True)
    ap.add_argument('--resource',type=Path,default=Path(__file__).resolve().parents[1]/'rsc')
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--fps',type=float,default=20)
    ap.add_argument('--per-outcome',type=int,default=3)
    ap.add_argument('--follow',action='store_true')
    args=ap.parse_args()
    root=args.results.resolve();plan=json.loads((root/'plan.json').read_text())
    entries={};errors=[];attempted=set();futures={}
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        while True:
            rows=load_rows(root/'episodes.jsonl');shortages=[]
            for obj in plan['objects']:
                for profile in plan['profiles']:
                    group=[r for r in rows if r['object']==obj and r['profile']==profile]
                    if len(group)!=plan['samples_per_object']:continue
                    for outcome,flag in [('success',True),('failure',False)]:
                        selected=sorted([r for r in group if r['valid'] and r['ood_goal_success']==flag],key=lambda r:r['sample_index'])
                        if len(selected)<args.per_outcome:
                            shortages.append(dict(object=obj,profile=profile,outcome=outcome,available=len(selected),shortage=args.per_outcome-len(selected)))
                        for row in selected[:args.per_outcome]:
                            if row['episode_id'] not in attempted:
                                attempted.add(row['episode_id'])
                                future=pool.submit(render_one,(str(root),str(args.resource.resolve()),row,args.fps))
                                futures[future]=row['episode_id']
            done,_=wait(list(futures),timeout=1,return_when=FIRST_COMPLETED) if futures else (set(),set())
            for future in done:
                episode=futures.pop(future)
                try:
                    e=future.result();entries[episode]=e
                    print('VIDEO',episode,'total',len(entries),flush=True)
                except Exception as error:
                    errors.append(dict(episode_id=episode,error=str(error)))
                    print('VIDEO ERROR',episode,str(error),flush=True)
            write_gallery(root/'videos',sorted(entries.values(),key=lambda e:(e['object'],e['profile'],e['outcome'],e['sample_index'])),shortages,errors,len(rows),len(plan['episodes']))
            if not futures and (not args.follow or len(rows)==len(plan['episodes'])):break
            if not futures:time.sleep(15)
    print('VIDEOS DONE',len(entries),'errors',len(errors),'gallery',root/'videos/index.html',flush=True)
    return 1 if errors else 0


if __name__=='__main__':sys.exit(main())
