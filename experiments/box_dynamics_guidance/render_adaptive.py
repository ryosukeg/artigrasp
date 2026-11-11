#!/usr/bin/env python3
"""Render six saved adaptive rollouts; never run the policy or physics again."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from render_threeway import HERE, REPO, METHODS, LABELS, ENGLISH, FIELDS, camera_for, sha, save_json

DT, START = .01, 1260
COLORS = ('#475569', '#c66a23', '#2563eb')
REASONS = {'success_hold':'SUCCESS HOLD', 'stalled':'OBSERVED STALL', 'timeout':'TIME LIMIT', 'invalid':'INVALID'}
CASE_IDS = ('box__normal__0017', 'box__normal__0086', 'box__ood_articulation__0019',
            'box__ood_articulation__0041', 'box__ood_articulation__0037', 'box__ood_articulation__0053')


def render_case(root, output, episode, rows, fps, preview, hashes, settings, render, pyrender, device):
    import numpy as np
    from PIL import Image, ImageDraw
    started = time.monotonic()
    data = []
    for row in rows:
        path = root / row['trajectory_path']
        if not row['valid'] or sha(path) != row['trajectory_sha256']:
            raise ValueError('Invalid/changed source ' + str(path))
        stop = START + row['termination']['opening_step']
        with np.load(path, allow_pickle=False) as z:
            states, opening, goal = z['states'].copy(), z['final_states'].copy(), z['goal'].copy()
        if states.shape != (stop+1, 138) or opening.shape != (stop-START+1, 138) or not np.isfinite(states).all():
            raise ValueError('Unexpected variable state layout')
        np.testing.assert_array_equal(states[START+1:], opening[1:])
        states[START] = opening[0]
        data.append(dict(states=states, goal=goal, stop=stop))
    for d in data[1:]:
        np.testing.assert_array_equal(d['states'][:START+1], data[0]['states'][:START+1])
        np.testing.assert_array_equal(d['goal'], data[0]['goal'])
    maximum = max(d['stop'] for d in data)
    count = int(math.ceil(maximum * fps / 100))
    indices = np.minimum(np.rint(np.arange(count) * 100 / fps).astype(int), maximum)
    indices[-1] = maximum
    method_indices = {m:np.minimum(indices,d['stop']).tolist() for m,d in zip(METHODS,data)}
    terminations = {m:r['termination'] for m,r in zip(METHODS,rows)}
    key = dict(source_hashes=hashes, settings=settings,
               trajectories={m:r['trajectory_sha256'] for m,r in zip(METHODS,rows)}, terminations=terminations)
    video, poster = output/'videos'/(episode+'.mp4'), output/'posters'/(episode+'.png')
    terminal, metadata = output/'posters'/(episode+'_terminal.png'), video.with_suffix('.json')
    if not preview and all(p.exists() for p in (video,poster,terminal,metadata)):
        old=json.loads(metadata.read_text())
        if (old.get('cache_key')==key and old.get('video_sha256')==sha(video)
                and old.get('poster_sha256')==sha(poster) and old.get('terminal_image_sha256')==sha(terminal)):
            render.probe(video)
            print('Cached '+episode,flush=True)
            return old
    anchors=np.concatenate([d['states'][:,sl] for d in data for sl in (slice(122,125),slice(6,9),slice(64,67))]+[data[0]['goal'][None,:3]])
    overview=camera_for(anchors,[1.2,-1.7,1.2],1.6,render)
    scenes=[render.make_scene(str(REPO/'rsc'),r) for r in rows]
    renderer=pyrender.OffscreenRenderer(640,400)
    title,body,small=render.font(23),render.font(19),render.font(16)
    angles=np.concatenate([d['states'][:,129] for d in data]+[data[0]['goal'][7:8]])
    amin,amax=min(-.05,float(angles.min())-.05),max(.65,float(angles.max())+.15)
    goal=data[0]['goal']; image_cache={}

    def panel_images(k, state_step):
        # Same pose/meshes/materials and common cameras before divergence.
        cache_id=0 if state_step<=START else k
        if cache_id in image_cache and image_cache[cache_id][0]==state_step:
            return image_cache[cache_id][1:]
        scene,models,nodes,camera=scenes[k]
        state=data[k]['states'][state_step]; actual=state[122:130]
        models[0].update(scene,nodes[0],render.pose(state[6:9],state[9:13]),state[19:64])
        models[1].update(scene,nodes[1],render.pose(state[64:67],state[67:71]),state[77:122])
        models[2].update(scene,nodes[2],render.pose(actual[:3],actual[3:7]),[actual[7]])
        models[3].update(scene,nodes[3],render.pose(goal[:3],goal[3:7]),[goal[7]])
        scene.set_pose(camera,overview)
        wide,_=renderer.render(scene,flags=pyrender.RenderFlags.SKIP_CULL_FACES)
        center=models[2].world_bounds.mean(axis=0)
        scene.set_pose(camera,render.look_at(center,[-1.1,-1.5,1.1],.70))
        close,_=renderer.render(scene,flags=pyrender.RenderFlags.SKIP_CULL_FACES)
        images=(Image.fromarray(close),Image.fromarray(wide).resize((320,200)))
        image_cache[cache_id]=(state_step,)+images
        return images

    def frame_at(common_step):
        frame=Image.new('RGB',(1920,880),'#f3f5f8'); draw=ImageDraw.Draw(frame)
        draw.text((18,12),'{}  |  timeline {:.2f} / {:.2f} s  |  target {:.3f} rad'.format(episode,common_step*DT,maximum*DT,float(goal[7])),font=title,fill='#1b2e44')
        draw.text((18,46),'Opening starts at 12.60 s. STOPPED panels hold their last RECORDED state; no measured motion after stop.',font=small,fill='#4d6074')
        for k,(row,d) in enumerate(zip(rows,data)):
            x=k*640; step=min(common_step,d['stop']); stopped=common_step>=d['stop']
            actual=d['states'][step,122:130].astype(np.float64)
            close,wide=panel_images(k,step)
            draw.rectangle((x+8,80,x+631,119),fill=COLORS[k]); draw.text((x+20,87),ENGLISH[k],font=title,fill='white')
            frame.paste(close,(x,126)); frame.paste(wide,(x+8,555))
            if stopped:
                draw.rectangle((x+10,478,x+628,524),fill='#25394f')
                draw.text((x+20,491),'STOPPED: '+REASONS[row['termination']['reason']]+' | t={:.2f}s'.format(step*DT),font=body,fill='white')
            draw.text((x+14,532),'State t={:.2f}s | '.format(step*DT)+('held frame after end' if stopped else 'recorded motion'),font=small,fill='#4d6074')
            draw.text((x+14,759),'Overview: same fixed camera',font=small,fill='#4d6074')
            gx,gy,gw,gh=x+350,580,265,130
            def point(t,a):return gx+t/maximum*gw,gy+gh-(a-amin)/(amax-amin)*gh
            draw.rectangle((gx,gy,gx+gw,gy+gh),outline='#9eadbd')
            draw.line((*point(START,amin),*point(START,amax)),fill='#b7c3cf',width=2)
            draw.line((*point(0,goal[7]),*point(maximum,goal[7])),fill='#15956e',width=2)
            chosen=np.unique(np.append(np.arange(0,step+1,5),step))
            if len(chosen)>1:draw.line([point(int(i),float(d['states'][i,129])) for i in chosen],fill=COLORS[k],width=3)
            px,py=point(step,actual[7]);draw.ellipse((px-3,py-3,px+3,py+3),fill=COLORS[k])
            draw.line((*point(step,amin),*point(step,amax)),fill='#293c53',width=1)
            draw.text((gx,gy-24),'Lid angle / green: target',font=small,fill='#4d6074')
            draw.text((gx,gy+gh+8),'0s',font=small,fill='#4d6074')
            draw.text((gx+gw*START/maximum-18,gy+gh+8),'12.6',font=small,fill='#4d6074')
            draw.text((gx+gw-40,gy+gh+8),'{:.2f}s'.format(maximum*DT),font=small,fill='#4d6074')
            draw.text((gx,gy+gh+31),'now {:.3f} / {:.3f} rad'.format(actual[7],float(goal[7])),font=small,fill='#263b52')
            g=goal.astype(np.float64);pe=float(np.linalg.norm(actual[:3]-g[:3]));aae=abs(actual[7]-g[7])
            dot=abs(float(np.dot(actual[3:7]/np.linalg.norm(actual[3:7]),g[3:7]/np.linalg.norm(g[3:7]))))
            ae=2*math.acos(min(1.,dot))
            draw.text((x+16,791),'State: PE {:.3f}m | AAE {:.3f} | AE {:.3f}rad'.format(pe,aae,ae),font=body,fill='#25394f')
            draw.text((x+16,824),'End: '+REASONS[row['termination']['reason']]+' | opening {:.2f}s'.format(row['termination']['opening_seconds']),font=body,fill='#18715a' if row['termination']['reason']=='success_hold' else '#a1473b')
            if k:draw.line((x,80,x,854),fill='#bdc8d2',width=2)
        draw.text((18,858),'Saved physical states + actual URDF meshes. Detail follows each object at the same scale. No policy / physics rerun.',font=small,fill='#52677a')
        return frame

    encoder=None;partial=video.with_suffix('.partial.mp4')
    try:
        if preview:
            frame_at(min(1600,maximum)).save(poster);frame_at(maximum).save(terminal)
            print('Preview '+str(poster),flush=True);return None
        encoder=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x880','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(partial)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        poster_index=int(np.argmin(np.abs(indices-min(1600,maximum))))
        for number,index in enumerate(indices):
            frame=frame_at(int(index))
            if number==poster_index:frame.save(poster)
            if number==len(indices)-1:frame.save(terminal)
            encoder.stdin.write(frame.tobytes())
            if (number+1)%80==0:print('Render {}: {}/{} frames, {:.1f}s'.format(episode,number+1,len(indices),time.monotonic()-started),flush=True)
        encoder.stdin.close();error=encoder.stderr.read().decode();code=encoder.wait()
        if code:raise RuntimeError(error)
        partial.replace(video)
    except BaseException:
        if encoder is not None and encoder.poll() is None:encoder.kill();encoder.wait()
        raise
    finally:renderer.delete()
    encoded=render.probe(video)
    if encoded['frames']!=count or abs(encoded['duration']-count/fps)>.001:raise ValueError('Encoded frame/duration mismatch')
    condition='normal' if rows[0]['profile']=='normal' else ('OOD・小角' if goal[7]<.5 else 'OOD・大角')
    entry=dict(episode_id=episode,profile=rows[0]['profile'],title=condition+' / '+episode.rsplit('__',1)[-1],
        selection_reason='旧固定時間版と同じ試行を採用。新しい適応終了の成否で再選抜していません。6例は頻度を代表する標本ではありません。',
        video=str(video.relative_to(output)),poster=str(poster.relative_to(output)),terminal_image=str(terminal.relative_to(output)),
        video_sha256=sha(video),poster_sha256=sha(poster),terminal_image_sha256=sha(terminal),
        duration=maximum*DT,fps=fps,opening_start=12.6,target_angle_rad=float(goal[7]),terminations=terminations,
        frame_state_indices=indices.tolist(),method_frame_state_indices=method_indices,
        metrics=[dict(method=m,label=LABELS[k],PE=r[FIELDS[0]],AAE=r[FIELDS[1]],AE=r[FIELDS[2]],paper_success=r['paper_success'],composite_success=r['ood_goal_success']) for k,(m,r) in enumerate(zip(METHODS,rows))],
        cache_key=key,encoded=encoded,render_seconds=time.monotonic()-started,rendering_device=device,
        overview_camera_pose=overview.tolist(),detail_camera_distance_m=.70,
        rendering_note='Synchronized common timeline. Each column clamps every pose, camera, metric and curve to its own recorded stop; STOPPED status marks frozen frames. Exact post-reset state at 12.6s. Exact terminal states included.')
    save_json(metadata,entry);print('Saved {} ({:.1f}s)'.format(video,entry['render_seconds']),flush=True)
    return entry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=HERE/'runs/comparison_seed202_adaptive_v1')
    p.add_argument('--fps',type=int,default=20);p.add_argument('--limit',type=int);p.add_argument('--preview',action='store_true')
    args=p.parse_args()
    if args.fps!=20 or (args.limit is not None and args.limit<1):p.error('This report uses 20fps and positive limit')
    root=args.results.resolve();output=root/'qualitative'
    result=json.loads((root/'results.json').read_text());audit=json.loads((root/'independent_audit.json').read_text());plan=json.loads((root/'plan.json').read_text())
    if not result['complete'] or audit['status']!='completed' or audit['results_sha256']!=sha(root/'results.json') or result['plan_sha256']!=sha(root/'plan.json'):raise ValueError('Current audited results required')
    hashes={str(Path(__file__).resolve().relative_to(REPO)):sha(__file__),
            'experiments/box_dynamics_guidance/render_threeway.py':sha(HERE/'render_threeway.py'),
            'scripts/render_ood_five.py':sha(REPO/'scripts/render_ood_five.py'),
            'plan.json':sha(root/'plan.json'),'results.json':sha(root/'results.json'),'independent_audit.json':sha(root/'independent_audit.json')}
    assets={path:digest for path,digest in audit['verified_source_and_input_sha256'].items() if '/rsc/' in path}
    if not assets:raise ValueError('No audited rendering assets found')
    for path,digest in assets.items():
        if sha(path)!=digest:raise ValueError('Changed rendering asset '+path)
    hashes.update(assets)
    groups={}
    for row in result['episodes']:groups.setdefault(row['episode_id'],{})[row['method']]=row
    old_manifest=Path(plan['source_run'])/'qualitative/manifest.json'
    old=json.loads(old_manifest.read_text())
    if [c['episode_id'] for c in old['cases']]!=list(CASE_IDS):raise ValueError('Previous illustrative cases differ')
    hashes['previous_qualitative_manifest']=sha(old_manifest)
    for directory in ('videos','posters'):(output/directory).mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(HERE/'.render_deps'));sys.path.insert(0,str(REPO/'scripts'))
    os.environ['PYOPENGL_PLATFORM']='egl'
    settings=dict(fps=args.fps,width=1920,height=880,frame_mapping='common step, clamp per method; exact last state',
                  overview='same fixed camera',detail='object-following same orientation and .70m distance',
                  stopped_display='frozen terminal state explicitly labeled STOPPED; no extrapolation',
                  opening_state='post-reset final_states[0] at global step1260')
    entries=[]
    with tempfile.TemporaryDirectory(prefix='box_adaptive_egl_') as temporary:
        driver=Path('/.singularity.d/libs/libEGL_nvidia.so.0')
        if driver.is_file() and '__EGL_VENDOR_LIBRARY_FILENAMES' not in os.environ:
            vendor=Path(temporary)/'nvidia.json';vendor.write_text(json.dumps({'file_format_version':'1.0.0','ICD':{'library_path':str(driver)}}));os.environ['__EGL_VENDOR_LIBRARY_FILENAMES']=str(vendor)
        import pyglet
        pyglet.options['headless']=True;pyglet.options['shadow_window']=False
        import pyrender
        from OpenGL.GL import glGetString,GL_RENDERER
        import render_ood_five as render
        probe=pyrender.OffscreenRenderer(64,64);device=glGetString(GL_RENDERER).decode();probe.delete()
        print('Renderer: '+device,flush=True)
        for episode in CASE_IDS[:args.limit]:
            rows=[groups[episode][m] for m in METHODS]
            entry=render_case(root,output,episode,rows,args.fps,args.preview,hashes,settings,render,pyrender,device)
            if entry:entries.append(entry)
            if not args.preview:
                manifest=dict(schema_version=1,status='complete' if len(entries)==len(CASE_IDS) else 'partial',
                    created_at_utc=datetime.now(timezone.utc).isoformat(),source_plan_sha256=sha(root/'plan.json'),
                    source_results_sha256=sha(root/'results.json'),source_audit_sha256=sha(root/'independent_audit.json'),
                    source_sha256=hashes,settings=settings,cases=entries,planned_cases=list(CASE_IDS),
                    simulation_performed=False,selection_note='Same six examples as the fixed-time gallery; not reselected using adaptive outcomes.')
                save_json(output/'manifest.json',manifest)
    if not args.preview:print('Media manifest: '+str(output/'manifest.json'),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
