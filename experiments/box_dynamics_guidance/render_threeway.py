#!/usr/bin/env python3
"""Render matched three-way box trials from audited states, without simulation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
METHODS = ('baseline', 'guided_absolute', 'guided_delta')
LABELS = ('通常のArtiGrasp', '元モデルによる補正', '差分モデルによる補正')
ENGLISH = ('ArtiGrasp / PPO', 'PPO + absolute model', 'PPO + delta model')
COLORS = ('#3468ad', '#bd6b20', '#8a55b3')
FIELDS = ('paper_position_error_m', 'paper_articulation_error_rad', 'paper_orientation_error_rad')
START, STEPS, DT = 1260, 1560, .01


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def save_json(path, data):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def select_cases(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row['episode_id'], {})[row['method']] = row
    if any(set(v) != set(METHODS) for v in groups.values()):
        raise ValueError('Incomplete three-way trial groups')
    def subset(profile, side=None):
        return [key for key, g in groups.items() if g['baseline']['profile'] == profile
                and (side is None or (g['baseline']['target_articulation_rad'] < .5) == (side == 'small'))]
    def median(ids, value):
        ordered = sorted(ids, key=lambda key: (value(groups[key]), key))
        if not ordered:
            raise ValueError('No trial available for an illustrative stratum')
        rank = (len(ordered)-1)//2
        return ordered[rank], len(ordered), rank+1
    records = []
    for profile, side in [('normal', None), ('ood_articulation', 'large')]:
        for gain in (True, False):
            ids = [key for key in subset(profile, side)
                   if groups[key]['baseline']['paper_success'] == (not gain)
                   and groups[key]['guided_delta']['paper_success'] == gain]
            def change(g):
                d = g['baseline'][FIELDS[1]] - g['guided_delta'][FIELDS[1]]
                return d if gain else -d
            key, n, rank = median(ids, change)
            text = ('差分補正で失敗→成功' if gain else '差分補正で成功→失敗')
            records.append(dict(episode_id=key, profile=profile, ood_side=side,
                                title=('normal' if profile == 'normal' else 'OOD・大角')+'：'+text,
                                selection_reason='PPO→差分補正で'+('成功を獲得' if gain else '成功を喪失')+
                                'した{}件から、AAEの{}量が下側中央値（{}番目）の例。'.format(n, '改善' if gain else '悪化', rank)))
    def relative_error(g):
        return sum(r[FIELDS[0]]/.05 + r[FIELDS[1]]/.5 + r[FIELDS[2]]/.2 for r in g.values())/3
    for side, title in [('small', 'OOD・小角：角度条件と基部の安定性'), ('large', 'OOD・大角：3方式に共通する失敗')]:
        ids = subset('ood_articulation', side)
        if side == 'large':
            ids = [key for key in ids if not any(r['paper_success'] for r in groups[key].values())]
        key, n, rank = median(ids, relative_error)
        records.append(dict(episode_id=key, profile='ood_articulation', ood_side=side, title=title,
                            selection_reason='{}{}件から、3方式平均の PE/0.05 + AAE/0.5 + AE/0.2 が下側中央値（{}番目）の例。この値は例の選定専用。'.format('小角OODの' if side == 'small' else '3方式共通失敗の', n, rank)))
    order = {'box__normal__0017': 0, 'box__normal__0086': 1, 'box__ood_articulation__0019': 2,
             'box__ood_articulation__0041': 3, 'box__ood_articulation__0037': 4, 'box__ood_articulation__0053': 5}
    records.sort(key=lambda r: (order.get(r['episode_id'], 6), r['episode_id']))
    notes = {
        'box__normal__0017': '両補正で基部位置・姿勢を保ち、最終3条件を達成する例です。',
        'box__normal__0086': '基部は安定していますが、補正後は蓋がほとんど開かず角度条件が未達になります。',
        'box__ood_articulation__0019': '開閉開始時から角度誤差は0.5 rad未満です。角度条件を満たすことと、蓋が十分に動くことは区別してください。終端では基部位置・姿勢が未達です。',
        'box__ood_articulation__0041': '両補正で位置・姿勢・角度の誤差が減ります。角度0.1 rad未満の厳しい条件は未達です。',
        'box__ood_articulation__0037': '差分補正のAAEは0.500646 radで、0.5 rad閾値を約0.000646 rad超えています。成功ラベルだけで動作差を過大に解釈せず、角度曲線も確認してください。',
        'box__ood_articulation__0053': '3方式とも蓋はほぼ閉じたままです。開閉開始時から基部の姿勢誤差も0.2 radを超えています。',
    }
    for r in records:
        r['selection_reason'] += ' ' + notes.get(r['episode_id'], '')
        r['condition_label'] = 'normal' if r['profile'] == 'normal' else ('OOD・小角' if r['ood_side'] == 'small' else 'OOD・大角')
    return records, groups


def camera_for(anchors, direction, aspect, render):
    import numpy as np
    low, high = anchors.min(axis=0)-.13, anchors.max(axis=0)+.13
    center = (low+high)/2
    axes = render.look_at(center, direction, 1)[:3, :3]
    projected = np.abs(axes).T @ ((high-low)/2)
    distance = (max(projected[0]/(math.tan(math.pi/8)*aspect), projected[1]/math.tan(math.pi/8)) + projected[2])*1.1
    return render.look_at(center, direction, max(.8, float(distance)))


def render_case(output, case, group, fps, preview_only, source_hashes, settings, render, pyrender, renderer_name):
    import numpy as np
    from PIL import Image, ImageDraw
    started = time.monotonic()
    episode = case['episode_id']
    rows = [group[m] for m in METHODS]
    data = []
    for row in rows:
        path = output.parent/row['trajectory_path']
        if not row['valid'] or sha(path) != row['trajectory_sha256']:
            raise ValueError('Invalid or changed source trajectory '+str(path))
        with np.load(path, allow_pickle=False) as z:
            states = z['states'].copy()
            final_states = z['final_states'].copy()
            residuals = z['residuals'].copy()
            goal = z['goal'].copy()
        if states.shape != (1561,138) or final_states.shape != (301,138) or not np.isfinite(states).all():
            raise ValueError('Bad recorded state layout')
        np.testing.assert_array_equal(states[START+1:], final_states[1:])
        # Display the post-reset state at the exact start of final articulation.
        states[START] = final_states[0]
        data.append(dict(states=states, residuals=residuals, goal=goal))
    for d in data[1:]:
        np.testing.assert_array_equal(d['states'][:START+1], data[0]['states'][:START+1])
        np.testing.assert_array_equal(d['goal'], data[0]['goal'])
    indices = np.minimum(np.rint(np.arange(round(STEPS*DT*fps))/(fps*DT)).astype(int), STEPS)
    indices[-1] = STEPS
    key = dict(source_hashes=source_hashes, settings=settings,
               trajectories={r['method']:r['trajectory_sha256'] for r in rows})
    video = output/'videos'/(episode+'.mp4')
    metadata = video.with_suffix('.json')
    poster = output/'posters'/(episode+'.png')
    terminal = output/'posters'/(episode+'_terminal.png')
    if not preview_only and metadata.exists() and video.exists() and poster.exists() and terminal.exists():
        old = json.loads(metadata.read_text())
        if old.get('cache_key') == key and old.get('video_sha256') == sha(video) and old.get('poster_sha256') == sha(poster) and old.get('terminal_image_sha256') == sha(terminal):
            render.probe(video)
            print('Cached', episode, flush=True)
            return old
    anchors = np.concatenate([d['states'][:, sl] for d in data for sl in (slice(122,125),slice(6,9),slice(64,67))] + [data[0]['goal'][None,:3]])
    overview = camera_for(anchors, [1.2,-1.7,1.2], 1.6, render)
    scenes = [render.make_scene(str(REPO/'rsc'), r) for r in rows]
    renderer = pyrender.OffscreenRenderer(640,400)
    font_title, font_body, font_small = render.font(23), render.font(19), render.font(16)
    angles = np.concatenate([d['states'][:,129] for d in data] + [data[0]['goal'][7:8]])
    amin, amax = min(-.05, float(angles.min())-.05), max(.65, float(angles.max())+.15)
    graph_steps = np.linspace(0,STEPS,313).astype(int)
    goal = data[0]['goal']
    # Fixed magnification for all models and frames; center follows each object.
    detail_distance = .70
    def frame_at(index):
        frame = Image.new('RGB',(1920,880),'#f3f5f8')
        draw = ImageDraw.Draw(frame)
        draw.text((18,12), '{}   |   t = {:.2f} / 15.60 s   |   target lid = {:.3f} rad'.format(episode,index*DT,float(goal[7])),font=font_title,fill='#1b2e44')
        phase = 'Final articulation / guidance active' if index >= START else render.phase(index,rows[0]['timing'])
        draw.text((18,46), phase+'   |   right hand: coral   left: blue   box: gold   final target: translucent green',font=font_small,fill='#4d6074')
        for k,(row,d,parts) in enumerate(zip(rows,data,scenes)):
            scene, models, nodes, camera = parts
            x = k*640
            state = d['states'][index]; actual = state[122:130]
            models[0].update(scene,nodes[0],render.pose(state[6:9],state[9:13]),state[19:64])
            models[1].update(scene,nodes[1],render.pose(state[64:67],state[67:71]),state[77:122])
            models[2].update(scene,nodes[2],render.pose(actual[:3],actual[3:7]),[actual[7]])
            models[3].update(scene,nodes[3],render.pose(goal[:3],goal[3:7]),[goal[7]])
            scene.set_pose(camera,overview)
            wide,_ = renderer.render(scene,flags=pyrender.RenderFlags.SKIP_CULL_FACES)
            center = models[2].world_bounds.mean(axis=0)
            scene.set_pose(camera,render.look_at(center,[-1.1,-1.5,1.1],detail_distance))
            close,_ = renderer.render(scene,flags=pyrender.RenderFlags.SKIP_CULL_FACES)
            draw.rectangle((x+8,80,x+631,119),fill=COLORS[k])
            draw.text((x+20,87),ENGLISH[k],font=font_title,fill='white')
            frame.paste(Image.fromarray(close),(x,126))
            frame.paste(Image.fromarray(wide).resize((320,200)),(x+8,555))
            draw.text((x+14,532),'Detail: tracking, same magnification',font=font_small,fill='#4d6074')
            draw.text((x+14,759),'Overview: same fixed camera',font=font_small,fill='#4d6074')
            gx,gy,gw,gh = x+350,580,265,130
            def point(step, angle):
                return (gx+step/STEPS*gw,gy+gh-(angle-amin)/(amax-amin)*gh)
            draw.rectangle((gx,gy,gx+gw,gy+gh),outline='#9eadbd')
            draw.line((*point(START,amin),*point(START,amax)),fill='#b7c3cf',width=2)
            draw.line((*point(0,goal[7]),*point(STEPS,goal[7])),fill='#15956e',width=2)
            chosen = graph_steps[graph_steps <= index]
            if len(chosen)>1:
                draw.line([point(i,d['states'][i,129]) for i in chosen],fill=COLORS[k],width=3)
            draw.line((*point(index,amin),*point(index,amax)),fill='#293c53',width=1)
            draw.text((gx,gy-24),'Lid angle / green: target',font=font_small,fill='#4d6074')
            draw.text((gx,gy+gh+8),'0 s',font=font_small,fill='#4d6074')
            draw.text((gx+gw*START/STEPS-18,gy+gh+8),'12.6',font=font_small,fill='#4d6074')
            draw.text((gx+gw-23,gy+gh+8),'15.6',font=font_small,fill='#4d6074')
            draw.text((gx,gy+gh+31),'now {:.3f} / {:.3f} rad'.format(float(actual[7]),float(goal[7])),font=font_small,fill='#263b52')
            pe = float(np.linalg.norm(actual[:3]-goal[:3]))
            q = actual[3:7]/np.linalg.norm(actual[3:7]); qg=goal[3:7]/np.linalg.norm(goal[3:7])
            ae = 2*math.acos(min(1.,abs(float(np.dot(q,qg)))))
            aae = abs(float(actual[7])-float(goal[7]))
            draw.text((x+16,791),'Now: PE {:.3f} m   AAE {:.3f}   AE {:.3f} rad'.format(pe,aae,ae),font=font_body,fill='#25394f')
            draw.text((x+16,824),'Final 3 conditions: '+('PASS' if row['paper_success'] else 'FAIL')+'   |   with lift: '+('PASS' if row['ood_goal_success'] else 'FAIL'),font=font_body,fill='#18715a' if row['paper_success'] else '#a1473b')
            if k:
                draw.line((x,80,x,854),fill='#bdc8d2',width=2)
        draw.text((18,858),'Audited recorded-state playback + actual URDF meshes. Final articulation starts at 12.6 s. No policy / physics rerun.',font=font_small,fill='#52677a')
        return frame
    encoder = None
    partial = video.with_suffix('.partial.mp4')
    try:
        if preview_only:
            frame_at(1400).save(poster)
            frame_at(STEPS).save(terminal)
            print('Preview',poster,flush=True)
            return None
        encoder = subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x880','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(partial)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        poster_number = int(np.argmin(np.abs(indices-1400)))
        for number,index in enumerate(indices):
            frame = frame_at(int(index))
            if number == poster_number: frame.save(poster)
            if number == len(indices)-1: frame.save(terminal)
            encoder.stdin.write(frame.tobytes())
            if (number+1)%40 == 0:
                print('Render {}: {}/{} frames, {:.1f}s'.format(episode,number+1,len(indices),time.monotonic()-started),flush=True)
        encoder.stdin.close()
        error=encoder.stderr.read().decode(); code=encoder.wait()
        if code: raise RuntimeError(error)
        partial.replace(video)
    except BaseException:
        if encoder is not None and encoder.poll() is None:
            encoder.kill();encoder.wait()
        raise
    finally:
        renderer.delete()
    info=render.probe(video)
    if info['frames'] != len(indices) or abs(info['duration']-15.6)>1/fps+.001:
        raise ValueError('Encoded frame count / duration mismatch')
    metrics=[dict(method=m,label=LABELS[k],PE=row[FIELDS[0]],AAE=row[FIELDS[1]],AE=row[FIELDS[2]],paper_success=row['paper_success'],composite_success=row['ood_goal_success']) for k,(m,row) in enumerate(zip(METHODS,rows))]
    entry=dict(case,video=str(video.relative_to(output)),poster=str(poster.relative_to(output)),terminal_image=str(terminal.relative_to(output)),target_angle_rad=float(goal[7]),duration=15.6,fps=fps,opening_start=12.6,metrics=metrics,
               series=dict(time_s=(graph_steps*DT).tolist(),angle_rad={m:d['states'][graph_steps,129].astype(float).tolist() for m,d in zip(METHODS,data)}),
               cache_key=key,video_sha256=sha(video),poster_sha256=sha(poster),terminal_image_sha256=sha(terminal),
               encoded=info,render_seconds=time.monotonic()-started,rendering_device=renderer_name,
               frame_state_indices=indices.tolist(),overview_camera_pose=overview.tolist(),detail_camera_distance_m=detail_distance,
               rendering_note='Same fixed overview camera; same-orientation and same-magnification object-tracking detail. Exact post-reset state at 12.6s; terminal state included in last frame. Final target ghost is constant throughout.')
    save_json(metadata,entry)
    print('Saved',video,'{:.1f}s'.format(entry['render_seconds']),flush=True)
    return entry


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=HERE/'runs/comparison_seed202_threeway_v1')
    p.add_argument('--fps',type=int,default=20)
    p.add_argument('--limit',type=int)
    p.add_argument('--preview',action='store_true')
    args=p.parse_args()
    if not 1<=args.fps<=60 or (args.limit is not None and args.limit<1): p.error('Invalid fps/limit')
    root=args.results.resolve();output=root/'qualitative'
    result=json.loads((root/'results.json').read_text());summary=json.loads((root/'summary.json').read_text());audit=json.loads((root/'independent_audit.json').read_text())
    if not result['complete'] or not summary['comparison_eligible'] or audit['status']!='completed' or audit['results_sha256']!=sha(root/'results.json'):
        raise ValueError('Complete audited comparison required')
    asset_hashes = {path:digest for path,digest in audit['verified_source_and_input_sha256'].items() if '/rsc/' in path}
    for path,digest in asset_hashes.items():
        if sha(path) != digest: raise ValueError('Changed audited rendering asset '+path)
    cases,groups=select_cases(result['episodes'])
    print('Selected',*[c['episode_id'] for c in cases],sep='\n',flush=True)
    selected=cases[:args.limit] if args.limit else cases
    for directory in ('videos','posters'): (output/directory).mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(HERE/'.render_deps'));sys.path.insert(0,str(REPO/'scripts'))
    os.environ['PYOPENGL_PLATFORM']='egl'
    settings=dict(fps=args.fps,width=1920,height=880,overview='fixed common camera for three entire trajectories',detail='object tracking, fixed orientation and distance .70m',opening_state='final_states[0] at step1260',target='constant final saved goal',source='saved states; no simulation rerun')
    source_hashes={str(Path(__file__).resolve().relative_to(REPO)):sha(__file__),'scripts/render_ood_five.py':sha(REPO/'scripts/render_ood_five.py')}
    source_hashes.update(asset_hashes)
    with tempfile.TemporaryDirectory(prefix='box_threeway_egl_') as temporary:
        driver=Path('/.singularity.d/libs/libEGL_nvidia.so.0')
        if driver.is_file() and '__EGL_VENDOR_LIBRARY_FILENAMES' not in os.environ:
            vendor=Path(temporary)/'nvidia.json'
            vendor.write_text(json.dumps({'file_format_version':'1.0.0','ICD':{'library_path':str(driver)}}))
            os.environ['__EGL_VENDOR_LIBRARY_FILENAMES']=str(vendor)
        import pyglet
        pyglet.options['headless']=True;pyglet.options['shadow_window']=False
        import pyrender
        from OpenGL.GL import glGetString, GL_RENDERER
        import render_ood_five as render
        probe=pyrender.OffscreenRenderer(64,64);device=glGetString(GL_RENDERER).decode();probe.delete()
        print('Renderer:',device,flush=True)
        entries=[]
        for case in selected:
            entry=render_case(output,case,groups[case['episode_id']],args.fps,args.preview,source_hashes,settings,render,pyrender,device)
            if entry: entries.append(entry)
    if args.preview: return 0
    summary_rows=[]
    for profile,condition in [('normal','normal'),('ood_articulation','OOD')]:
        for k,method in enumerate(METHODS):
            g=summary['groups'][profile][method]
            summary_rows.append(dict(profile=profile,condition_label=condition,method=method,label=LABELS[k],n=g['n'],paper_success_count=g['success']['paper_success']['count'],composite_success_count=g['success']['ood_goal_success']['count'],PE=g['errors'][FIELDS[0]]['mean'],AAE=g['errors'][FIELDS[1]]['mean'],AE=g['errors'][FIELDS[2]]['mean']))
    manifest=dict(schema_version=1,status='complete' if len(entries)==len(cases) else 'partial',source_run='../',created_at_utc=datetime.now(timezone.utc).isoformat(),methods=[dict(id=m,label=LABELS[k],color=COLORS[k]) for k,m in enumerate(METHODS)],summary_rows=summary_rows,cases=entries,planned_cases=cases,settings=settings,source_sha256=source_hashes,source_result_sha256=sha(root/'results.json'),source_summary_sha256=sha(root/'summary.json'),source_audit_sha256=sha(root/'independent_audit.json'),selection_note='Six stratified illustrative examples (two gained successes, two lost successes, two common failures); not a sample representing outcome frequencies.',simulation_performed=False)
    from threeway_gallery import write_gallery
    manifest['gallery_source_sha256']=sha(HERE/'threeway_gallery.py')
    save_json(output/'manifest.json',manifest)
    write_gallery(output,manifest)
    print('Gallery:',output/'index.html',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
