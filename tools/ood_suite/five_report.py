"""Compact tables for normal and the five requested OOD conditions."""

import csv
import html
import json
import math
from pathlib import Path

from .io import read_json, read_rows, write_json

LABELS = {'normal':'通常（normal）', 'horizontal':'水平', 'height':'高さ', 'orientation':'向き', 'articulation':'関節角度', 'all':'全て'}
ERRORS = {'PE':'paper_position_error_m', 'AE':'paper_orientation_error_rad', 'AAE':'paper_articulation_error_rad'}


def summarize(rows, planned):
    valid=[r for r in rows if r.get('valid')]
    result=dict(planned=planned, completed=len(rows), valid=len(valid), invalid=len(rows)-len(valid),
                successes=sum(r.get('valid') and r.get('ood_goal_success',False) for r in rows),
                paper_successes=sum(r.get('valid') and r.get('paper_success',False) for r in rows))
    for key,count in [('SucT','successes'),('paper_SucT','paper_successes')]:
        result[key]=result[count]/planned if planned else None
    for key,field in ERRORS.items():
        values=[float(r[field]) for r in valid if r.get(field) is not None and math.isfinite(r[field])]
        result[key]=sum(values)/len(values) if values else None
        result[key+'_n']=len(values)
    return result


def fmt(value, percentage=False):
    if value is None:
        return '—'
    return '{:.1%}'.format(value) if percentage else '{:.3f}'.format(value)


def generate_report(results_dir):
    root=Path(results_dir)
    plan=read_json(root/'plan.json')
    rows=read_rows(root/'episodes.jsonl')
    expected={s['episode_id'] for s in plan['episodes']}
    if any(r['episode_id'] not in expected for r in rows):
        raise ValueError('Results contain an episode outside the saved plan')
    groups=[]
    aggregates=[]
    count=plan['samples_per_object']
    for profile in plan['profiles']:
        subset=[r for r in rows if r['profile']==profile]
        aggregates.append(dict(profile=profile,**summarize(subset,count*len(plan['objects']))))
        for obj in plan['objects']:
            groups.append(dict(object=obj,profile=profile,
                               **summarize([r for r in subset if r['object']==obj],count)))
    summary=dict(completed=len(rows),planned=len(expected),complete=len(rows)==len(expected),
                 aggregates=aggregates,objects=groups,
                 primary_success='final PE<0.05m, AE<0.2rad, AAE<0.5rad AND lift height error<0.05m',
                 errors='Unclipped final-state means over all valid trials, including failures.')
    write_json(root/'summary.json',summary)
    fields=['profile','object','planned','completed','invalid','successes','SucT','PE','AE','AAE',
            'paper_successes','paper_SucT','PE_n','AE_n','AAE_n']
    with (root/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(groups)
    profiles=plan['profiles']
    has_normal='normal' in profiles
    has_ood=any(profile!='normal' for profile in profiles)
    title=('通常条件（normal）の評価' if has_normal and not has_ood else
           '{}評価：{}条件'.format('通常条件とOOD' if has_normal else 'OOD',len(profiles)))
    object_title='全{}物体'.format(len(plan['objects']))
    sampling='各物体・条件{}シーケンス。同じsample番号では参照姿勢と初期状態を共有。'.format(count)
    links=[('全{}組のCSV'.format(len(groups)),'metrics.csv'),('全試行の計画','plan.json')]
    for label,path in [('物理制約の検査記録','physics_constraints.json'),
                       ('実行環境','execution.json'),('成功・失敗動画','videos/index.html')]:
        if (root/path).is_file():
            links.append((label,path))
    lines=['# '+title, '', '**{}/{}試行完了**{}'.format(len(rows),len(expected),
           '' if summary['complete'] else '（計算中。成功率は予定数を分母とする暫定下限）'), '',
           sampling, '',
           '- **Suc.T**：最終PE < 0.05 m、AE < 0.2 rad、AAE < 0.5 radに加え、持ち上げ区間で要求上昇量との誤差 < 0.05 m。動画もこの定義で分類。',
           '- **論文Suc.T**：最終PE・AE・AAEの3条件のみ。',
           '- **PE / AE / AAE**：正常に測定できた全試行の最終誤差の平均。失敗例を含み、クリップしない。PEはm、AE/AAEはrad。異常試行は成功率では失敗とし、誤差平均の測定数をCSVに記録。', '',
           '## '+object_title+'の集計', '',
           '| 条件 | 完了/予定 | 成功数 | Suc.T | PE (m) | AE (rad) | AAE (rad) | 論文Suc.T | 異常 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def table_row(label,r):
        return '| {} | {}/{} | {} | {} | {} | {} | {} | {} | {} |'.format(
            label,r['completed'],r['planned'],r['successes'],fmt(r['SucT'],True),fmt(r['PE']),
            fmt(r['AE']),fmt(r['AAE']),fmt(r['paper_SucT'],True),r['invalid'])
    lines += [table_row(LABELS[r['profile']],r) for r in aggregates]
    for profile in plan['profiles']:
        lines += ['', '## '+LABELS[profile], '',
                  '| 物体 | 完了/予定 | 成功数 | Suc.T | PE (m) | AE (rad) | AAE (rad) | 論文Suc.T | 異常 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        lines += [table_row(r['object'],r) for r in groups if r['profile']==profile]
    lines += ['', '## 目標と物理制約', '']
    if has_normal:
        lines += ['通常条件は論文Supplementary B.4に合わせ、水平x/y各−0.10〜−0.05 m、世界Z軸yaw ±0.40 rad、絶対関節角0.50〜0.60 rad。途中の要求上昇量は0.15 m（ketchupは0.30 m）。']
    if has_ood:
        lines += ['水平0.30–0.80 m／高さ0.60–0.90 m／向きは世界Z軸yaw ±60–90°。変化させない因子は通常目標のまま。',
                  '関節OODは可動域内の大角度1.60–3.00 rad（ketchupは2.45 rad以下）を優先。espressomachineは上限1.50 radのため0.20–0.40 rad。phoneは机と可動部が干渉する中間角を避け2.40–2.45 rad。物体間で難度が同じとは限らない。']
    lines += [
              '配置目標には、衝突メッシュが支持面に埋まらない最小の上方補正（最大2 cm）を計画段階で適用。途中の要求上昇量は変更しない。',
              '初期角から既に許容誤差内の関節目標もあり得る。試行ごとの初期誤差・実際の関節移動量はepisodes.jsonlに保存。',
              '物体ベースは浮動で高さ上限なし。手首のローカル並進可動域、物体関節可動域、支持面と衝突形状を全目標について検査。学習方策の成功や十分な接触・保持を保証する検査ではない。制御時間・PDゲイン・質量・重力・関節可動域は公開設定を維持。', '']
    lines += ['- [{}]({})'.format(label,path) for label,path in links]+['']
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    # A compact local report: results first, then one table per condition.
    body=[]
    for section_title,rs in [(object_title,aggregates)]+[(LABELS[p],[r for r in groups if r['profile']==p]) for p in plan['profiles']]:
        body.append('<h2>'+html.escape(section_title)+'</h2><table><thead><tr><th>条件 / 物体</th><th>成功 / 予定</th><th>Suc.T</th><th>PE [m]</th><th>AE [rad]</th><th>AAE [rad]</th><th>論文Suc.T</th><th>異常</th></tr></thead><tbody>')
        for r in rs:
            values=[r.get('object',LABELS[r['profile']]),str(r['successes'])+'/'+str(r['planned']),
                    fmt(r['SucT'],True),fmt(r['PE']),fmt(r['AE']),fmt(r['AAE']),fmt(r['paper_SucT'],True),str(r['invalid'])]
            body.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
        body.append('</tbody></table>')
    page='''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TITLE</title>
<style>body{font:16px system-ui,sans-serif;color:#213047;background:#f5f7fa;margin:0}main{max-width:1100px;margin:auto;padding:32px}h1{font-size:28px}h2{margin-top:36px;font-size:21px}p{line-height:1.8}table{border-collapse:collapse;width:100%;background:white;margin:16px 0}th,td{padding:11px;text-align:right;border-bottom:1px solid #dde3ea}th:first-child,td:first-child{text-align:left}th{background:#e8eef6}a{color:#175aa9}.note{background:#e7eef8;padding:16px;border-radius:8px}</style>
<main><h1>TITLE</h1><p>PROGRESS</p><p class="note"><b>Suc.Tは要求した高さの達成も必須。</b>論文Suc.Tは最終状態のみ。PE・AE・AAEは正常測定できた全試行の無加工平均です。失敗例も含みます。<br>SAMPLING物理条件と測定数は詳細記録に保存しています。</p><p>LINKS</p>TABLES</main></html>'''
    html_links=[('条件・定義の詳細','report.md')]+links
    page=page.replace('TITLE',html.escape(title)).replace('SAMPLING',html.escape(sampling))
    page=page.replace('LINKS',' · '.join('<a href="{}">{}</a>'.format(html.escape(path,quote=True),html.escape(label)) for label,path in html_links))
    (root/'report.html').write_text(page.replace('PROGRESS','{}/{}試行完了{}'.format(len(rows),len(expected),'' if summary['complete'] else '（計算中）')).replace('TABLES',''.join(body)),encoding='utf-8')
    return root/'report.md'
