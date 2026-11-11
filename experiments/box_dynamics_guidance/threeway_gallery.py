"""Write a self-contained, local-file-compatible gallery of saved trajectories.

Manifest and table metrics use metres and radians, matching the main comparison.
Target/trajectory angles additionally use degrees for visual interpretation.
No simulation, rendering, video encoding, network requests or model inference.
"""
import json
import math
import os
from pathlib import Path, PurePosixPath
import re


METHOD_IDS = ("baseline", "guided_absolute", "guided_delta")


def _number(value, label, nullable=False):
    if nullable and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("{} must be a finite number".format(label))


def _asset(value, label):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("{} must be a relative local asset path".format(label))
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("{} must remain inside the gallery directory".format(label))


def _validate(manifest):
    if manifest.get("schema_version") != 1 or manifest.get("status") not in ("complete", "partial"):
        raise ValueError("Expected a complete or partial schema-version-1 gallery manifest")
    methods = manifest.get("methods", [])
    if len(methods) != 3 or {method["id"] for method in methods} != set(METHOD_IDS):
        raise ValueError("Expected the three comparison methods")
    for method in methods:
        if not isinstance(method["label"], str) or not re.fullmatch(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?", method["color"]):
            raise ValueError("Each method needs a label and hexadecimal color")
    cases = manifest.get("cases", [])
    if not cases or len({case["episode_id"] for case in cases}) != len(cases):
        raise ValueError("Expected nonempty cases with unique episode IDs")
    for case in cases:
        if case["profile"] not in ("normal", "ood_articulation"):
            raise ValueError("Unknown case profile")
        for key in ("target_angle_rad", "duration", "fps", "opening_start"):
            _number(case[key], key)
        if not 0 <= case["opening_start"] < case["duration"] or case["fps"] <= 0:
            raise ValueError("Invalid case duration, opening time or frame rate")
        if "frame_state_indices" in case:
            indices = case["frame_state_indices"]
            if (not indices or any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices)
                    or any(a > b for a, b in zip(indices, indices[1:]))
                    or abs(indices[-1] / 100 - case["duration"]) > 1e-6):
                raise ValueError("Frame state indices must be ordered physical steps ending at the true terminal state")
        for key in ("video", "poster", "terminal_image"):
            _asset(case[key], key)
        if {row["method"] for row in case["metrics"]} != set(METHOD_IDS) or len(case["metrics"]) != 3:
            raise ValueError("Each case requires metrics for all three methods")
        for row in case["metrics"]:
            for key in ("PE", "AAE", "AE"):
                _number(row[key], key, nullable=True)
        times = case["series"]["time_s"]
        if not times:
            raise ValueError("Angle series must not be empty")
        for value in times:
            _number(value, "series.time_s")
        if any(a > b for a, b in zip(times, times[1:])) or times[0] < 0 or times[-1] > case["duration"] + 1e-6:
            raise ValueError("Series timestamps must be ordered and inside the case duration")
        for method in METHOD_IDS:
            angles = case["series"]["angle_rad"][method]
            if len(angles) != len(times):
                raise ValueError("Angle/time series lengths differ")
            for value in angles:
                _number(value, "angle_rad", nullable=True)
    for row in manifest.get("summary_rows", []):
        if row["method"] not in METHOD_IDS or row["profile"] not in ("normal", "ood_articulation"):
            raise ValueError("Unknown summary method/profile")
        for key in ("PE", "AAE", "AE"):
            _number(row[key], key, nullable=True)
        for key in ("n", "paper_success_count", "composite_success_count"):
            if isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] < 0:
                raise ValueError("Summary counts must be nonnegative integers")
        if max(row["paper_success_count"], row["composite_success_count"]) > row["n"]:
            raise ValueError("Success count exceeds trial count")


def write_gallery(output: Path, manifest: dict):
    """Write output/index.html; all assets are relative to that directory."""
    _validate(manifest)
    # Escape HTML-sensitive characters in JSON even though it is a data script.
    payload = json.dumps(manifest, ensure_ascii=False, allow_nan=False)
    for literal, escaped in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                             ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        payload = payload.replace(literal, escaped)
    html = _HTML.replace("__GALLERY_DATA__", payload)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "index.html"
    temporary = output / "index.html.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(html)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return destination


_HTML = r'''<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light">
<title>ArtiGrasp | 3方式の動作比較</title>
<style>
:root{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans JP",sans-serif;color:#182431;background:#f3f5f7;font-synthesis:none;--green:#16834b;--border:#dce2e8;--muted:#607080}
*{box-sizing:border-box}body{margin:0}button,select,a{touch-action:manipulation}button,select{font:inherit}button,select,.download{border:1px solid var(--border);border-radius:8px;background:#fff;color:inherit;padding:9px 13px}button{cursor:pointer}button:hover,.download:hover{background:#edf2f7}button:focus-visible,select:focus-visible,a:focus-visible,svg:focus-visible{outline:3px solid #7aa9ed;outline-offset:3px}button[aria-pressed="true"]{background:#e6effb;border-color:#4372ae;color:#18487f}button.primary{background:#243e60;color:white;border-color:#243e60;min-width:84px}button.primary:hover{background:#31527d}button:disabled{opacity:.5;cursor:default}a{color:#245b99}.page{max-width:1580px;margin:0 auto;padding:30px 30px 24px}.eyebrow{font-size:12px;letter-spacing:.08em;font-weight:700;color:var(--muted);margin:0 0 7px}h1{font-size:clamp(24px,2.5vw,34px);letter-spacing:-.035em;margin:0 0 10px}h2{font-size:21px;margin:0}p{line-height:1.7}.intro{margin:0;color:var(--muted);max-width:1000px;font-size:14px}.layout{display:grid;grid-template-columns:260px minmax(0,1fr);gap:22px;margin-top:25px;align-items:start}.sidebar{position:sticky;top:18px}.filters{display:flex;gap:5px}.filters button{flex:1;padding:8px 5px;font-size:13px}.case-count{font-size:12px;color:var(--muted);margin:10px 2px}.case-list{display:flex;flex-direction:column;gap:9px}.case-card{display:block;width:100%;text-align:left;padding:13px;line-height:1.5}.case-card[aria-current="true"]{border-color:#436b9b;box-shadow:inset 3px 0 #436b9b;background:#edf3fb}.case-card strong{display:block;font-size:14px;margin:6px 0}.case-card .case-angle{display:block;font-size:12px;color:var(--green)}.badge{display:inline-block;background:#e9edf1;color:#465769;font-size:11px;border-radius:5px;padding:2px 6px;font-weight:650}.badge.ood{background:#f8ecd8;color:#8d5916}.card{background:#fff;border:1px solid var(--border);border-radius:12px;overflow:hidden}.case-heading{padding:20px 22px 15px}.title-row{display:flex;justify-content:space-between;align-items:flex-start;gap:18px}.title-group{min-width:0}.title-group h2{margin-top:8px}.goal{flex-shrink:0;text-align:right;background:#eef8f1;border:1px solid #c5e5d0;padding:10px 14px;border-radius:9px;color:var(--green)}.goal span{display:block;font-size:12px}.goal strong{display:block;font-size:24px;line-height:1.35}.goal small{font-size:12px}.reason{margin:11px 0 0;font-size:13px;color:var(--muted)}.model-legend{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1px;border-top:1px solid var(--border);background:var(--border)}.model-label{background:#f8fafb;padding:10px 8px;text-align:center;font-weight:650;font-size:13px;border-top:3px solid var(--method-color)}.media{background:#121920;position:relative;aspect-ratio:24/11}.media video,.media img{display:block;width:100%;height:100%;object-fit:contain}.media [hidden]{display:none}.terminal-label{position:absolute;top:12px;right:12px;background:rgba(15,25,35,.85);color:#fff;font-size:12px;border-radius:6px;padding:7px 10px;pointer-events:none}.controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:13px 16px 4px}.controls button,.controls select{font-size:13px}.speed{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--muted)}.time{font-variant-numeric:tabular-nums;font-size:13px;color:var(--muted);margin-left:auto;white-space:nowrap}.media-foot{display:flex;justify-content:space-between;align-items:center;gap:15px;padding:5px 17px 13px;font-size:12px;color:var(--muted)}.status{margin:0;min-height:20px;font-size:12px;line-height:1.5}.download{padding:5px 9px;text-decoration:none;white-space:nowrap;font-size:12px}.chart-panel{padding:16px 19px 12px;border-top:1px solid var(--border)}.chart-title{display:flex;justify-content:space-between;gap:12px;align-items:baseline}.chart-title h3{font-size:14px;margin:0}.chart-title span{font-size:11px;color:var(--muted)}#angleChart{display:block;width:100%;height:auto;cursor:crosshair;border-radius:4px}.angle-readout{display:flex;gap:15px;flex-wrap:wrap;font-size:12px;font-variant-numeric:tabular-nums;margin:0 0 1px 8px}.readout-item{border-left:3px solid var(--method-color);padding-left:7px}.table-panel{padding:15px 19px 19px;border-top:1px solid var(--border)}.table-panel h3{font-size:14px;margin:0 0 11px}.scroll-table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums;white-space:nowrap}th,td{padding:10px 9px;border-bottom:1px solid #e7ebef;text-align:right}th{background:#f7f9fb;color:#546476;font-weight:600;font-size:11px}th:first-child,td:first-child{text-align:left}tr:last-child td{border-bottom:0}.method-cell{display:inline-flex;align-items:center;gap:8px;font-weight:550}.method-dot{width:9px;height:9px;border-radius:50%;background:var(--method-color);flex:none}.result{font-size:11px;border-radius:4px;padding:3px 6px;background:#f0f2f5;color:#687582}.result.success{background:#e5f5eb;color:#187143}.summary{margin-top:24px;padding:22px}.summary p{color:var(--muted);font-size:13px;margin:8px 0 15px}.summary td:nth-child(2),.summary th:nth-child(2){text-align:left}.summary tr.group-start td{border-top:2px solid #d9e1e9}.definitions{margin-top:15px;font-size:12px;color:var(--muted);line-height:1.8}.definitions summary{cursor:pointer;font-weight:600}.definitions dl{display:grid;grid-template-columns:100px 1fr;gap:4px 14px;margin:10px 0 0}.definitions dt{font-weight:650}.definitions dd{margin:0}.note{font-size:12px;line-height:1.8;color:var(--muted);margin:17px 2px 0}.note strong{color:#425367}#empty{padding:30px;text-align:center;color:var(--muted)}[hidden]{display:none!important}
@media(max-width:1100px){.page{padding:22px 18px}.layout{grid-template-columns:210px minmax(0,1fr);gap:14px}.case-heading{padding:17px}.goal{padding:8px 10px}.goal strong{font-size:21px}.model-label{font-size:11px}.controls{gap:6px}.time{margin-left:0}.title-group h2{font-size:18px}}
@media(max-width:780px){.layout{display:block}.sidebar{position:static;margin-bottom:15px}.filters{max-width:320px}.case-list{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.case-card{padding:10px}.case-card strong{font-size:12px}.title-row{gap:10px}.goal strong{font-size:20px}.model-label{font-size:10px;padding:7px 3px}.controls{padding:10px 10px 4px}.controls button{padding:8px 9px;font-size:12px}.chart-panel,.table-panel{padding:13px 10px}.summary{padding:16px 12px}.chart-title span{max-width:140px;text-align:right}.goal small{font-size:10px}.title-group h2{font-size:17px}.reason{font-size:12px}.angle-readout{font-size:10px;gap:9px}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
</style>
</head>
<body>
<main class="page">
 <p class="eyebrow">ARTIGRASP / 保存軌道の比較</p>
 <h1>通常 PPO・絶対状態モデル・差分モデルの動作比較</h1>
 <p class="intro">同じ初期状態と目標で実行した 3 方式を、1 本の同期動画で比較します。各ケースの 15.6 秒の全編を同時に描画しています。緑が目標の蓋角度です。</p>
 <p id="generationStatus" role="status" style="margin:16px 0 0;padding:11px 14px;background:#fff3d8;border:1px solid #ebd6a5;border-radius:8px;color:#77551d;font-size:13px" hidden></p>
 <div class="layout">
  <aside class="sidebar" aria-label="比較ケースの選択">
   <div class="filters" role="group" aria-label="条件で絞り込む">
    <button type="button" data-filter="all" aria-pressed="true">全部</button>
    <button type="button" data-filter="normal" aria-pressed="false">normal</button>
    <button type="button" data-filter="ood_articulation" aria-pressed="false">OOD</button>
   </div>
   <p id="caseCount" class="case-count"></p>
   <nav id="caseList" class="case-list" aria-label="ケース"></nav>
  </aside>
  <section id="casePanel" class="card" aria-labelledby="caseTitle">
   <div class="case-heading">
    <div class="title-row"><div class="title-group"><span id="conditionBadge" class="badge"></span><h2 id="caseTitle"></h2></div>
     <div class="goal"><span>目標の蓋角度</span><strong id="goalDegrees"></strong><small id="goalRadians"></small></div>
    </div><p id="selectionReason" class="reason"></p>
   </div>
   <div id="modelLegend" class="model-legend" aria-label="左から順に表示する方式"></div>
   <div class="media">
    <video id="video" controls playsinline preload="metadata" aria-label="3方式を横に並べた同期比較動画"></video>
    <img id="terminalImage" alt="3方式の保存された終点状態" hidden>
    <span id="terminalLabel" class="terminal-label" hidden>保存された終点状態</span>
   </div>
   <div class="controls">
    <button id="playPause" class="primary" type="button">再生</button>
    <button id="openingButton" type="button">12.6秒から</button>
    <button id="loopButton" type="button" aria-pressed="false">開閉区間をループ</button>
    <button id="terminalButton" type="button" aria-pressed="false">終点の静止画</button>
    <label class="speed">速度<select id="speed" aria-label="再生速度"><option value="1">1×</option><option value="0.5">0.5×</option><option value="0.25">0.25×</option></select></label>
    <span id="timeReadout" class="time"></span>
   </div>
   <div class="media-foot"><p id="mediaStatus" class="status" role="status" aria-live="polite"></p><a id="videoDownload" class="download" download>MP4を保存</a></div>
   <p class="reason" style="margin:0;padding:0 17px 15px">上段の拡大表示は物体中心を追従します。基部の位置ずれは、下段左の共通固定視点と PE で確認してください。下段右は角度の推移です。</p>
   <div class="chart-panel">
    <div class="chart-title"><h3>蓋の角度の推移</h3><span>グラフをクリックするとその時刻へ移動</span></div>
    <svg id="angleChart" viewBox="0 0 920 232" role="slider" tabindex="0" aria-label="蓋角度の比較と再生時刻" aria-valuemin="0"></svg>
    <div id="angleReadout" class="angle-readout"></div>
   </div>
   <div class="table-panel"><h3>このケースの終点</h3><div class="scroll-table"><table><thead><tr><th>方式</th><th>蓋角度</th><th>PE (m)</th><th>AAE (rad)</th><th>AE (rad)</th><th>終点成功</th><th>複合成功</th></tr></thead><tbody id="caseMetrics"></tbody></table></div></div>
  </section>
  <div id="empty" class="card" hidden>この条件のケースはありません。</div>
 </div>
 <section class="summary card" aria-labelledby="summaryTitle">
  <h2 id="summaryTitle">全試行の集計</h2>
  <p>上の選択例とは別に、保存された全試行を集計しています。誤差は終点の平均値です。</p>
  <div class="scroll-table"><table><thead><tr><th>条件</th><th>方式</th><th>試行数</th><th>終点成功</th><th>複合成功</th><th>PE (m)</th><th>AAE (rad)</th><th>AE (rad)</th></tr></thead><tbody id="summaryRows"></tbody></table></div>
  <details class="definitions"><summary>指標の定義</summary><dl>
   <dt>PE</dt><dd>物体基部の位置誤差。表では m で表示しています（0.01 m = 1 cm）。</dd>
   <dt>AAE</dt><dd>目標に対する蓋の角度誤差。表では rad を小数 6 桁で表示しています。</dd>
   <dt>AE</dt><dd>物体基部の姿勢誤差。表では rad で表示しています。</dd>
   <dt>終点成功</dt><dd>PE &lt; 0.05 m、AE &lt; 0.2 rad、AAE &lt; 0.5 rad を同時に満たす成功です。</dd>
   <dt>複合成功</dt><dd>終点条件に加え、途中の持ち上げ条件も満たす成功です。</dd>
  </dl></details>
 </section>
 <p class="note"><strong>描画と比較の範囲：</strong>seed 202 の単一 seed による探索的比較です。保存された状態と URDF から描画しており、再シミュレーションは行っていません。元の比較結果は独立監査済みです。動画は 20 fps で抽出し、最終フレームには真の終点（15.6 秒）を使用しています。時刻表示と角度カーソルは表示フレームの物理時刻に合わせています。終点ボタンも保存された終点状態の静止画を表示します。選択例は動きの理解を助けるためのもので、全試行の性能平均の代わりにはなりません。</p>
</main>
<script id="galleryData" type="application/json">__GALLERY_DATA__</script>
<script>
"use strict";
(() => {
 const data = JSON.parse(document.getElementById("galleryData").textContent);
 const ids = ["baseline", "guided_absolute", "guided_delta"];
 const methods = ids.map(id => data.methods.find(method => method.id === id));
 const byId = Object.fromEntries(methods.map(method => [method.id, method]));
 const $ = id => document.getElementById(id);
 if(data.status === "partial") {$("generationStatus").hidden = false; $("generationStatus").textContent = "動画生成中（現時点 "+data.cases.length+" 本）— 完了したケースだけを表示しています。";}
 const video = $("video"), terminalImage = $("terminalImage"), chart = $("angleChart");
 const degrees = value => value * 180 / Math.PI;
 const fmt = (value, digits = 2) => Number.isFinite(value) ? value.toLocaleString("ja-JP", {minimumFractionDigits:digits, maximumFractionDigits:digits}) : "—";
 const metric = (value, digits = 4) => value === null || value === undefined ? "—" : fmt(value,digits);
 const node = (tag, text, className) => {const element = document.createElement(tag); if(text !== undefined) element.textContent = text; if(className) element.className = className; return element;};
 const svgNode = (tag, attrs, text) => {const element = document.createElementNS("http://www.w3.org/2000/svg", tag); for(const [key,value] of Object.entries(attrs || {})) element.setAttribute(key,String(value)); if(text !== undefined) element.textContent = text; return element;};
 let current = null, filter = "all", loopOpening = false, terminalMode = false, pendingSeek = null, playToken = 0, frameRequest = 0, plot = null;

 function methodLabel(id) {
  const method = byId[id], label = node("span", undefined, "method-cell");
  label.style.setProperty("--method-color", method.color);
  label.append(node("span", undefined, "method-dot"), node("span", method.label));
  return label;
 }
 function successCell(success) {const td = node("td"); td.append(node("span", success ? "成功" : "未達", "result" + (success ? " success" : ""))); return td;}
 function conditionText(item) {return item.profile === "normal" ? "normal" : "OOD" + (item.ood_side === "small" ? " · 小角度" : item.ood_side === "large" ? " · 大角度" : "");}
 function setLoop(enabled) {loopOpening = enabled; $("loopButton").setAttribute("aria-pressed",String(enabled));}
 function showVideo() {terminalMode = false; terminalImage.hidden = true; $("terminalLabel").hidden = true; video.hidden = false; $("terminalButton").setAttribute("aria-pressed","false"); $("terminalButton").textContent = "終点の静止画";}
 function stopAnimation() {if(frameRequest) cancelAnimationFrame(frameRequest); frameRequest = 0;}
 function pause() {playToken += 1; video.pause(); stopAnimation(); $("playPause").textContent = "再生";}
 function requestPlay() {
  if(!current) return;
  showVideo();
  const token = ++playToken;
  if(video.ended && pendingSeek === null) seekTo(loopOpening ? current.opening_start : 0, false);
  try {
   const promise = video.play();
   if(promise && typeof promise.catch === "function") promise.catch(error => {
    if(token !== playToken || error.name === "AbortError") return;
    $("mediaStatus").textContent = "再生を開始できませんでした。動画の再生ボタン、または「MP4を保存」から確認してください。";
    $("playPause").textContent = "再生";
   });
  } catch(error) {$("mediaStatus").textContent = "このブラウザでは動画を再生できません。「MP4を保存」から確認してください。";}
 }
 function seekTo(seconds, playAfter = false) {
  if(!current) return;
  showVideo();
  const target = Math.max(0,Math.min(current.duration,seconds));
  if(loopOpening && target < current.opening_start - 0.01) setLoop(false);
  pendingSeek = target;
  if(video.readyState >= 1) {
   const limit = Number.isFinite(video.duration) ? Math.max(0,video.duration - 0.001) : current.duration;
   try {video.currentTime = Math.min(target,limit); pendingSeek = null;} catch(error) {/* metadata callback retries */}
  }
  updateTimeline();
  if(playAfter) requestPlay();
 }
 function showTerminal() {
  if(!current) return;
  pause(); setLoop(false); terminalMode = true; video.hidden = true;
  terminalImage.hidden = false; $("terminalLabel").hidden = false;
  $("terminalButton").setAttribute("aria-pressed","true"); $("terminalButton").textContent = "動画に戻る";
  $("mediaStatus").textContent = "保存された物理終点の静止画です。下表も同じ終点の値です。";
  updateTimeline();
 }
 function angleAt(method, time) {
  const times = current.series.time_s, values = current.series.angle_rad[method];
  let lo = 0, hi = times.length - 1;
  while(lo < hi) {const mid = Math.floor((lo + hi + 1) / 2); if(times[mid] <= time) lo = mid; else hi = mid - 1;}
  const next = Math.min(lo + 1,times.length - 1), a = values[lo], b = values[next];
  if(!Number.isFinite(a)) return null;
  if(!Number.isFinite(b) || next === lo || times[next] <= times[lo]) return a;
  const ratio = Math.max(0,Math.min(1,(time-times[lo])/(times[next]-times[lo])));
  return a + (b-a)*ratio;
 }
 function buildChart() {
  chart.replaceChildren(); $("angleReadout").replaceChildren();
  const width = 920, height = 232, margin = {left:54,right:18,top:26,bottom:31};
  const values = [degrees(current.target_angle_rad)];
  for(const method of methods) for(const angle of current.series.angle_rad[method.id]) if(Number.isFinite(angle)) values.push(degrees(angle));
  let ymin = Math.min(0,...values), ymax = Math.max(10,...values), span = ymax-ymin;
  ymin -= Math.max(3,span*.07); ymax += Math.max(3,span*.09); span = ymax-ymin;
  const x = time => margin.left+(width-margin.left-margin.right)*time/current.duration;
  const y = angle => height-margin.bottom-(height-margin.top-margin.bottom)*(angle-ymin)/span;
  chart.append(svgNode("rect",{x:x(current.opening_start),y:margin.top,width:x(current.duration)-x(current.opening_start),height:height-margin.top-margin.bottom,fill:"#f0f4fa"}));
  chart.append(svgNode("text",{x:x(current.opening_start)+5,y:16,fill:"#6b7d92","font-size":10},"開閉区間"));
  for(let i=0;i<=4;i++) {const value=ymin+span*i/4, yy=y(value); chart.append(svgNode("line",{x1:margin.left,x2:width-margin.right,y1:yy,y2:yy,stroke:"#e5eaf0"})); chart.append(svgNode("text",{x:margin.left-8,y:yy+4,"text-anchor":"end",fill:"#6e7c8b","font-size":10},fmt(value,0)+"°"));}
  for(let i=0;i<=6;i++) {const t=current.duration*i/6, xx=x(t); chart.append(svgNode("text",{x:xx,y:height-10,"text-anchor":"middle",fill:"#6e7c8b","font-size":10},fmt(t,1)+"s"));}
  chart.append(svgNode("line",{x1:margin.left,x2:width-margin.right,y1:y(degrees(current.target_angle_rad)),y2:y(degrees(current.target_angle_rad)),stroke:"#16834b","stroke-width":2,"stroke-dasharray":"7 5"}));
  chart.append(svgNode("text",{x:margin.left+5,y:y(degrees(current.target_angle_rad))-6,fill:"#16834b","font-size":11,"font-weight":600},"目標 "+fmt(degrees(current.target_angle_rad),1)+"°"));
  const markers = {}, readings = {};
  for(const method of methods) {
   let path = "", connected = false;
   current.series.time_s.forEach((time,index) => {const value=current.series.angle_rad[method.id][index]; if(!Number.isFinite(value)){connected=false;return;} path+=(connected?"L":"M")+x(time).toFixed(2)+","+y(degrees(value)).toFixed(2)+" "; connected=true;});
   chart.append(svgNode("path",{d:path,fill:"none",stroke:method.color,"stroke-width":2,"stroke-linejoin":"round","stroke-linecap":"round"}));
   const item=node("span",undefined,"readout-item"); item.style.setProperty("--method-color",method.color); item.append(node("span",method.label+" "));
   const reading=node("strong"); item.append(reading); $("angleReadout").append(item); readings[method.id]=reading;
   const marker=svgNode("circle",{r:4,fill:method.color,stroke:"white","stroke-width":1.5}); markers[method.id]=marker;
  }
  const cursor=svgNode("line",{x1:x(0),x2:x(0),y1:margin.top,y2:height-margin.bottom,stroke:"#23354c","stroke-width":1.2,"stroke-dasharray":"3 3"});
  chart.append(cursor); for(const marker of Object.values(markers)) chart.append(marker);
  chart.setAttribute("aria-valuemax",String(current.duration));
  plot={x,y,cursor,markers,readings,left:margin.left,right:width-margin.right,width};
 }
 function frameTime(item, videoSeconds) {
  const time = Math.max(0,Math.min(item.duration,videoSeconds));
  const indices = item.frame_state_indices;
  if(!Array.isArray(indices) || !indices.length) return time;
  const frame = Math.min(indices.length-1,Math.max(0,Math.floor(time*item.fps+1e-6)));
  return indices[frame]/100;
 }
 function updateTimeline() {
  if(!current) return;
  const time = terminalMode ? current.duration : frameTime(current,pendingSeek === null ? (video.currentTime || 0) : pendingSeek);
  $("timeReadout").textContent = (terminalMode ? "終点 " : "")+fmt(time,2)+" / "+fmt(current.duration,1)+" 秒";
  chart.setAttribute("aria-valuenow",time.toFixed(2)); chart.setAttribute("aria-valuetext",fmt(time,2)+"秒");
  if(!plot) return;
  plot.cursor.setAttribute("x1",plot.x(time)); plot.cursor.setAttribute("x2",plot.x(time));
  for(const method of methods) {const angle=angleAt(method.id,time), marker=plot.markers[method.id]; plot.readings[method.id].textContent=Number.isFinite(angle)?fmt(degrees(angle),1)+"°":"—"; marker.setAttribute("visibility",Number.isFinite(angle)?"visible":"hidden"); if(Number.isFinite(angle)){marker.setAttribute("cx",plot.x(time));marker.setAttribute("cy",plot.y(degrees(angle)));}}
 }
 function animate() {
  frameRequest=0;
  if(current && loopOpening && !terminalMode && !video.paused && video.currentTime >= Math.min(current.duration,Number.isFinite(video.duration)?video.duration:current.duration)-.025) seekTo(current.opening_start,false);
  updateTimeline();
  if(!video.paused && !video.ended && !terminalMode) frameRequest=requestAnimationFrame(animate);
 }
 function renderMetrics() {
  const tbody=$("caseMetrics"); tbody.replaceChildren();
  for(const method of methods) {
   const row=current.metrics.find(item=>item.method===method.id), tr=node("tr"), first=node("td"); first.append(methodLabel(method.id));tr.append(first);
   const angle=angleAt(method.id,current.duration);
   tr.append(node("td",Number.isFinite(angle)?fmt(degrees(angle),1)+"°":"—"),node("td",metric(row.PE)),node("td",metric(row.AAE,6)),node("td",metric(row.AE)),successCell(row.paper_success),successCell(row.composite_success));tbody.append(tr);
  }
 }
 function selectCase(item) {
  pause(); current=item; setLoop(false); showVideo(); pendingSeek=null;
  $("caseTitle").textContent=item.title;
  $("conditionBadge").textContent=conditionText(item); $("conditionBadge").className="badge"+(item.profile==="normal"?"":" ood");
  $("selectionReason").textContent="選定理由："+item.selection_reason;
  $("goalDegrees").textContent=fmt(degrees(item.target_angle_rad),1)+"°"; $("goalRadians").textContent=fmt(item.target_angle_rad,3)+" rad";
  $("openingButton").textContent=fmt(item.opening_start,1)+"秒から";
  video.src=item.video;video.poster=item.poster;video.playbackRate=Number($("speed").value);video.load();
  terminalImage.src=item.terminal_image;terminalImage.alt=item.title+"：3方式の保存された終点状態";
  $("videoDownload").href=item.video; $("mediaStatus").textContent="再生すると3方式が同じ時刻で進みます。";
  $("casePanel").hidden=false;$("empty").hidden=true;
  buildChart();renderMetrics();updateTimeline();
  for(const button of $("caseList").querySelectorAll("button")) button.setAttribute("aria-current",String(button.dataset.episode===item.episode_id));
 }
 function renderCaseList() {
  const visible=data.cases.filter(item=>filter==="all"||item.profile===filter);
  $("caseCount").textContent=visible.length+" / "+data.cases.length+" ケース";$("caseList").replaceChildren();
  for(const item of visible) {
   const button=node("button",undefined,"case-card");button.type="button";button.dataset.episode=item.episode_id;
   button.append(node("span",conditionText(item),"badge"+(item.profile==="normal"?"":" ood")),node("strong",item.title),node("span","目標 "+fmt(degrees(item.target_angle_rad),1)+"°","case-angle"));
   button.addEventListener("click",()=>selectCase(item));button.setAttribute("aria-current",String(current&&current.episode_id===item.episode_id));$("caseList").append(button);
  }
  if(!visible.length){pause();current=null;$("casePanel").hidden=true;$("empty").hidden=false;return;}
  if(!current||!visible.some(item=>item.episode_id===current.episode_id))selectCase(visible[0]);
 }
 function renderSummary() {
  const rows=data.summary_rows||[], total=rows.reduce((sum,row)=>sum+row.n,0);$("summaryTitle").textContent="全 "+total.toLocaleString("ja-JP")+" 試行の集計";
  for(const profile of ["normal","ood_articulation"]) {
   methods.forEach((method,index)=>{const row=rows.find(item=>item.profile===profile&&item.method===method.id);if(!row)return;
    const tr=node("tr",undefined,index===0?"group-start":""), label=node("td");label.append(methodLabel(method.id));
    const rate=(count)=>count+" / "+row.n+" ("+(row.n?fmt(100*count/row.n,0):"—")+"%)";
    tr.append(node("td",row.condition_label),label,node("td",String(row.n)),node("td",rate(row.paper_success_count)),node("td",rate(row.composite_success_count)),node("td",metric(row.PE)),node("td",metric(row.AAE,6)),node("td",metric(row.AE)));$("summaryRows").append(tr);
   });
  }
 }
 for(const method of methods){const label=node("div",method.label,"model-label");label.style.setProperty("--method-color",method.color);$("modelLegend").append(label);}
 for(const button of document.querySelectorAll("[data-filter]"))button.addEventListener("click",()=>{filter=button.dataset.filter;for(const other of document.querySelectorAll("[data-filter]"))other.setAttribute("aria-pressed",String(other===button));renderCaseList();});
 $("playPause").addEventListener("click",()=>{if(terminalMode){showVideo();$("mediaStatus").textContent="";requestPlay();}else if(video.paused||video.ended)requestPlay();else pause();});
 $("openingButton").addEventListener("click",()=>{if(current)seekTo(current.opening_start,true);});
 $("loopButton").addEventListener("click",()=>{if(!current)return;setLoop(!loopOpening);if(loopOpening)seekTo(current.opening_start,true);});
 $("terminalButton").addEventListener("click",()=>{if(terminalMode){showVideo();$("mediaStatus").textContent="";updateTimeline();}else showTerminal();});
 $("speed").addEventListener("change",()=>{video.playbackRate=Number($("speed").value);});
 video.addEventListener("loadedmetadata",()=>{video.playbackRate=Number($("speed").value);if(pendingSeek!==null)seekTo(pendingSeek,false);updateTimeline();});
 video.addEventListener("play",()=>{$("playPause").textContent="一時停止";$("mediaStatus").textContent="";stopAnimation();frameRequest=requestAnimationFrame(animate);});
 video.addEventListener("pause",()=>{$("playPause").textContent="再生";stopAnimation();updateTimeline();});
 video.addEventListener("timeupdate",updateTimeline);video.addEventListener("seeked",updateTimeline);
 video.addEventListener("ended",()=>{stopAnimation();if(loopOpening&&current&&!terminalMode)seekTo(current.opening_start,true);else{$("playPause").textContent="再生";updateTimeline();}});
 video.addEventListener("error",()=>{$("mediaStatus").textContent="動画を読み込めません。HTMLと videos フォルダを同じ配置に保って開いてください。";$("playPause").textContent="再生";});
 terminalImage.addEventListener("error",()=>{if(terminalMode)$("mediaStatus").textContent="終点画像を読み込めません。「動画に戻る」から動きを確認できます。";});
 chart.addEventListener("pointerdown",event=>{if(!current||!plot)return;const rect=chart.getBoundingClientRect(),x=(event.clientX-rect.left)*plot.width/rect.width;const t=(x-plot.left)/(plot.right-plot.left)*current.duration;seekTo(t,false);});
 chart.addEventListener("keydown",event=>{if(!current)return;let target=terminalMode?current.duration:video.currentTime;const step=event.shiftKey?1:.1;if(event.key==="ArrowLeft")target-=step;else if(event.key==="ArrowRight")target+=step;else if(event.key==="Home")target=0;else if(event.key==="End")target=current.duration;else return;event.preventDefault();seekTo(target,false);});
 renderSummary();renderCaseList();
})();
</script>
</body>
</html>
'''
