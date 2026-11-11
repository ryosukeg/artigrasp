#!/usr/bin/env python3
"""Render selected saved box trajectories after numerical replay verification."""

import argparse
from datetime import datetime, timedelta, timezone
import html
import json
import math
import os
from pathlib import Path
import sys
import tempfile


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXAMPLES = {
    "box__normal__0002": ("最終状態の目標を達成した例", "蓋の開き方と、物体基部の安定性を確認してください。持ち上げを含む複合条件では未達です。"),
    "box__normal__0000": ("蓋が動き 物体全体もずれた例", "最終開閉中の基部の移動・傾きと、目標の半透明モデルとの位置関係に注目してください。"),
    "box__normal__0004": ("持ち上げ後 最終開閉が未達の例", "開閉区間での手と蓋の動き、終了時の蓋の角度を確認してください。"),
}


def make_gallery(output, entries, replay_results, renderer):
    cards = []
    for entry in entries:
        episode = entry["episode_id"]
        title, note = EXAMPLES.get(episode, (episode, "保存した実測軌跡の確認用です。"))
        video = "../videos/" + entry["video"]
        poster = "../videos/" + entry["preview"]
        replay = replay_results[episode]
        final_status = "達成" if entry["paper_SucT"] else "未達"
        composite_status = "達成" if entry["SucT"] else "未達"
        cards.append("""<article>
<h2>{title}</h2><p>{episode} · {duration:.1f}秒 · 最終3条件：{final_status} · 持ち上げ込み：{composite_status}</p>
<video controls preload="metadata" poster="{poster}" src="{video}"></video>
<div class="controls"><button class="start">最初から再生</button><button class="articulate">最終開閉から再生</button>
<label>再生速度 <select class="speed"><option value="1">1倍</option><option value="0.5">0.5倍</option><option value="0.25">0.25倍</option></select></label>
<a href="{video}">MP4を開く</a></div>
<p>{note}</p><p>最終誤差：位置 {pe:.3f} m ／ 姿勢 {ae:.3f} rad ／ 関節角 {aae:.3f} rad<br>
数値再生：元の軌跡と一致（最大状態差 {difference:.3g}）。<a href="replays/{episode}.json">再生検証記録</a></p>
</article>""".format(title=html.escape(title), episode=html.escape(episode),
            duration=entry["duration"], final_status=final_status, composite_status=composite_status,
            video=html.escape(video), poster=html.escape(poster), note=html.escape(note),
            pe=entry["PE"], ae=entry["AE"], aae=entry["AAE"], difference=replay["max_state_difference"]))
    document = """<!doctype html>
<html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>box 補正なし動作の確認動画</title>
<style>body{margin:0;background:#f4f6f8;color:#223247;font:16px system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:24px}p{line-height:1.8}article{background:white;border:1px solid #d5dee7;border-radius:10px;padding:20px;margin:24px 0}h1{font-size:26px}h2{font-size:21px}video{width:100%;background:#e8edf2}.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:12px}button,select{font:inherit;padding:7px 12px}a{color:#145cad}.note{background:#eaf2fa;padding:12px 18px;border-radius:8px}</style>
<main><h1>box 補正なし動作の確認動画</h1>
<p>初回8試行から選んだ3例です。右手は赤、左手は青、物体は金色、目標は半透明の緑です。全体表示と右上の物体拡大表示で動きを確認できます。</p>
<p class="note">動画は評価時に保存した手・物体の物理状態と、実際のURDF形状から描画しています。Unityの画面録画ではありません。別途、同じ行動をRaiSimで再実行し、元の軌跡との一致を確認しています。<br>
動画内の SUCCESS / FAILURE は持ち上げも含む複合条件です。「最終状態の目標を達成した例」も、持ち上げ条件が未達のため動画内では FAILURE と表示されます。</p>
<p>最終開閉は12.6秒からです。「最終開閉から再生」で移動でき、0.5倍・0.25倍でも確認できます。<a href="manifest.json">動画・出典・再生検証の記録</a></p>
""" + "\n".join(cards) + """</main>
<script>for(const card of document.querySelectorAll('article')){const v=card.querySelector('video');const playAt=t=>{v.currentTime=t;v.play().catch(()=>{});};card.querySelector('.start').onclick=()=>playAt(0);card.querySelector('.articulate').onclick=()=>playAt(12.6);card.querySelector('.speed').onchange=e=>{v.playbackRate=Number(e.target.value);};}</script></html>"""
    (output / "index.html").write_text(document, encoding="utf-8")
    manifest = {
        "created_at_jst": datetime.now(timezone(timedelta(hours=9))).isoformat(),
        "method": "saved physical state + URDF rendering, independently checked with RaiSim action replay",
        "rendering_device": renderer, "entries": entries, "replays": replay_results,
        "selection": {key: {"title": value[0], "note": value[1]} for key, value in EXAMPLES.items()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "runs/baseline_smoke_seed101")
    parser.add_argument("--episodes", nargs="+", default=list(EXAMPLES))
    parser.add_argument("--fps", type=float, default=20)
    args = parser.parse_args()
    if not math.isfinite(args.fps) or not 1 <= args.fps <= 60:
        parser.error("fps must be between 1 and 60")
    root = args.results.resolve()
    output = root / "qualitative"
    output.mkdir(parents=True, exist_ok=True)
    rows = {row["episode_id"]: row for row in
            (json.loads(line) for line in (root / "episodes.jsonl").read_text().splitlines() if line.strip())}
    replay_results = {}
    for episode in args.episodes:
        row = rows[episode]
        if row["object"] != "box" or not row["valid"]:
            raise ValueError("Expected a valid box episode: " + episode)
        replay = json.loads((output / "replays" / (episode + ".json")).read_text())
        if replay.get("episode_id") != episode or replay.get("verified") is not True:
            raise ValueError("Numerical replay has not been verified: " + episode)
        replay_results[episode] = replay

    sys.path.insert(0, str(HERE / ".render_deps"))
    sys.path.insert(0, str(REPO / "scripts"))
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    # Singularity exposes the driver library but not the host GLVND vendor JSON.
    with tempfile.TemporaryDirectory(prefix="box_render_egl_") as temporary:
        driver = Path("/.singularity.d/libs/libEGL_nvidia.so.0")
        if driver.is_file() and "__EGL_VENDOR_LIBRARY_FILENAMES" not in os.environ:
            vendor = Path(temporary) / "nvidia.json"
            vendor.write_text(json.dumps({"file_format_version": "1.0.0", "ICD": {"library_path": str(driver)}}))
            os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(vendor)
        import pyglet
        pyglet.options["headless"] = True
        pyglet.options["shadow_window"] = False
        import pyrender
        from OpenGL.GL import glGetString, GL_RENDERER
        import render_ood_five as renderer
        probe = pyrender.OffscreenRenderer(160, 120)
        device = glGetString(GL_RENDERER).decode()
        probe.delete()
        print("Rendering device:", device, flush=True)
        entries = []
        for episode in args.episodes:
            print("Rendering", episode, flush=True)
            entry = renderer.render_one((str(root), str(REPO / "rsc"), rows[episode], args.fps))
            entries.append(entry)
            make_gallery(output, entries, replay_results, device)
            print("Saved", entry["video"], "frames:", entry["frames"], flush=True)
    print("Gallery:", output / "index.html", flush=True)


if __name__ == "__main__":
    main()
