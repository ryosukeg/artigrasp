# OOD評価：水平・高さ・向き・関節角度・全て

既定は `configs/ood_five.json`。全10物体 × 5条件 × 100シーケンス = 5,000試行。
2因子・3因子の組合せ条件やnormal条件は追加しない。

主指標Suc.Tは最終PE < 0.05 m、AE < 0.2 rad、AAE < 0.5 radに加えて、
持ち上げ区間で要求上昇量との誤差 < 0.05 mを必須とする。
論文Suc.T（最終状態だけ）も別列に残す。動画は主指標で成功・失敗を分類。
PE/AE/AAEは正常測定できた全試行の最終誤差平均。失敗例を含み、クリップしない。

| 因子 | 通常目標 | OOD目標 |
|---|---|---|
| 水平 | x,y各−0.10〜−0.05 m | 同じ方向に0.30〜0.80 m |
| 高さ | 途中の上昇量0.15 m、ketchupは0.30 m | 0.60〜0.90 m |
| 向き | 世界Z軸yaw ±0.40 rad | 世界Z軸yaw ±60〜90°（配置時） |
| 関節角 | 絶対角0.50〜0.60 rad | 原則1.60〜3.00 radを可動域内で抽出 |

espressomachineは上限1.50 radなので関節OODは0.20〜0.40 rad。
ketchupは2.45 rad以下。phoneは可動部が机と干渉する中間角を避け2.40〜2.45 rad。
初期角が既に許容誤差内のケースもあるため、初期誤差と実際の関節移動量を保存する。
物体間で同じ難度とはみなさない。

全目標について関節可動域、手首ローカル並進可動域、衝突メッシュと支持面を確認。
支持面への埋まりを防ぐ最小の配置高さ補正（最大2 cm）は計画に明記する。
途中の要求上昇量は変えない。物体ベースは浮動で高さ上限なし。
物理条件・可動域・制御時間・学習済み重みは変更しない。
幾何的な制約の確認は、学習方策による接触・保持・動作の成功を保証するものではない。

```bash
python tools/ood_eval.py run --output results/ood5_seed1 --device cuda --num-threads 24
python tools/ood_eval.py report --results results/ood5_seed1
```

主な結果は `report.html`、`report.md`、`metrics.csv`。各条件に10物体の表をまとめる。
`physics_constraints.json` に全目標の物理制約検査、`episodes.jsonl` に全試行、
`trajectories/*.npz` に全時刻の実測状態と操作列を保存する。

可視化は保存済み状態から実際のURDF形状を描画する。
成功・失敗をそれぞれ3本まで選び、不足を記録する。追加試行や成功例の捏造は行わない。
動画一覧は `videos/index.html`。

既存のtext2hoi環境で動画を生成する場合:

```bash
conda run -n text2hoi python scripts/render_ood_five.py --results results/ood5_seed1 --workers 2 --fps 20
```

数値計算と並行して、100例が揃った組から順次描画する場合は `--follow` を追加する。
描画元の軌跡とMP4のSHA256、描画コードの版、動画の長さ・フレーム数を記録する。

旧8条件は `--config configs/ood_multiaxis.json` を明示したときだけ使用する。
