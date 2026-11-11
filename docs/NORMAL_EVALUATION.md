# normal と既存 OOD 5条件の比較

`results/ood5_seed1` と同じ評価コード、重み、test split、seed=1、
10物体×各100試行を使い、OOD の目標変更を一切加えない normal を評価する。
同じ物体・sample番号の初期参照姿勢と乱数シードを OOD と共有する。
設定は `configs/normal_five.json`。既存 OOD の結果は上書きしない。

## 通常条件

| 項目 | 条件 |
|---|---|
| 最終位置の x/y 変位 | 各 −0.10〜−0.05 m |
| 最終位置の z 変位 | 原則0 m。既存 OOD と同じ支持面の微小補正は適用・記録する |
| 中間の要求上昇量 | 0.15 m、ketchup は0.30 m |
| 向き | 世界 Z 軸 yaw ±0.40 rad |
| 関節角 | 絶対角0.50〜0.60 rad |
| 推論・物理計算 | CUDA、batch 100、CPU physics 24 threads |

目標分布は[論文 Supplementary B.4](https://arxiv.org/html/2309.03891v2)
と既存 OOD の `normal_goal` に合わせる。公開された元の runner には初期角を基準にした
相対関節角や ketchup 固有の範囲などの差があるため、元 runner の標準出力との数値一致を
意味しない。今回の100試行/物体は対応のある比較用の抽出であり、論文 Table 3 の
8物体・約7,500試行の全組合せを厳密再現するものではない。

## 評価指標

- **Suc.T**: 最終 PE < 0.05 m、SO(3) AE < 0.20 rad、AAE < 0.50 rad、
  かつ中間の上昇量誤差 < 0.05 m。既存 OOD と同じ判定を使う。
- **論文Suc.T**: 最終 PE・AE・AAE の3条件のみ。
- **PE / AE / AAE**: 正常に測定できた全試行の最終誤差平均。失敗も含め、クリップしない。
  単位は m / rad / rad。異常試行は成功率の分母に含め、誤差平均からは除外し測定数を記録する。

全10物体と論文対象8物体をそれぞれ集計する。`capsulemachine` と `phone` は論文対象外。
差分はすべて **OOD − normal**。成功率の差は percentage points（ポイント）、
誤差の差は元の単位で示す。誤差平均同士の差に加え、両条件とも有効な対応試行だけの差も別に残す。

## 実行

既存の Python 3.8.20 / numpy 1.23.1 / torch 2.4.1+cu121 /
ruamel.yaml 0.17.16 環境と RaiSim ライセンスが見える場所で実行する。

```bash
cd /home/cvrt/goto/artigrasp
bash scripts/run_normal_comparison.sh
```

既存 interpreter の既定パスは `/home/cvrt/miniconda3/envs/artigrasp/bin/python`。
別パスの場合は `ARTIGRASP_PYTHON=/path/to/python bash scripts/run_normal_comparison.sh` とする。
このスクリプトは依存環境と既存入力のハッシュを先に確認し、途中結果があれば同じ計画で再開する。
実験の途中で評価ソースを編集しないこと。

Singularity 内ではホストの既存環境全体を同じパスで共有し、
ライセンスを `$HOME/.raisim/activation.raisim` に見える状態にする。
今回の実行では `/home/cvrt/miniconda3/envs/artigrasp/bin/python` と
`$HOME/.raisim/activation.raisim` の共有・読み取りを確認済み。
ライセンス内容をチャットへ貼り付ける必要はない。

比較だけを再生成する場合:

```bash
python3 scripts/compare_normal_ood.py \
  --normal results/normal_seed1 --ood results/ood5_seed1 \
  --output results/normal_vs_ood_seed1
```

比較 CLI は標準ライブラリだけで動作する。未完了の実験、対応しない試行、
計算コード・バイナリ・重み・入力・実行条件が異なる実験を完成した比較として出力しない。

## 出力

- `results/normal_seed1/`: 全1,000試行、軌跡、計画、実行環境、normal の数値レポート。
- `results/normal_vs_ood_seed1/report.html` / `report.md`: 全体と物体別の比較表。
- 同ディレクトリの `comparison.csv`, `per-object.csv`, `paired_comparison.csv`, `summary.json`:
  指標、normal との差、対応試行の内訳と監査情報。

normal の実測値が存在しない段階では、normal の精度や比較差分を記載しない。
