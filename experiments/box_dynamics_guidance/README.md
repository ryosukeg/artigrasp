**box の予測モデルによる行動補正実験**

既存の ArtiGrasp PPO を固定し、最終開閉で右手首へ小さな補正を加えます。補正後の物体の動きを予測するモデルを教師あり学習し、推論時に候補を比較して、開閉角 OOD の達成率が改善するか検証します。

作業記録は [WORK_LOG.md](WORK_LOG.md)、収集器の実装仕様は [DATA_COLLECTION.md](DATA_COLLECTION.md) に保存します。以下のコマンドはリポジトリのルートから実行します。

**現在の状態**

- **適応終了による600試行の比較を実行しました。** 最低3秒の開閉後に成功を0.5秒維持・物体が2秒停滞・上限10秒で評価を終了します。成功維持率はPPO／元モデル／差分の順にnormal 9%／10%／9%、OOD 20%／21%／18%。0037の差分モデルは全体16.11秒で成功維持に到達しました。[動画・角度推移と終了理由のHTML](runs/comparison_seed202_adaptive_v1/index.html)／[詳細](runs/comparison_seed202_adaptive_v1/ADAPTIVE_REVIEW.md)。従来の固定3秒成功率とは判定基準が異なります。手順9に再現方法を記載しています。
- **3方式の[定性比較HTML](runs/comparison_seed202_threeway_v1/qualitative/index.html)を作成しました。** normal2例・OOD小角1例・OOD大角3例の計6例で、同じ試行のPPO・元モデル補正・差分モデル補正を横並びの同期動画として確認できます。開閉区間への移動・ループ・スロー再生・終点静止画・角度曲線に対応しています。
- **3方式のnormal／OOD評価と独立監査が完了しました。** 各100条件×3方式、計600試行を新規実行しました。最終3条件成功率はnormalがPPO／元モデル／差分モデルの順に9%／11%／9%、OODが19%／19%／15%でした。固定3秒という時間制限で差分モデルのOOD改善は確認できませんでした。持上げ込み成功は全方式・両条件とも0/100です。 [3方式の比較結果](runs/comparison_seed202_threeway_v1/THREEWAY_REVIEW.md)／[独立監査](runs/comparison_seed202_threeway_v1/independent_audit.json)
- box 専用の補正なし動作確認設定と、最終比較の評価条件を用意しました。
- 既存 CLI で、事前学習済み PPO の実行と軌跡保存ができます。
- 補正付き収集器、学習用データの loader、教師あり学習に加え、学習済みモデルによる補正選択と対応比較を実装しました。
- 2 設定の依存環境・資産・計画生成を確認済みです。準備時の確認結果は [verification.json](verification.json) に保存しています。
- 補正なし 8 試行は実行・確認済みです。[確認結果](BASELINE_REVIEW.md)を参照してください。収集器のゼロ補正8試行も元の action・軌跡と完全一致し、±0.5／±1.0 の較正52試行も数値異常なく完了しました。
- **手順3〜5は完了しています。** 少量40試行の確認後、1,000試行（train800／val200）を収集し、本学習は59 epochで早期終了、最良は39 epochでした。有効窓はtrain12,000／val2,985です。[本学習結果](checkpoints/box_response_1000_seed301_full/report.json)のval角度RMSEは0.1405 rad、状態維持予測は0.1153 radで、予測精度の改善は未達です。
- **手順6の初回比較と監査も完了しました。** 2026年11月11日 JST、最良モデルを再学習せず、公式testの通常100件・開閉角OOD100件を2方式で実行しました。OODの最終3条件成功は19/100→19/100、平均角度誤差は1.2643→1.2271 rad、角度0.1 rad以内は11/100→7/100でした。今回の設定で明確なOOD改善は確認できません。通常の最終3条件成功は9/100→11/100、持上げ込み成功は両条件・両方式とも0/100です。[比較結果](runs/comparison_seed202_v1/COMPARISON_REVIEW.md)を参照してください。
- 全200対で補正前の状態・指令・目標が完全一致しました。独立監査で400試行の行動・6,000窓・指標を再確認し、代表600窓の再推論も保存値と完全一致。数値異常・ゼロfallbackは0件です。[独立監査](runs/comparison_seed202_v1/independent_audit.json)
- 代表 3 試行の [確認動画](runs/baseline_smoke_seed101/qualitative/index.html) を作成しました。最終開閉への移動とスロー再生ができます。保存行動を RaiSim で再実行した結果は、3 件とも元の軌跡と完全に一致しました。

**実験条件**

| 項目 | 初期設定 |
|---|---|
| 物体 | box のみ |
| 既存方策 | 公開 checkpoint を固定。box の右手は最終開閉でも `cooperative_right` |
| 補正する部分 | 最終開閉の右手首の並進 3 成分 |
| 補正しない部分 | 前段の把持・移送、左手、右手首回転、指関節 |
| シミュレーション | 既存の複合タスクを最後まで実行。全 1,560 step |
| 補正を入れる区間 | 0 始まりの step 1,260〜1,559。右手 mode 8 の 300 step |
| 予測期間 | 最初は 20 step、0.2 秒。1 試行で最大 15 個の重複しない窓 |
| 収集用参照 | 公式 train split。学習・検証は参照元単位で分割 |
| 最終評価用参照 | 公式 test split。学習や設定調整に使わない |
| 収集時の目標角 | 0.5〜1.5 rad。収集専用分布として明示的に設定 |
| 本収集の初期補正候補 | ゼロと各軸±1.0 action単位の7候補を窓ごとに一様抽出 |
| 収集・参照分割 seed | 301。公式 train 内の14元シーケンス／4元シーケンスに分割 |
| 最終評価 | normal 0.5〜0.6 rad と、開閉角 OOD。各 100 試行を PPO・元モデル・差分モデルの 3 方式で比較（手順6b） |

収集用の目標範囲は公開 PPO 学習コードの範囲に合わせています。既存の normal 評価の目標範囲 0.5〜0.6 rad とは異なります。`baseline_smoke.json` は後者による補正なし動作確認用で、本収集の設定ではありません。

OOD は `[0.2, 0.4] ∪ [1.6, 3.1]` rad と、box の関節可動域から両端 0.05 rad を除いた範囲の共通部分です。今回は `factorial_v1` を使い、小角度側も含めます。抽出確率は区間の長さに比例するため、小角度側と大角度側が同数になる設定ではありません。結果は両側を分けても集計します。以前の `ood_five.json` の `articulation` 条件とは目標分布が異なるため、過去の成功率をそのまま対照群にしません。

**コンテナでは環境の Python を直接指定する**

現在の [Singularity 設定](../../../.singularity/container.yaml) は、ホストの `/home/cvrt/miniconda3/envs/artigrasp` だけを読み取り専用で共有しています。Conda 本体や `conda.sh` はコンテナから見えないため、`conda activate artigrasp` は `conda: command not found` になります。仮想環境の Python とインストール済みパッケージは利用できます。

以下の手順では `conda activate` を実行せず、`ARTIGRASP_PYTHON` に環境の Python を指定します。この方法で NumPy・PyTorch の import、CUDA の認識、box 用の `Runtime check: OK` を確認済みです。既存環境を使う今回の実験に、Conda の再インストールは必要ありません。

Conda の activate は PATH の変更に加えて環境固有の初期化スクリプトも実行します。Python の直接指定がすべての環境で activate と同じ処理をするわけではなく、ここでは今回の依存関係で確認した実行方法を使います。[Conda 公式説明](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment)

**1 既存環境を確認する**

```bash
cd /home/cvrt/goto/artigrasp
ARTIGRASP_PYTHON=/home/cvrt/miniconda3/envs/artigrasp/bin/python
BOX_EXPERIMENT=experiments/box_dynamics_guidance

"$ARTIGRASP_PYTHON" tools/ood_eval.py check \
  --config "$BOX_EXPERIMENT/configs/baseline_smoke.json"
```

`Runtime check: OK` を確認します。これは Python 依存関係、拡張 module、checkpoint、参照資産の確認で、シミュレーションを起動する処理ではありません。

**2 補正なしで 8 試行を実行する**

この工程は 2026年11月7日 JST に完了・確認済みです。`runs/baseline_smoke_seed101` は既に存在するため、同じ出力先で `plan` を再実行しません。以下は初回に使う手順として残します。新しく再実行する場合は別の出力先を指定します。

まず参照・目標の計画を作り、続いて保存した計画を実行します。次の `run` は実際にシミュレーションを開始します。

```bash
"$ARTIGRASP_PYTHON" tools/ood_eval.py plan \
  --config "$BOX_EXPERIMENT/configs/baseline_smoke.json" \
  --output "$BOX_EXPERIMENT/runs/baseline_smoke_seed101"

"$ARTIGRASP_PYTHON" tools/ood_eval.py run \
  --config "$BOX_EXPERIMENT/configs/baseline_smoke.json" \
  --output "$BOX_EXPERIMENT/runs/baseline_smoke_seed101" \
  --resume --device cuda --batch-size 8 --num-threads 4
```

`plan` は空の出力先で一度だけ実行します。中断後は同じ `run --resume` を実行します。コード・入力資産・実行 device を変更した場合は、別の出力先で計画を作り直します。

保存結果の `episodes.jsonl`、`trajectories/*.npz`、`performance/*.json`、`report.md` を確認します。NPZ には状態と左右の実行行動、初期化用参照が含まれます。この段階は事前学習モデルの動作と保存形式の確認です。8 試行の成功率から性能を判断しません。

**3 補正付きデータ収集器を実装し 少量で確認する**

この工程は完了・確認済みです。以下のゼロ8試行、較正52試行、学習用40試行、確認学習の成果物は既に存在します。[結果と限界](COLLECTOR_REVIEW.md)を確認してください。

[DATA_COLLECTION.md](DATA_COLLECTION.md) に従い、PPO の出力と `env.step` の間へ補正を追加し、開始観測・補正量・実測結果を保存します。まず 100 試行程度で次を確認します。

- 補正ゼロの収集器が、同じ参照・目標・seed の既存動作を再現する。
- 補正が最終開閉だけに入り、指定した action 成分へ指定した値が加わる。
- 補正方向と物理的な手首目標の変化が対応し、実際の物体の動きに測定できる差が出る。
- 20 step の開始・終了時刻が正しく対応し、reset や数値異常をまたぐ窓が教師データへ混ざらない。

手首並進 action の 1 単位は約 1 mm の制御目標の差に対応します。実際の手首が必ず 1 mm 動くという意味ではありません。補正量はこの確認で決め、決定値を作業記録へ残してから本収集へ進みます。

実装は [collect.py](collect.py)、計画生成は [planning.py](planning.py)、読み戻しは [dataset.py](dataset.py) です。以下は再現用のコマンドです。既に計画が存在する出力先では `plan` を再実行せず、`run` だけ実行します。`run` は完了 batch を hash 確認して再利用します。コード・資産・device を変える場合は新しい出力先を使います。

```bash
# 既存8試行と同じ参照・目標・seedで補正ゼロを確認
"$ARTIGRASP_PYTHON" "$BOX_EXPERIMENT/collect.py" plan --kind zero \
  --output "$BOX_EXPERIMENT/runs/collector_zero_seed101_v2" --batch-size 8
"$ARTIGRASP_PYTHON" "$BOX_EXPERIMENT/collect.py" run \
  --output "$BOX_EXPERIMENT/runs/collector_zero_seed101_v2"
"$ARTIGRASP_PYTHON" "$BOX_EXPERIMENT/audit_collection.py" \
  --dataset "$BOX_EXPERIMENT/runs/collector_zero_seed101_v2" \
  --baseline "$BOX_EXPERIMENT/runs/baseline_smoke_seed101"

# 4初期条件 × (ゼロ + 3軸 × 正負 × 0.5/1.0) = 52試行
"$ARTIGRASP_PYTHON" "$BOX_EXPERIMENT/collect.py" plan --kind calibration \
  --output "$BOX_EXPERIMENT/runs/collector_calibration_seed101_v2" --batch-size 52
"$ARTIGRASP_PYTHON" "$BOX_EXPERIMENT/collect.py" run \
  --output "$BOX_EXPERIMENT/runs/collector_calibration_seed101_v2"
"$ARTIGRASP_PYTHON" "$BOX_EXPERIMENT/audit_collection.py" \
  --dataset "$BOX_EXPERIMENT/runs/collector_calibration_seed101_v2"

# 各20step窓で補正を抽出。学習32試行、検証8試行
"$ARTIGRASP_PYTHON" "$BOX_EXPERIMENT/collect.py" plan --kind random \
  --episodes 40 --amplitude 1.0 --seed 301 --batch-size 20 \
  --output "$BOX_EXPERIMENT/datasets/box_response_smoke_seed301"
"$ARTIGRASP_PYTHON" "$BOX_EXPERIMENT/collect.py" run \
  --output "$BOX_EXPERIMENT/datasets/box_response_smoke_seed301"
"$ARTIGRASP_PYTHON" "$BOX_EXPERIMENT/audit_collection.py" \
  --dataset "$BOX_EXPERIMENT/datasets/box_response_smoke_seed301"

# 新しい出力先へ、読み込み・教師あり学習・検証が通るか確認
"$ARTIGRASP_PYTHON" "$BOX_EXPERIMENT/train_smoke.py" \
  --dataset "$BOX_EXPERIMENT/datasets/box_response_smoke_seed301" \
  --output "$BOX_EXPERIMENT/checkpoints/smoke_seed301" --epochs 5
```

ゼロ・較正試行は予測モデルの train/val に混ぜません。学習入力は571成分、教師は0.2秒後の物体状態9成分です。確認用学習は128ユニット×2層で、検証用参照を使った推論までを確認する用途です。精度やOOD改善は、この完走だけでは判断しません。

較正の応答差は、同じ52並列内の対応するゼロ補正との差です。並列数を変えると初期actionの微小差が物理軌跡の差につながる例があるため、8並列baselineとの直接比較に置き換えません。今後の性能比較では並列数・batch構成も揃えます。

**4 学習用データを収集する**

最初は合計 1,000 試行を目安に、train split 内の学習用・検証用参照を先に分けて収集します。成功・失敗を両方残し、補正ゼロも含めます。必要なら 3,000 試行へ拡張します。1,000 試行なら最大約 15,000 件、3,000 試行なら最大約 45,000 件の 20 step 応答が得られます。

同じ参照は繰り返し抽出されるので、試行数は独立した参照数ではありません。同じ試行の窓や同じ参照元を学習側と検証側へ無作為に散らしません。test split と最終 OOD 目標は本収集に混ぜません。

**以下は完了済みの本収集・本学習の再現コマンドです。** 学習800／検証200試行の成果物は既に存在します。seed301では参照元の分割を維持しています。新しい設定で収集・学習する場合は別の `--dataset`／`--output` を指定します。

```bash
cd /home/cvrt/goto/artigrasp
/home/cvrt/miniconda3/envs/artigrasp/bin/python -u \
  experiments/box_dynamics_guidance/run_training.py \
  --episodes 1000 --epochs 100 --patience 20 --device cuda
```

この1コマンドが以下を順に行い、途中で失敗した場合は後の工程へ進みません。

1. 目標角0.5〜1.5 rad、20並列、補正±1.0 action単位で1,000試行を収集。
2. 保存した補正、応答窓、参照分割、hashを検証。
3. 256ユニット×3層の予測モデルを、最大100 epoch学習。検証誤差が20 epoch改善しなければ早期終了。

最初は `Collect` に収集進捗、保存済み試行数、経過時間、残り時間（ETA）が表示されます。単位の `control steps` は全試行で合計した制御step数です。既存収集器からログが届く500 stepごと・batch完了ごとに更新されます。次の `Predictor training` ではepoch進捗と `train`、`val`、`best`、蓋角度RMSEが毎epoch更新されます。端末以外への出力では上書きバーの代わりに行単位のログになります。

`train` と `val` は全9出力の正規化MSEです。trainが下がるだけでは精度改善と判断せず、未学習の参照によるvalも確認します。`best=...@12` はepoch12の検証誤差がそれまでで最小という意味です。学習のETAは最大100 epochまでの目安で、早期終了すればそれより早く終わります。早期終了時に進捗表示が100%未満でも正常です。

中断した場合は、**同じコマンドを再実行**します。完了済みの収集batchを検証して再利用し、学習は最後に保存したepochから再開します。処理中だった未完了batch／epochはやり直します。再開時はseed、device、並列数、モデル設定、コード・データを変更しません。変更する場合は新しい `--dataset`／`--output` を指定します。まだ `last.pt` が作られる前の読み込み失敗等では、原因を解消したうえで新しい `--output` を使います。

出力先は次のとおりです。

- 収集データ：`datasets/box_response_1000_seed301/`
- 学習結果：`checkpoints/box_response_1000_seed301_full/`
- 収集・検証の詳細ログ：`logs/box_response_1000_seed301__box_response_1000_seed301_full/`

すべてこの実験ディレクトリからの相対パスです。起動せずに設定・コマンドだけ確認する場合は、上のコマンド末尾に `--dry-run` を付けます。OOD目標での収集や推論時の候補選択・比較評価は、このコマンドには含めていません。

収集だけを個別に実行する場合のコマンドも残します。既に計画がある場合、`plan` は省略します。

```bash
"$ARTIGRASP_PYTHON" "$BOX_EXPERIMENT/collect.py" plan --kind random \
  --episodes 1000 --amplitude 1.0 --seed 301 --batch-size 20 \
  --output "$BOX_EXPERIMENT/datasets/box_response_1000_seed301"
"$ARTIGRASP_PYTHON" "$BOX_EXPERIMENT/collect.py" run \
  --output "$BOX_EXPERIMENT/datasets/box_response_1000_seed301"
"$ARTIGRASP_PYTHON" "$BOX_EXPERIMENT/audit_collection.py" \
  --dataset "$BOX_EXPERIMENT/datasets/box_response_1000_seed301"
```

**5 小さな予測モデルを学習する**

開始観測・目標・補正量から、20 step 後の関節角、角速度、物体基部の位置・姿勢を予測します。PPO の重みは更新しません。[train_model.py](train_model.py) の既定設定は256ユニット×3層のMLP、最大100 epoch、batch size 256、Adamの学習率0.001、seed301です。正規化の統計量は学習データだけから求めます。

早期終了は、val正規化MSEが前の改善基準から0.0001を超えて改善しないepochが20回続いた場合です。一方、`best.pt` には改善幅にかかわらずval正規化MSEが最小のepochを保存します。最終epochと最良epochは同じとは限りません。

手順4の一括コマンドを使えば、学習の別操作は不要です。収集が既に終わっており、学習だけを開始する場合は次を使います。中断後は同じコマンドに `--resume` を追加します。

```bash
/home/cvrt/miniconda3/envs/artigrasp/bin/python -u \
  experiments/box_dynamics_guidance/train_model.py \
  --dataset experiments/box_dynamics_guidance/datasets/box_response_1000_seed301 \
  --output experiments/box_dynamics_guidance/checkpoints/box_response_1000_seed301_full \
  --epochs 100 --patience 20 --device cuda
```

| 保存ファイル | 内容 |
|---|---|
| `best.pt` | 検証誤差が最小のモデル、正規化値、設定 |
| `last.pt` | 最後に完了したepochのモデル・optimizer・乱数状態。再開用 |
| `history.jsonl` | epochごとのtrain/val loss、角度・位置RMSE、時間 |
| `status.json` | 現在の状態、完了epoch、早期終了やエラーの情報 |
| `report.json` | 最良モデルの予測誤差と、現在状態をそのまま予測する基準との比較 |

完了後は `report.json` の `val_model` と `val_persistence` を比較します。学習処理の完了は、予測精度やOOD達成率の改善を意味しません。少量確認と本学習では試行数とモデル規模も変わるため、その差をepoch数だけの効果とは解釈しません。

本学習59 epochの [lossグラフ（PNG）](checkpoints/box_response_1000_seed301_full/plots/loss_curve.png)、[PDF](checkpoints/box_response_1000_seed301_full/plots/loss_curve.pdf)、[SVG](checkpoints/box_response_1000_seed301_full/plots/loss_curve.svg)、[数値CSV](checkpoints/box_response_1000_seed301_full/plots/loss_history.csv) を保存しました。青は更新中のミニバッチ訓練lossの平均、橙は各epoch終了時の検証loss、横破線は検証データでの現在状態維持予測です。縦点線は最良の39 epochで、59 epochで早期終了しています。平滑化はしていません。

学習対象は800試行の最終開閉300 stepを20 stepずつにまとめた12,000遷移で、各窓の開始観測から0.2秒後の状態を予測します。中間フレームを逐次入力する動画・系列モデルではありません。571入力・256×3層・9出力のMLPは280,329パラメータで、59 epochの実更新回数は2,773回。記録された65.64秒はデータ読込・検証・保存等を含み、epoch内の訓練と検証の合計は30.22秒です。

図は [plot_training_history.py](plot_training_history.py) から再生成できます。Matplotlib 3.7.5を学習環境とは別の一時ディレクトリに用意した場合のコマンドです。

```bash
/home/cvrt/miniconda3/envs/artigrasp/bin/python -m pip install \
  --target /tmp/box_plot_deps_20261111 'matplotlib==3.7.5'
PYTHONPATH=/tmp/box_plot_deps_20261111 \
  /home/cvrt/miniconda3/envs/artigrasp/bin/python \
  experiments/box_dynamics_guidance/plot_training_history.py \
  --checkpoint-dir experiments/box_dynamics_guidance/checkpoints/box_response_1000_seed301_full
```

未使用の参照による検証で、現在状態をそのまま予測する基準より良いか、補正の効果を予測できるかを確認します。候補の順位を検証する場合は、同じ初期条件・同じ前段から補正だけ変えた比較データを別途用意します。完全な途中状態復元は現行 API にありません。

**5b 既存データで角度・位置の差分モデルを学習する（完了）**

既存1,000試行から「0.2秒後 − 現在」の蓋角度・基部位置を教師として保存し、元と同じ参照分割・12,000訓練遷移・2,985検証遷移・MLP規模・seed301・学習設定で新規学習しました。角速度とquaternionは絶対値のままです。元の出力標準偏差を維持して損失の重みを揃え、比較時は現在状態を加えて絶対状態へ戻しています。追加収集やOOD目標の追加はありません。

78 epochで早期終了、best58、教師保存・読込等を含む学習処理は76.76秒。元モデルに対して角度RMSEは0.140528→0.093844 rad、位置成分RMSEは0.103232→0.016503 m、全体MSEは0.161677→0.130830に改善しました。ただし状態維持予測の全体MSE 0.107181には届いていません。差分モデルによる候補選択・OOD成功率の評価は手順6bを参照してください。

[詳細・再現コマンド](checkpoints/box_response_1000_seed301_delta_angle_position/DELTA_REVIEW.md)、[lossグラフ](checkpoints/box_response_1000_seed301_delta_angle_position/plots/loss_curve.png)、[再評価JSON](checkpoints/box_response_1000_seed301_delta_angle_position/absolute_vs_delta.json) を保存しています。教師は同出力の `targets/train.npz` / `targets/val.npz`、重みは `best.pt` です。

入口は [train_delta.py](train_delta.py)。収集済みdatasetを読み、教師保存→差分学習→検証を行い、epoch進捗を表示します。[compare_delta.py](compare_delta.py) は両モデルを再ロードして条件一致と誤差を照合します。差分checkpoint用adapterの接続と3方式評価は手順6bを参照してください。

**6 学習済みモデルで補正を選び 同じ条件で比較する**

実装は [guidance.py](guidance.py)、実行は [evaluate_guidance.py](evaluate_guidance.py)、集計は [evaluate_report.py](evaluate_report.py) です。補正なしと各並進軸±1.0 action単位の計7候補から、最小の予測コストを選びます。初回比較ではパラメータ探索を行わず、タスク成功の閾値を尺度とする次の式を評価前に固定しました。

`cost = (角度誤差 / 0.5 rad)^2 + (基部位置誤差 / 0.05 m)^2 + (基部姿勢誤差 / 0.2 rad)^2 + 0.01 * ||補正||^2`

姿勢誤差は予測quaternionを正規化したSO(3)距離です。元の最終目標に対して採点し、同点はゼロ補正を優先します。無効候補は選ばず、全候補が無効ならゼロへ戻します。

当初案の「数stepで再選択」から、今回は学習データと一致する **20step保持・20stepごとの再選択** にしました。step1260の手のreset後の観測・物体状態・元目標を入力し、PPO自体は毎step再計算します。左右の実行指令、PPO出力、補正、状態、全候補の予測・costをNPZへ保存します。

`configs/evaluation.json` は通常と開閉角OOD各100件、計200件の条件です。同じ計画を2方式で実行し計400試行を比較します。PPOは100並列・物理24thread・CUDA・seed202、各条件内の試行順も両方式で一致させます。候補モデルの700入力とPPOの100並列は別です。

```bash
cd /home/cvrt/goto/artigrasp
# 初回だけplan。既存の出力先を上書きしない。
/home/cvrt/miniconda3/envs/artigrasp/bin/python experiments/box_dynamics_guidance/evaluate_guidance.py plan \
  --output experiments/box_dynamics_guidance/runs/comparison_seed202_v1
# 既存計画の実行／再開
/home/cvrt/miniconda3/envs/artigrasp/bin/python -u experiments/box_dynamics_guidance/evaluate_guidance.py run \
  --output experiments/box_dynamics_guidance/runs/comparison_seed202_v1
```

完了batchはhashを確認して再利用します。計画・実装・PPO・モデル・sim資産を変えた場合は、新しい出力先を使います。現在の `comparison_seed202_v1` は計画済みなので、上の `plan` は再実行しません。`report --output ...` は保存済み結果から再集計するだけで、simを再実行しません。

`COMPARISON_REVIEW.md` と `summary.json` に、持上げ込み成功率、最終位置・姿勢・角度の同時達成率、角度のみの成功率、最終誤差、異常数、補正頻度を出します。小角OODは初期状態で角度誤差0.5 rad以内になり得るので、開始時の達成件数と厳しい0.1 rad閾値も併記します。全200対の補正前状態・指令・reset後状態・目標の一致を監査し、不一致時には改善の結論を保留します。

これは追加simで学習した補正機構の探索的な単一seed比較です。モデルの候補順位精度は未検証で、PPO単体と学習予算を揃えた比較でもありません。今回のtest結果を見て採点重みを調整しません。

**6b PPO・元モデル・差分モデルの3方式比較**

2026年11月11日、実行・集計・独立監査まで完了しました。[比較結果](runs/comparison_seed202_threeway_v1/THREEWAY_REVIEW.md)と[数値CSV](runs/comparison_seed202_threeway_v1/metricstable.csv)を保存しています。以下は再現・再集計の手順です。

[evaluate_threeway.py](evaluate_threeway.py) で、PPO単体（`baseline`）、元の絶対値モデルによる補正（`guided_absolute`）、差分モデルによる補正（`guided_delta`）を比較します。公式testのnormal100件と開閉角OOD100件を各方式で実行するため、合計600試行です。参照・目標・seed202・試行順・100並列・物理24thread・CUDAを揃え、前段から各方式を実行します。200件の条件は手順6の計画とも同一です。

候補はゼロと各軸±1.0 action単位の7候補、scoreは手順6の式をそのまま使い、最終開閉のstep1260〜1559で20step保持・20stepごとに再選択します。両モデルの入力、候補、採点尺度を固定し、この比較のための再学習や設定調整は行いません。

差分モデルは専用の [guidance_delta.py](guidance_delta.py) で読み込みます。モデル出力を逆正規化した後、その窓の開始時点の角度・基部位置を加え、0.2秒後の絶対状態へ戻してから共通のscoreへ渡します。角速度・quaternionは元から絶対値です。専用checkpointの形式・特徴量・教師定義を検証し、元のabsolute用Selectorとの取り違えを拒否します。

出力先は `runs/comparison_seed202_threeway_v1` です。**この出力先には既に計画があるため、`plan` は再実行しません。** 以下の初回計画コマンドは再現用です。新たに一式を実行する場合は、空の別出力先へ変更してから使います。

```bash
cd /home/cvrt/goto/artigrasp
# 初回だけplan。現在の出力先では実行しない。
/home/cvrt/miniconda3/envs/artigrasp/bin/python experiments/box_dynamics_guidance/evaluate_threeway.py plan \
  --output experiments/box_dynamics_guidance/runs/comparison_seed202_threeway_v1 \
  --samples 100 --seed 202 --batch-size 100 --num-threads 24 --device cuda
```

既存計画の実行と中断後の再開には、同じ `run` を使います。実行中の処理と二重起動せず、停止を確認してから再開します。完了batchはhashを確認して再利用し、未完了batchだけを実行します。計画・実装・モデル・PPO・sim資産を変える場合は別の出力先を使います。

```bash
/home/cvrt/miniconda3/envs/artigrasp/bin/python -u experiments/box_dynamics_guidance/evaluate_threeway.py run \
  --output experiments/box_dynamics_guidance/runs/comparison_seed202_threeway_v1
```

全batchの実行後、保存結果を再集計する場合と、独立監査を実行する場合のコマンドです。どちらも物理シミュレーションは再実行しません。監査では保存された観測から両モデルをCUDAで再推論し、候補予測・score・選択結果も照合します。

```bash
/home/cvrt/miniconda3/envs/artigrasp/bin/python experiments/box_dynamics_guidance/evaluate_threeway.py report \
  --output experiments/box_dynamics_guidance/runs/comparison_seed202_threeway_v1
/home/cvrt/miniconda3/envs/artigrasp/bin/python experiments/box_dynamics_guidance/audit_threeway.py \
  --output experiments/box_dynamics_guidance/runs/comparison_seed202_threeway_v1
```

集計は [report_threeway.py](report_threeway.py)、独立監査は [audit_threeway.py](audit_threeway.py) です。`THREEWAY_REVIEW.md` と `summary.json` で成功率・最終位置誤差（PE）・関節角度誤差（AAE）・基部姿勢誤差（AE）を方式別に比較します。normalとOOD、小角OODと大角OODを分け、同じ条件で成否が変わった件数も確認します。`independent_audit.json` では資産と軌跡のhash、各方式の補正前状態・指令・目標の一致、補正範囲、保存指標、代表窓の再推論を確認します。

**夜間実行の準備と時間**

手順4の一括コマンドで、収集 → データ検証 → 学習までを連続実行できます。本学習は完了済みです。学習済みモデルを使った比較評価の実行・再開は手順6、3方式比較は手順6bを参照してください。

前景で実行する場合はターミナルを開いたまま、PCがスリープしない状態にします。端末を閉じても継続させたい場合は次のように起動し、`tail -f` で進捗ログを見られます。前景コマンドとの二重起動は避けます。`tail` をCtrl+Cで閉じても、バックグラウンドの学習は継続します。

```bash
cd /home/cvrt/goto/artigrasp
mkdir -p experiments/box_dynamics_guidance/logs
BOX_TRAIN_LOG="experiments/box_dynamics_guidance/logs/full_$(date +%Y%m%d_%H%M%S).log"
nohup /home/cvrt/miniconda3/envs/artigrasp/bin/python -u \
  experiments/box_dynamics_guidance/run_training.py \
  --episodes 1000 --epochs 100 --patience 20 --device cuda \
  > "$BOX_TRAIN_LOG" 2>&1 < /dev/null &
tail -f "$BOX_TRAIN_LOG"
```

今回の収集器では、20並列×2batchの40試行で、準備・物理実行・保存に計116.5秒かかりました。固定した20並列の速度からは1,000試行で約49分の単純見積もりとなり、初回は収集だけで50〜60分程度を目安にします。参照・接触状態・PC負荷で変わります。40試行データの読み込みを含む5 epoch確認学習は約4.1秒でした。本学習・精度調整・比較評価の時間は別途測定します。

既存コードの確認先は [runtime.py](../../tools/ood_suite/runtime.py)、[targets.py](../../tools/ood_suite/targets.py)、[評価 CLI](../../tools/ood_eval.py) です。

**3方式の比較動画を見る**

[normal／OODの3方式比較HTML](runs/comparison_seed202_threeway_v1/qualitative/index.html)をブラウザで開きます。サーバーや追加インストールは不要です。左側でnormal／OODやケースを選び、「12.6秒から」で最終開閉へ移動します。0.5倍・0.25倍、開閉区間のループ、「終点の静止画」に対応しています。角度グラフをクリックして、その時刻へ移動することもできます。

同じ試行の3方式を1本の動画へ合成したため、列間の再生時刻は一致します。各15.6秒、20fps、1,920×880px、312フレームで、6動画の合計は約7.8MBです。上段は物体中心に追従する同倍率の拡大表示、下段左は3方式共通の固定視点です。基部の位置ずれは固定視点とPEで確認します。赤系が右手、青が左手、金が物体、半透明緑が最終目標です。

normal2例・OOD小角1例・大角3例を選びました。成功獲得・成功喪失・共通失敗を見比べるための層別例で、出現頻度を再現した標本ではありません。選定規則と注目点は各ケースに表示し、下部に全600試行の集計を併記しています。最終3条件の成功と持上げ込み成功は区別しています。

動画は独立監査済みの保存状態とURDFから描画したものです。物理シミュレーションや方策推論を再実行していません。step1260の手のreset後状態を反映し、最後のフレームと終点画像には実際の終端状態を使っています。HTMLの時刻と角度カーソルも表示フレームの物理時刻に対応させています。

[manifest.json](runs/comparison_seed202_threeway_v1/qualitative/manifest.json)に選定理由・カメラ・フレーム対応・hash、[メディア検証](runs/comparison_seed202_threeway_v1/qualitative/media_verification.json)に全動画decodeと18軌跡の数値照合、[ブラウザ検証](runs/comparison_seed202_threeway_v1/qualitative/browser_verification.json)に再生・操作・表示の確認結果を保存しています。

生成済みなので再実行は不要です。同じ保存結果から再生成する入口は以下です。既存の動画はsource・設定・軌跡・動画と画像のhashを照合して再利用します。評価定義・結果を変更する場合は新しいrunディレクトリで生成します。

```bash
cd /home/cvrt/goto/artigrasp
/home/cvrt/miniconda3/envs/artigrasp/bin/python -u \
  experiments/box_dynamics_guidance/render_threeway.py \
  --results experiments/box_dynamics_guidance/runs/comparison_seed202_threeway_v1
```

**確認動画を見る**

[動画一覧](runs/baseline_smoke_seed101/qualitative/index.html) をブラウザで開きます。`0002` は最終位置・姿勢・角度の条件を達成した例、`0000` は蓋が動く一方で基部がずれた例、`0004` は持ち上げ後の最終開閉が未達の例です。各 15.6 秒で、最終開閉は 12.6 秒からです。

現在のコンテナはホスト GUI を共有していないため、動画は保存された手・物体の物理状態と URDF 形状を EGL で描画しています。Unity の画面録画ではありません。保存行動を別途 RaiSim で数値再生し、3 試行とも全状態と成功分類の一致、最大状態差 0.0 を確認しています。

動画内の `FAILURE` は持ち上げ込みの複合条件に対する表示です。`0002` も持ち上げ条件は未達なので、その表示になります。最終 3 条件の成否は動画一覧に別記しています。

描画用パッケージは `.render_deps/` に分離し、既存の ArtiGrasp 環境は変更していません。動画を再生成する入口は以下です。再生検証 JSON が `qualitative/replays/` に存在することを確認してから描画します。

```bash
/home/cvrt/miniconda3/envs/artigrasp/bin/python \
  experiments/box_dynamics_guidance/render_selected.py
```

描画依存関係を同じ隔離先へ用意する場合は、次を使います。

```bash
/home/cvrt/miniconda3/envs/artigrasp/bin/python -m pip install \
  --target experiments/box_dynamics_guidance/.render_deps \
  -r experiments/box_dynamics_guidance/requirements-render.txt
```

新しい試行を追加するときの数値再生の例です。結果の `verified` を確認してから、描画入口の `--episodes` へその ID を指定します。

```bash
/home/cvrt/miniconda3/envs/artigrasp/bin/python tools/ood_eval.py replay \
  --results experiments/box_dynamics_guidance/runs/baseline_smoke_seed101 \
  --episode-id box__normal__0002 \
  --replay-result experiments/box_dynamics_guidance/runs/baseline_smoke_seed101/qualitative/replays/box__normal__0002.json
```

全動画の元軌跡 hash、MP4 hash、フレーム数、再生検証は [manifest.json](runs/baseline_smoke_seed101/qualitative/manifest.json) に保存しています。描画方式の仕様は [Pyrender のオフスクリーン描画](https://pyrender.readthedocs.io/en/latest/examples/offscreen.html)を参照してください。

## 9 適応的な終了判定で比較する

従来の15.6秒終了（最終開閉は12.6〜15.6秒の3秒間）を見直すため、[evaluate_adaptive.py](evaluate_adaptive.py) を追加しました。元の三方式比較を参照し、参照・目標・PPO・予測モデル・7候補・0.2秒の再選択周期・バッチ構成を維持します。追加の学習は行いません。

開閉開始を0秒として、最低3秒は実行し、以下を100Hzで判定します。

| 終了理由 | 条件 |
|---|---|
| 成功維持 | PE < 0.05m、AAE < 0.5rad、AE < 0.2radを同時に0.5秒間連続して満たす |
| 観測上の停滞 | 直近2秒間の蓋角度の最大値−最小値 < 0.02rad、基部位置の窓開始からの最大変位 < 2mm、基部姿勢の窓開始からの最大回転角 < 0.02radを全て満たす |
| 時間上限 | 開閉開始から10秒（全体22.6秒）までに上記に該当しない |
| 無効 | 非有限の状態・観測・行動、シミュレータ自動reset等。最低3秒を待たず無効終了 |

成功条件に入っていて0.5秒の維持確認中の場合は、停滞終了を抑制します。優先順は無効→成功維持→停滞→時間上限です。0.5秒は51状態・50区間、2秒は201状態・200区間です。一瞬の角速度や窓の始終差だけでは判定しないため、途中の往復・箱全体の移動も確認します。

停滞は「この観測窓で物体の変化が小さい」という診断であり、その先の成功が不可能であることは意味しません。手が接触を探している間や非常に遅い進行を打ち切る可能性は残ります。閾値は実用上の初期設定として固定し、今回のtest成功率で最適化しません。

成果物は [適応終了の比較HTML](runs/comparison_seed202_adaptive_v1/index.html)、[詳細](runs/comparison_seed202_adaptive_v1/ADAPTIVE_REVIEW.md)、[数値CSV](runs/comparison_seed202_adaptive_v1/metricstable.csv)、[独立監査](runs/comparison_seed202_adaptive_v1/independent_audit.json) です。HTMLでは200条件の角度推移と各方式の終了時刻・終了理由を確認できます。同じページに、以前と同じ6例（normal0017・0086、OOD0019・0041・0037・0053）の適応終了までの3方式比較動画を追加しています。初期表示の「動画ありのみ」を外すと全200条件へ戻れます。各動画は3方式共通の時間軸で再生し、先に終了した列は終了時刻・理由を明示して、その最後の保存状態で静止表示します。これは終了後の計測や動作の継続ではありません。開閉開始12.6秒への移動・速度変更・開閉ループ・終点画像・角度グラフとの時刻連動に対応します。従来動画は固定時間版のまま保存しています。

下記 `plan` は空の別出力先に対してだけ使います。既存計画の再開・再集計では `run` または `report` を使います。

```bash
cd /home/cvrt/goto/artigrasp
/home/cvrt/miniconda3/envs/artigrasp/bin/python experiments/box_dynamics_guidance/evaluate_adaptive.py plan \
  --source-run experiments/box_dynamics_guidance/runs/comparison_seed202_threeway_v1 \
  --output experiments/box_dynamics_guidance/runs/comparison_seed202_adaptive_v1
/home/cvrt/miniconda3/envs/artigrasp/bin/python -u experiments/box_dynamics_guidance/evaluate_adaptive.py run \
  --output experiments/box_dynamics_guidance/runs/comparison_seed202_adaptive_v1
/home/cvrt/miniconda3/envs/artigrasp/bin/python experiments/box_dynamics_guidance/audit_adaptive.py \
  --output experiments/box_dynamics_guidance/runs/comparison_seed202_adaptive_v1
/home/cvrt/miniconda3/envs/artigrasp/bin/python experiments/box_dynamics_guidance/evaluate_adaptive.py report \
  --output experiments/box_dynamics_guidance/runs/comparison_seed202_adaptive_v1
```

各試行は終了判定時点で評価を固定し、保存軌跡もそこで切ります。並列計算のバッチ形状を保つため、終了済み環境も他の環境が終わるまで内部の計算には残りますが、その後の状態・異常・結果を評価へ混ぜません。元の3秒時点を保存し、旧実験との状態・行動の一致を監査します。

「3秒時点の成功」「適応終端1時点の成功」「0.5秒維持した成功」は別々に報告します。成功時刻が方式間で異なるため、終端PE・AAE・AEの比較には観測時間の違いも含まれます。今回のモデルは3秒区間で収集した遷移から学習しており、延長後の状態分布で同じ予測精度が保証されるわけではありません。


適応終了の動画を再生成する場合は、シミュレーションを実行せず以下を使います。描画依存は既存の `.render_deps` に隔離され、元の訓練環境は変更しません。

```bash
/home/cvrt/miniconda3/envs/artigrasp/bin/python -u experiments/box_dynamics_guidance/render_adaptive.py \
  --results experiments/box_dynamics_guidance/runs/comparison_seed202_adaptive_v1
/home/cvrt/miniconda3/envs/artigrasp/bin/python experiments/box_dynamics_guidance/verify_adaptive_media.py \
  --results experiments/box_dynamics_guidance/runs/comparison_seed202_adaptive_v1
/home/cvrt/miniconda3/envs/artigrasp/bin/python experiments/box_dynamics_guidance/report_adaptive.py \
  --output experiments/box_dynamics_guidance/runs/comparison_seed202_adaptive_v1
```

同じ動画・素材・描画ソース・結果hashが揃えば描画を再利用します。動画と対応メタデータは `runs/comparison_seed202_adaptive_v1/qualitative/` に保存します。6例は旧固定時間版との追跡比較用で、新しい結果で選び直した例や成功頻度を代表する標本ではありません。
