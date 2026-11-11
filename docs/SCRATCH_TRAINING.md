# 公開policy重みを使わない学習

`scripts/train_scratch.py` は、左右の固定base学習から両手の自由base学習までを1つの学習seedで順に実行する入口です。既定では計画を表示するだけです。実際の学習には `--execute` が必要です。

## baselineの定義

`configs/training/scratch_baseline_v1.json` は、現在のリポジトリの報酬・ネットワーク・PPO実装を基準として、今後の比較条件を固定するための設定です。論文の性能や収束を厳密に再現した設定という意味ではありません。各stageの元の `cfg_reg.yaml` を `configs/training/scratch_baseline_v1/` に保存し、報酬係数を変えずに可視化・録画を無効にし、物理計算のthread数を32に揃えています。

- 学習対象: box、laptop、notebook、waffleiron、mixer、ketchup、espressomachine、microwaveの8物体。
- データ: 既存の処理済み参照姿勢のtrain split。評価用test splitは学習に使いません。
- 1つの学習seedで8物体をまとめて学習します。物体ごとに別のpolicyを学習する設定ではありません。
- 左右のactor・criticはランダム初期化から事前学習します。両手stageだけが、その実行で得られた左右の最終重みを引き継ぎます。
- 観測の標準化は無効です。学習用NumPy samplerと評価用seedを分離し、Python・NumPy・PyTorch・環境の乱数を設定します。各子processの起動前に `PYTHONHASHSEED` を学習seedで固定し、その内部ではstage別に乱数seedを導出します。PyTorchの決定的な演算を有効にします。同じ実行環境での再現を助ける設定であり、異なるCPU/GPU・ライブラリ間でのbit一致を保証するものではありません。
- 各stageは独立したoptimizerで開始します。両手stageは自作のactor・critic重みを引き継ぎます。学習途中からの再開は未対応です。
- この環境ではRaiSimとPyTorchがOpenMP設定を共有し、環境初期化時にPyTorchのthread数も変わります。baselineでは両者を32 thread、smokeでは両者を1 threadに揃えます。manifestには要求したthread数と環境初期化後の実測値を残します。
- PPO更新後に探索分散のNumPy cacheを同期します。非有限値と、学習環境で数値異常を意味する自動resetを検出した場合は停止し、完了checkpointとして採用しません。

| stage | 内容 | iteration | 並列環境数 | 環境step数 |
|---|---|---:|---:|---:|
| `left_fixed` | 左手・固定base | 20,001 | 1,000 | 6,000,300,000 |
| `multi_obj_arti` | 右手・固定base | 50,001 | 1,408 | 21,120,422,400 |
| `general_two` | 両手・自由base | 9,001 | 2,000 | 5,400,600,000 |

iteration数は現在のrunnerの既定値を固定したものです。右手の環境数は元の1,400から1,408へ調整し、8物体×2種類のtaskに均等配分しています。左手は物体別、右手・両手は物体とtaskの組合せ別に参照を抽出します。1 iterationは1環境あたり300 stepです。1 seedセットは合計32,521,322,400環境stepで、両手stageでは左右それぞれのPPOを更新します。環境step数は方策の勾配更新回数とは異なります。

[論文§4.4](https://arxiv.org/html/2309.03891v2#S4.SS4)は左右別・固定baseから両手・自由baseへ移るカリキュラムを説明しています。[§6.1と付録A](https://arxiv.org/html/2309.03891v2#S6.SS1)にはRTX 6000 1枚＋128 CPU coresで約3日、主要なPPO設定の記載があります。公開runnerの既定値と付録の表には差があるため、この固定した学習予算の所要時間を論文の約3日から換算することはできません。手元のiteration当たり所要時間と学習曲線で確認してください。

## 準備と計画の確認

既存のArtiGrasp Python環境、ビルド済みの3学習環境、RaiSimのactivation key、`rsc/` 以下の処理済みデータと物体・手の資産を使います。policyの再学習には公開checkpointは不要です。手・データの資産を元データから作り直す場合は、別途MANO/ARCTICの準備が必要になります。

リポジトリのrootから実行してください。`ARTIGRASP_PYTHON` はArtiGraspを導入したPythonの絶対pathです。

```bash
ARTIGRASP_PYTHON=/home/cvrt/miniconda3/envs/artigrasp/bin/python
"$ARTIGRASP_PYTHON" scripts/train_scratch.py --seed 1
```

計画には、解決済みの実行command、設定・runnerのhash、各stageのstep数、出力先がJSONで表示されます。計画の表示は標準libraryだけで動作し、学習・結果directoryの作成・checkpointの読込みを行いません。

## 最小の起動確認

```bash
"$ARTIGRASP_PYTHON" scripts/train_scratch.py --smoke --seed 1 \
  --output results/training/scratch_baseline_v1_seed1_smoke
"$ARTIGRASP_PYTHON" scripts/train_scratch.py --smoke --seed 1 \
  --output results/training/scratch_baseline_v1_seed1_smoke --execute
```

`--smoke` は各stageを1 iterationだけ、CPU・物理1 thread・PyTorch 1 threadで実行します。環境数は左8・右16・両手16で、8物体を含めます。全体で12,000環境stepです。依存関係、データ読込み、PPO更新、更新後の保存、左右重みの引継ぎまでを確認する用途です。成功率や収束の確認には使えません。

## 1 seedセットの学習

```bash
"$ARTIGRASP_PYTHON" scripts/train_scratch.py --seed 1 \
  --output results/training/scratch_baseline_v1_seed1 --execute
```

既定のdeviceは `auto` で、利用可能ならCUDAを使います。CPUや明示的なCUDAで固定したい場合は、training JSONを複製して `runtime.device` を変更し、別名のbaselineと出力先で `--config` に渡してください。異なる条件の実験は、設定と結果を別々に保存してください。

各stageの標準出力・エラーを `<stage>.log` に保存します。`training_plan.json` に計画、`execution.json` に進捗と最終的な成否を保存します。計画でhashを記録したpipeline本体・元のtraining JSON・3つのrunner・設定snapshot・評価templateに加え、3つの `Environment.hpp` と実行する拡張binary (`.so`)、共通の学習helper・label generator・回転処理・frame定義・PPO実装・環境wrapper・C++共通headerを `pipeline_sources/` へコピーします。計画の `source_snapshots` に元の絶対pathとの対応を保存します。計画時には各学習stageのビルド済み `.so` が1つだけ存在することも確認しますが、Python moduleをimportしたり学習を起動したりはしません。

後日、実行に使った入口・設定・実装・binaryを参照できます。各stageの起動前と評価設定の生成前に、計画時に固定した全sourceとbinaryのhashを再確認し、前段の学習中に対象が編集・再ビルドされていた場合は後段を止めます。評価設定の生成には保存したtemplateを使用します。

各stageの `training_result.json` が `complete: true` に加えて、予定のstage名・学習seed・物体リスト・環境数・完了iteration数、具体的なdevice、PPO更新後の保存を示す `checkpoint_timing: after_completed_ppo_updates`、実在するそのstage内の最終checkpointを示したときだけ次段階へ進みます。途中でcommandが失敗した場合や完了記録が欠けた場合は後段を起動しません。

既存の出力directoryは、空でも拒否します。失敗後に同じ出力先で再実行することもできません。調査用の記録を保管し、修正後は新しい出力先を指定してください。長時間の本学習を再開する機能はありません。

旧runnerを直接起動する場合も、対象の既定値は8物体です。`--weight` は明示した重みの転送、協調学習は `--pretrained-left` と `--pretrained-right` の組で指定します。`-re` だけで公開checkpointを自動選択する処理はありません。`--load-optimizer` はoptimizerだけの追加転送で、RNGやPPOの学習率管理まで復元するresume機能ではありません。`full_N` はN回の更新を完了した後の重みを表します。評価は専用CLIを使用し、`-t`・`-debug`・`-evaluate`の代わりに少数の実バッチで起動確認してください。

複数の独立学習seedを比較する場合は、例えば `--seed 2` と別の出力先で3段階すべてを実行します。同じ事前学習重みを全seedで共有すると、全工程を独立に再学習する実験とは異なるため、区別して報告してください。

## 最終policyのnormal評価

3段階が完走すると、その実行の最終checkpoint3本（両手stageの左右、右手pretrain）を指す `normal_evaluation.json` を生成します。途中の `full_0` などをdirectoryから探索して選ぶ処理はありません。評価は自動実行しません。

生成する評価設定は `configs/normal_five.json` を基に、対象8物体・test split・normal条件とします。評価seedは学習seedとは別の `--eval-seed`（既定1）、1物体あたりの評価数は `--eval-samples-per-object`（既定100）で設定できます。複数の学習seedを同一評価ケースで比較する場合、評価seedと評価設定は揃えてください。生成される既定800試行は評価の入口であり、十分な統計精度を保証する数ではありません。

```bash
"$ARTIGRASP_PYTHON" tools/ood_eval.py check \
  --config results/training/scratch_baseline_v1_seed1/normal_evaluation.json
"$ARTIGRASP_PYTHON" tools/ood_eval.py run \
  --config results/training/scratch_baseline_v1_seed1/normal_evaluation.json \
  --output results/normal_scratch_baseline_v1_seed1
```

smoke後に生成された評価設定も同じ仕組みですが、重みは1回しか更新していません。まず起動と引継ぎの検証を済ませ、その後に学習曲線・複数学習seed・共通の評価指標を確認して、報酬アブレーションのbaselineとして採用できるか判断してください。

## 整備時に確認した範囲

関連40件のテストに加え、8物体・CPU 1 thread・各stage 2 iterationの通し学習を確認しました。同じseedの2実行では抽出参照とcheckpoint内の全tensor（optimizerを含む）が一致し、別seedでは参照と方策重みが変わりました。CUDAでも左手stageを2 iteration実行し、保存まで確認しています。記録は `results/training/scratch_training_two_update_verification.json`、実行結果は `results/training/validation_2updates_seed1_a/` などにあります。

これは起動・PPO更新・探索分散同期・自作重み引継ぎの検証です。本学習の収束や性能、32 threadや別hardwareでのbit一致を示すものではありません。
