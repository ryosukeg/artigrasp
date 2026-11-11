**box 行動補正実験の作業記録**

**2026年11月7日 JST 実験準備**

依頼内容は、対象を box に限定し、動作データ収集から予測モデルの学習・評価までの手順と作業過程を新しいディレクトリへ保存すること。

今回作成したもの：

- `README.md`：実験の全体手順、最初に実行するコマンド、時間の目安。
- `DATA_COLLECTION.md`：補正付き応答データの収集仕様。
- `configs/baseline_smoke.json`：公式 train split、normal、8 試行、seed 101 の補正なし動作確認。
- `configs/evaluation.json`：公式 test split、normal と開閉 OOD 各 100 試行、seed 202 の比較条件。
- `.gitignore`：今後の大きなデータ・実行結果・重みを追跡対象から除外。

確認した既存仕様：

- box は最終開閉でも `cooperative_right` を使う。PPO checkpoint は変更しない。
- 全 1,560 step のうち最終 300 step が補正対象。制御周期は 0.01 秒。
- 既存 NPZ は状態・実行 action・参照を保存するが、今回必要な左右 raw 観測や明示的な補正は保存しない。
- 新しい予測モデルは、固定 PPO と補正を組み合わせた短時間応答を学習する。
- 本収集の 0.5〜1.5 rad は計画値。既存 normal 動作確認の 0.5〜0.6 rad と区別する。
- 今回の OOD 条件は factorial 設定の小角度側と大角度側を含み、従来の five 設定とは異なる。新しい対照実験を行う。

既存ソースには今回以前から未コミットの変更があるため、今回の準備はこの実験ディレクトリ内へ追加する。既存 runtime・学習 runner・過去の結果は変更しない。

2026年11月7日 17時50分 JST に設定を検証し、結果を `verification.json` へ保存した。

- 両設定で既存 CLI の `check` が `Runtime check: OK`。Python 3.8.20、PyTorch 2.4.1+cu121、compose_eval 拡張 module と checkpoint・資産を確認した。
- 両設定で既存 CLI の `plan` を一時ディレクトリへ実行。baseline は train の 8 試行、評価は test の 200 試行を生成できた。
- baseline の目標角は 0.5〜0.6 rad。評価は normal 100 試行と、公開学習目標範囲外の OOD 100 試行。全目標角が box の関節可動域内だった。
- 評価 seed 202 の OOD は小角度側 14 試行、大角度側 86 試行。normal と OOD の sample ごとの参照・seed が対応することを確認した。
- 検証用の一時計画は削除済み。README の `runs/baseline_smoke_seed101` はまだ作成していないため、記載どおり初回の計画を生成できる。
- シミュレーション・教師あり学習・比較評価はまだ実行していない。今回の確認は物理 rollout の動作保証ではなく、依存環境・入力資産・計画生成の確認である。

次の作業は README の手順 2 による補正なし 8 試行の動作確認。その後、収集仕様に従って補正付き収集器を実装し、少量でラベルの対応と補正ゼロの再現を確認する。

以後の作業では、実行日、目的、変更内容、実行コマンド、設定と seed、出力先、完了状況、得られた事実、次の作業を追記する。推測と実測結果を区別し、途中結果から成功率の改善を確定しない。

**2026年11月7日 JST コンテナで conda が見つからない原因を確認**

`conda activate artigrasp` が `conda: command not found` になるという報告を調査した。

- `.singularity/container.yaml` と実際の mount 情報で、`/home/cvrt/miniconda3/envs/artigrasp` だけが読み取り専用で共有されていることを確認した。
- `/home/cvrt/miniconda3/bin/conda`、`condabin/conda`、`etc/profile.d/conda.sh` はコンテナ内に存在しない。PATH の追加だけでは解決しない。
- 環境の `/home/cvrt/miniconda3/envs/artigrasp/bin/python` は存在し、直接実行できた。`sys.prefix` も同じ artigrasp 環境を指していた。
- 直接実行で NumPy 1.23.1、PyTorch 2.4.1+cu121、CUDA 利用可能、RTX A5500 の認識を確認した。
- 同じ Python で `tools/ood_eval.py check --config experiments/box_dynamics_guidance/configs/baseline_smoke.json` を再実行し、`Runtime check: OK` を確認した。

README に、activate を使わず環境の Python を直接指定する理由を追記した。コンテナ設定やインストール済み環境は変更せず、シミュレーションも開始していない。次の作業は引き続き、補正なし 8 試行の動作確認。

**2026年11月7日 18時01分 JST 補正なし 8 試行を確認**

ユーザーが実行した `runs/baseline_smoke_seed101` を確認した。実行開始記録は同日 17時57分37秒 JST。今回の確認ではシミュレーションを再実行せず、保存データと hash を照合した。

- 計画 8 件と完了 8 件の ID が一致。重複・未実行・数値異常は 0 件。
- 8 本の NPZ を読み戻し、状態が 1,561 × 138、左右 action が各 1,560 × 51、全配列が有限値であることを確認した。
- 各軌跡の hash が記録と一致。`verify_provenance` でソース・入力・計画の hash が現在のファイルと一致した。
- 保存状態から主要な誤差と成功判定を再計算し、全件で記録と一致した。
- CUDA の actor 入出力と parameter、物理 4 thread 設定を確認。rollout 22.04 秒、記録された準備・保存込み 24.85 秒。
- 最終状態の論文 Suc.T 条件は 2 / 8、公開実装互換 4 / 8、角度誤差 < 0.5 rad のみ 6 / 8、角度誤差 < 0.1 rad のみ 1 / 8。持ち上げは 1 / 8、持ち上げ込みの複合達成は 0 / 8。
- 最終開閉開始時に基部の位置・姿勢閾値を満たすのは 3 / 8。開閉以前の失敗が複合指標を制限するため、開始状態を記録し、角度改善と基部安定性も個別に評価する。

`BASELINE_REVIEW.md` と `baseline_smoke_review.json` に結果を保存し、README を「手順 2 完了、次は手順 3」へ更新した。収集仕様に開閉開始時の状態と前段の成否の記録を明記した。

判断は「補正なしの実行・保存の動作確認は完了」。raw 観測と補正ラベルは現行 NPZ にないため、本学習用データの準備が完了したわけではない。次は補正付き収集器を実装し、ゼロ補正の再現と小さな補正でのラベル対応を確認する。

**2026年11月7日 JST 代表 3 試行の再生と確認動画を追加**

定性的に少数の動きを確認したいという依頼に対し、`0002`（最終 3 条件達成）、`0000`（蓋とともに基部もずれる）、`0004`（持ち上げ達成・最終開閉未達）を選んだ。

- 現コンテナは DISPLAY・X11 socket がなく、Unity の GUI 録画は利用できなかった。
- 既存 `tools/ood_eval.py replay` で保存済み action を 3 試行とも再実行した。全件で `verified: true`、`trajectory_matches: true`、`classification_matches: true`、`max_state_difference: 0.0` を確認した。
- 実測状態と URDF を描画する既存 `scripts/render_ood_five.py` を再利用し、指定した試行だけを処理する `render_selected.py` を実験ディレクトリに追加した。
- 必要なパッケージを `.render_deps/` に分離して導入し、版を `requirements-render.txt` に記録した。ArtiGrasp の既存 Python 環境は変更していない。
- EGL をこの描画プロセスだけで設定し、NVIDIA RTX A5500 で描画した。GUI 用の初期化を避けるため pyglet 2.0.18 の headless 設定を使った。
- 3 本とも 1,200 × 720、20 fps、312 フレーム、15.6 秒の MP4 を作成した。全編のデコード、MP4・元軌跡の hash、終端フレームの表示、動画一覧の相対リンクを確認した。
- `qualitative/index.html` に再生ボタン、最終開閉 12.6 秒への移動、0.5 倍・0.25 倍速を追加した。全体表示と物体の拡大表示を併記している。
- 動画内の成否表示は既存 renderer の持ち上げ込み判定なので全件 FAILURE。`0002` の最終 3 条件達成とは区別して動画一覧に説明した。

動画は Unity の画面録画ではなく、評価済み物理状態の 3D 描画。手・物体の軌跡は保存値を使う。既存 renderer の机の中心高さは 0.550 m、実験の stage は 0.551 m であり、画面から 1 mm 単位の隙間や接触を判定しない。接触力や接触判定自体の可視化ではない。

出力は `runs/baseline_smoke_seed101/qualitative/index.html`、再生検証は同じ場所の `replays/`、MP4 は `runs/baseline_smoke_seed101/videos/box/normal/failure/`。元の数値評価結果や PPO は変更していない。

**2026年11月7日 UTC 補正付き収集器を実装し100試行で確認**

ユーザーの「補正付きデータ収集器を実装し、少量で確認する」という依頼を実施した。変更はこの実験ディレクトリ内のみ。既存PPO、runtime、C++環境、baseline結果を維持した。

追加した実装：

- `collect.py`：既存Controllerを継承し、step1,260〜1,559の右並進actionだけ補正。20step窓ごとに一定の補正を加え、PPOは毎step再計算する。raw観測・PPO行動・実行行動・global state・mode・異常flagを保存。
- `planning.py`：ゼロ8試行、4条件×13候補の較正52試行、窓ごとに7候補から選ぶ収集計画。公式trainのsubject/sequence単位で参照を先に分割。目標角0.5〜1.5radを収集専用分布として記録。
- `dataset.py`：計画・完了manifest・hash・schema・分割を検証。571次元入力と9次元教師に変換する。数値的に正常なタスク失敗を残し、reset/異常以降の窓を除く。
- `train_smoke.py`：128×2層MLP、trainだけで正規化、5epochの確認学習、val推論、現在状態維持の基準、checkpoint保存。
- `audit_collection.py`：ゼロの完全一致、各窓の補正、前段の同一性、補正量別の応答差を検証する。
- `planning_test.py`、`test_collect.py`、`test_dataset.py`：32テスト成功。状態切替・終点・異常ラッチ・参照漏れ・保存改ざん・再開・学習完走を確認。

時刻上の重要な実装判断：step1,260では既存Controllerが手の状態を切り替える。既存形式の `states[1260]` を変更せず、切替後の `final_states[0]` とraw観測を別保存した。末尾には追加の物理stepを進めず、最後の観測を取得して300 actionに対する301観測を保存した。

実測結果：

| 出力先 | 試行／有効窓 | 準備・実行・保存時間 | 結果 |
|---|---|---|---|
| `runs/collector_zero_seed101_v2` | 8／120 | 24.4秒 | 左右action・全stateが元の8試行と完全一致。最大差0 |
| `runs/collector_calibration_seed101_v2` | 52／780 | 154.7秒 | 各条件内の補正前履歴が完全一致。0.5と1.0actionの応答差を確認 |
| `datasets/box_response_smoke_seed301` | 40／600 | 116.5秒 | train32／480窓、val8／120窓。数値異常0、全件読込成功 |

最終版の合計100試行で非有限値・自動resetは0。開発時にも最初のゼロ8試行を `runs/collector_zero_seed101` で実行した。その後、再開時の資産確認へURDF参照mesh36件のhashを追加したため、最終版の記録はv2へ新規作成した。旧 `runs/collector_calibration_seed101` はその変更前の未実行計画であり、完成datasetではない。

ゼロv2の `run` を再実行したところ、workerを起動せず完了batchを再利用した。batch・NPZ・ログ10ファイルのhash・mtime・sizeは不変。`resume_check.json` に記録した。ソース・拡張binary・checkpoint・参照資産・mesh・計画のhashを固定し、変更後の混在を拒否する。

本収集の初期設定を、seed301、目標角0.5〜1.5rad、H20、補正ゼロと各軸±1.0の7候補、20並列・物理4thread・PPO CUDAとした。参照元はtrain14／val4グループ、把持27／6、開閉12／7。公式testは未使用。1.0actionの最初の0.2秒における蓋角度差は中央値0.000229rad、最大0.0477rad、手首位置差の中央値0.985mm。効果が小さい状態も多く、この設定が最適・OODに有効であるとは判断しない。

並列数の注意：52並列内のゼロ補正は元の8並列baselineと軌跡が異なる。初期stateは一致するが最初のactionに約10^-6の差があり、その後の軌跡が分岐する。較正は52並列内で前段の履歴が完全一致する対応比較なので、8並列baselineの終端スコアを対照に使わない。`batch_size_comparison.json` に記録し、今後の比較では並列数・batch構成・device・seedを揃えることをREADMEに追記した。

実データの5epoch確認学習は約4.1秒で完走。train480窓だけで正規化し、val120窓を評価した。val角度RMSEは0.4494rad、現在状態維持の基準は0.1497rad。学習処理は通ったが、精度はまだ基準を上回っていない。このcheckpointを候補選択には採用しない。本学習後に精度と候補順位を検証する。checkpoint・レポートは `checkpoints/smoke_seed301` に保存した。

実行コマンドはREADMEの手順3へ保存した。結果は `COLLECTOR_REVIEW.md` と `collector_smoke_review.json` にまとめた。次は手順4の1,000試行本収集。今回の40試行の速度から同じ20並列では約49分の単純見積もりになるため、収集だけで50〜60分程度を初期予算とする。本収集、本学習、推論時の候補選択、OOD比較評価はまだ実施していない。

**2026年11月7日 UTC 本収集・本学習の進捗表示と起動手順を追加**

ユーザーが自身で本学習を開始できるよう、進捗バー付きの起動コマンドを準備した。本番の1,000試行収集・本学習は開始していない。確認には保存済み40試行だけを使用した。

- `run_training.py`：収集計画 → 収集 → データ監査 → 教師あり学習を連続実行。既定は1,000試行、20並列、seed301、CUDA、最大100 epoch。前段の失敗時は後段へ進まない。`--dry-run` はファイル作成・処理起動を行わず、コマンドを表示する。
- `progress.py`：追加パッケージ不要の進捗表示。収集は全試行の制御step数と保存済み試行数、学習はepoch、train/val loss、最良epoch、角度RMSE、経過時間・ETAを表示。通常の端末では行を更新し、ログへの出力では行単位で記録する。
- `train_model.py`：571入力 → 256×3層 → 9出力、batch256、Adam 0.001。目標表現は確認学習と同じ絶対状態。trainだけで正規化し、val正規化MSEで最良モデルを選ぶ。最大100 epoch、改善幅0.0001を超える改善が20 epochなければ早期終了。
- 学習結果は `best.pt`、`last.pt`、`history.jsonl`、`status.json`、`report.json`。最良重みと再開用重みを分け、`last.pt` にoptimizer・乱数・shuffle・正規化・履歴も保存する。途中で最良モデルやレポートの保存が中断しても、最後に完了したepochから復旧できる。
- 再開時はデータ・設定・ソースの一致を検証する。一括コマンドでは同じdatasetへの二重起動を排他し、Ctrl+C時は収集workerの子孫まで停止する。収集計画や既存collectorのコードは変更していない。

検証結果：

- `test_training_workflow.py` 11件：進捗・ETA、計画不一致の拒否、未完了NPZの扱い、監査失敗時の学習抑止、二重起動、中断時のprocess group停止等を確認。
- `test_train_model.py` 3件：epoch途中の中断・再開と連続実行の重み／optimizer／乱数等の一致、厳密な最良epoch選択と早期終了の分離、最終epoch保存中の中断から追加学習なしで復旧することを確認。
- 既存40試行でGPU 3 epochを実行し、端末のバー・loss・ETA表示、checkpoint保存、同じepoch上限での再開を確認。出力は `checkpoints/trainer_progress_check_cuda`。保存した `best.pt` を `weights_only=True` で読み戻し、CPU再評価がレポートと許容誤差内で一致した。
- 一括コマンドを既存40試行＋2 epochで確認。完了した2 batchの再利用 → 600窓の監査 → GPU学習まで完走。新しい物理試行は0件。出力は `checkpoints/workflow_progress_check_cuda`。収集manifestのhashが元の確認学習時と同一であることを確認した。
- 短期実行の読込・学習は約2.8〜2.9秒。これは40試行／2〜3 epochの動作確認時間であり、1,000試行／最大100 epochの所要時間ではない。
- 本番の `datasets/box_response_1000_seed301` と `checkpoints/box_response_1000_seed301_full` がまだ存在しないことを確認した。

検証の詳細とソースhashは [training_setup_review.json](training_setup_review.json) に保存した。READMEの手順4にユーザー起動用の1コマンド、手順5に保存ファイルとlossの読み方、末尾にバックグラウンド実行方法を追記した。学習・検証の目標角は0.5〜1.5 rad。OOD目標は引き続き学習・設定調整に使わない。

本学習後は最良モデルのval誤差を状態維持予測と比較する。学習loss低下や処理の完了だけでは、候補の順位精度やOOD成功率の向上を判断しない。今回の短期チェックポイントは進捗・保存・再開の確認用であり、候補選択に採用するモデルではない。

**2026年11月11日 JST 学習済みモデルによる初回OOD比較を開始**

ユーザーの「一旦学習した状態遷移モデルを使ったOODの検証を行って、ArtiGraspとの比較を行いたい」という依頼により、候補選択と比較実行を実装した。PPO・物理環境・既存収集器・学習済み重みは変更しない。

事前に保存済み本学習結果を確認した。`box_response_1000_seed301_full` は2026年11月7日20時19分 JSTに完了しており、旧READMEの「本収集未開始」は更新されていなかった。train800／val200試行、有効窓12,000／2,985、59epoch早期終了、best39。val角度RMSEは0.140528rad、状態維持は0.115336radで基準を上回らない。今回の依頼では再学習せず、この `best.pt` でタスク性能を測定する。

追加した実装：

- `guidance.py`：学習時と同じ571入力で7候補の0.2秒後状態を予測。元の目標への角度・基部位置・SO(3)姿勢誤差で採点し、右手首の補正を選ぶ。既存collectorのscheduleを選択直後に更新し、PPO出力そのものは変えない。
- `evaluate_guidance.py`：同じ計画をbaseline／guidedで実行。既存collectorをそのまま再利用し、元PPO行動・実行行動・補正・状態・候補予測とcostを保存。計画、入力資産、sim binary、モデル、実行sourceをhash固定。完了batchの再利用と同じ出力への二重実行防止に対応。
- `evaluate_report.py`：保存軌跡の対応比較と日本語レポート。補正前状態・左右行動・phase reset後状態・目標の一致を確認し、失敗時には補正効果の結論を保留する。
- `test_guidance.py`：7テスト成功。特徴量と学習loaderの完全一致、quaternionの符号／長さ不変性、無効候補、phase後入力、20step保持、ゼロ選択時の既存collectorとの全配列一致を確認。

設定はtest評価前に固定した。7候補はゼロ＋各軸±1.0 action単位、step1260〜1559で20step保持・20stepごと再選択。当初案の短い再選択間隔より、初回は学習時の保持時間に一致させた。costは `(角度誤差/0.5)^2 + (位置誤差/0.05)^2 + (SO3姿勢誤差/0.2)^2 + 0.01||補正||^2`。閾値を尺度とする事前固定で、最終testによるパラメータ探索は行わない。無効候補を除外し、全候補無効ならゼロへ戻す。

公式train参照4件の技術確認を `runs/guidance_smoke_seed909` で実行。seed909、4並列、4thread、normalのみ、2方式の計8試行。4/4対で補正前履歴・開始状態・目標が完全一致し、数値異常0。baseline15.0秒、guided14.3秒（準備・保存込み）。これは性能判断用の評価ではない。

本番は `runs/comparison_seed202_v1`。既存 `configs/evaluation.json` に従い、公式test・seed202、normal100／開閉角OOD100、各2方式で計400試行。100並列、物理24thread、PPOと予測器はCUDA。両方式のbatch構成・順序を揃える。目標角はnormal0.5〜0.6rad、OODは[0.2,0.4]∪[1.6,3.1]rad。旧OOD評価の成功率を対照群として流用しない。

実行コマンド：

```bash
cd /home/cvrt/goto/artigrasp
/home/cvrt/miniconda3/envs/artigrasp/bin/python experiments/box_dynamics_guidance/evaluate_guidance.py plan --output experiments/box_dynamics_guidance/runs/comparison_seed202_v1
/home/cvrt/miniconda3/envs/artigrasp/bin/python -u experiments/box_dynamics_guidance/evaluate_guidance.py run --output experiments/box_dynamics_guidance/runs/comparison_seed202_v1
```

2026年11月11日10時28分 JST頃に本番を開始した。既存のユーザー変更を維持し、今回の追加・更新は実験ディレクトリ内のみ。最終結果と独立監査は完了後に以下へ追記する。

**2026年11月11日10時34分 JST 本比較400試行・独立監査完了**

全400試行の物理実行・保存と、200対の対応条件照合を完了した。数値異常は0件。結果は `runs/comparison_seed202_v1/COMPARISON_REVIEW.md`、機械可読集計は同ディレクトリの `summary.json`、試行別数値は `results.json` に保存した。

| 条件・指標 | PPO単体 | PPO＋予測補正 |
|---|---:|---:|
| 通常：最終位置・姿勢・角度の同時達成 | 9/100 | 11/100 |
| 通常：平均角度絶対誤差 | 0.623419 rad | 0.629018 rad |
| OOD：最終位置・姿勢・角度の同時達成 | 19/100 | 19/100 |
| OOD：角度誤差0.5 rad以内 | 43/100 | 46/100 |
| OOD：角度誤差0.1 rad以内 | 11/100 | 7/100 |
| OOD：平均角度絶対誤差 | 1.264346 rad | 1.227065 rad |
| OOD：平均基部位置誤差 | 0.101995 m | 0.095418 m |
| OOD：平均基部姿勢誤差 | 0.659062 rad | 0.626998 rad |
| 持上げ込み成功（通常／OODそれぞれ） | 0/100 | 0/100 |

最終3条件の閾値は位置0.05m、姿勢0.2rad、開閉角0.5rad未満。OODの失敗→成功は5対、成功→失敗も5対で、成功率は変わらなかった。角度誤差の対応差（補正−PPO）は平均−0.037281rad、参考の試行単位bootstrap95%区間は[−0.108920, 0.032723]rad。単一seed、同じ参照の再利用を含むため、一般的な改善や統計的確証は主張しない。

小角OOD14件は最終3条件3→3、大角86件は16→16。小角側は角度0.5rad基準が全14件で達成される一方、0.1rad基準は1件のみである。大角側の0.1rad基準は10→6。通常・OODとも前段の持上げだけは13/100で達成しているが、最終3条件との同時達成が0件だった。補正は最終開閉にのみ入るので、前段の持上げ結果を変更しない。

技術検証：

- `test_guidance.py` 7件、`test_evaluate_report.py` 11件の計18テスト成功。後者は欠損・重複・hash改変・補正前状態不一致・無効試行の分母・smokeとtestの区別等を確認した。
- 200/200対で、`states[:1261]`、左右の`actions[:1260]`、phase reset後の`final_states[0]`、元目標が完全一致。成果物hashも照合した。
- 別処理による全400試行の独立監査で、624,000 control step、6,000窓の補正・20step保持・候補選択、17,600指標の再計算を確認。guidedの窓0/7/14を100並列の形状のまま計600窓再推論し、保存予測・costとの差はともに0.0、選択候補も全件一致。
- 全候補無効によるゼロfallbackは0件。補正をゼロと選んだ窓は通常8/1,500、OOD4/1,500であり、補正が実際に作用している。
- 独立監査は `independent_audit.json`、監査時のスクリプトは `independent_audit_source.py` に保存した。

実行時間は100試行batchごとに通常PPO75.20秒、通常guided79.35秒、OOD PPO74.91秒、OOD guided78.01秒で、合計307.46秒（準備・保存込み、process起動と最終監査は除く）。guidedの候補選択は計0.2588秒、100環境×7候補の選択1回平均8.63ms。これは転送・特徴量構築・採点込みのbatch時間で、1環境の遅延ではない。方式間の全体時間差は負荷・物理軌跡の差も含み、純粋な推論追加コストとは扱わない。

結論は「今回の学習済みモデル・事前固定した補正選択・このseedでは、OOD成功率の改善は確認できなかった」。平均角度誤差の小さい低下と、厳しい角度閾値の悪化を両方残す。test結果を見た再調整・再学習は行っていない。今後の改善検討ではtrain/valに戻り、同じ状態で補正候補の効果を区別できるかを確認する。

**2026年11月11日 JST 学習時間・サンプル数の確認とlossグラフ作成**

ユーザーの「66秒で学習が完了するのは正常か、モーション／フレーム／遷移数はいくつか、lossの推移を見たい」という依頼を確認した。今回の作業では再学習・シミュレーションを行わず、保存済みreport/history/checkpointを読み取った。

- 収集は1,000本×1,560 control step＝1,560,000 step。初期状態を含む保存stateは各1,561行。制御周期は0.01秒、各動作15.6秒。
- 学習で使用するのは各動作の最終300 step（3秒）。20 step（0.2秒）ごとに開始と終点の組を作るため、1試行15窓。train800試行＝12,000遷移、val200試行から無効な1試行15窓を除いて2,985遷移。
- 入力は各窓の開始時点の571成分、教師は終点の9成分。中間フレームを逐次入力するモデルではない。12,000件を59 epoch繰り返し、延べ708,000件を訓練に提示した。
- MLPは571→256→256→256→9、280,329パラメータ。batch256は47更新/epoch。`last.pt`をweights_onlyで読み、8個のparameter tensorすべてのAdam stepが2,773であることを確認した。
- `report.json`の全処理時間は65.6389秒。各epochの訓練＋検証時間合計は30.2213秒、平均0.5122秒。残り35.4176秒は読込・検証・初期化・保存・最終評価等の合計差分であり、個別の内訳は未計測。PPO更新やsim実行は含まない。
- 最良は39 epochのval正規化MSE 0.161677、59 epochでpatience20による早期終了。状態維持のval MSE 0.107181はどのepochのモデルより小さい。短時間で学習ループが終わることと、予測精度が十分であることは別に評価する。

`plot_training_history.py`を追加し、元historyのhash・epoch連続性・最良epochをレポートと照合して作図した。出力は `checkpoints/box_response_1000_seed301_full/plots/` の `loss_curve.png`、`.pdf`、`.svg`、`loss_history.csv`、`plot_metadata.json`。図は全体lossと検証lossの拡大表示で、平滑化なし。訓練曲線は更新中のミニバッチloss平均、検証曲線は各epoch終了後の固定重みによる値であることを注記した。

Matplotlib 3.7.5と依存ライブラリは `/tmp/box_plot_deps_20261111` に隔離した。既存ArtiGrasp環境のNumPyは1.23.1のままであることを確認。図の表示を目視し、凡例と軸ラベルの重なりを調整した。READMEに図へのリンクと再生成コマンドを追記した。

**2026年11月11日 JST 既存データから差分教師を作成・同条件で本学習**

ユーザーの「既存データから差分の教師データを作る／同じ参照分割・モデル規模で新たに学習する」という依頼により、追加収集なしで実行した。出力は `checkpoints/box_response_1000_seed301_delta_angle_position/`。詳細は同出力の [DELTA_REVIEW.md](checkpoints/box_response_1000_seed301_delta_angle_position/DELTA_REVIEW.md) に保存した。

- `delta_targets.py` を追加。角度index0・基部位置index2:5のみ `未来 − 現在` に変換。角速度・姿勢は絶対値を維持。train/valの教師・元絶対教師・現在状態・試行ID・窓stepをpickle不要のNPZへ保存し、元データ・archive・sourceのhashをmanifestへ記録。
- `train_delta.py` を追加。元のResponseMLPと保存・乱数管理helperを再利用し、学習loop・Adam・shuffle・早期停止条件を維持。差分4成分は中心0、全出力の尺度は元の絶対train教師の標準偏差。元データや外れ値、参照分割、元モデルを変更せず、新規初期化から学習した。
- `compare_delta.py` を追加。保存された両方のbestを別プロセスで再ロードし、同一の検証窓に対する絶対状態の予測誤差を再計算する。teacherの対応、train-only正規化、設定・モデル規模・optimizer・source・成果物hashと実際のAdam更新数を照合。
- `test_delta_targets.py` 10件と `test_train_delta.py` 9件、計19テスト成功。差分変換・復元・物理lossの等価性、source不変、元モデルとの正規化一致、best再読込、中断再開の重み/optimizer/RNG/history一致、早期停止後の復旧、archive改変拒否を確認。専用checkpoint型を既存absolute用Selectorが拒否することも確認。

既存800train試行・12,000遷移、200val試行のうち無効15窓を除く2,985遷移を維持した。目標角は0.5〜1.5 rad、OOD目標を学習へ混ぜていない。571→256→256→256→9の280,329パラメータ、batch256、Adam lr0.001、seed301、最大100epoch、patience20、min_delta0.0001、RTX A5500という条件で実行。

学習は78epochで早期終了、best58。実更新は3,666回（47回/epoch）、延べ936,000訓練サンプル。全処理76.7608秒、各epochの訓練＋検証合計39.3247秒。残りには読込・検証・教師保存・初期化・checkpoint保存・最終評価等を含む。学習設定は同じだが、元モデルの59epoch・2,773更新とは早期停止の時点が異なる。

| 検証指標・小さいほど良い | 状態維持 | 元の絶対モデル | 今回の差分モデル |
|---|---:|---:|---:|
| 蓋角度RMSE [rad] | 0.115336 | 0.140528 | 0.093844 |
| 蓋角速度RMSE [rad/s] | 0.718340 | 0.695051 | 0.685823 |
| 基部位置成分RMSE [m] | 0.008517 | 0.103232 | 0.016503 |
| quaternion成分RMSE | 0.031970 | 0.136743 | 0.129292 |
| 全9出力の正規化MSE | 0.107181 | 0.161677 | 0.130830 |

元モデル比で角度33.2%・位置84.0%・全体MSE19.1%改善。状態維持に対して角度・角速度は改善するが、位置・姿勢・全体MSEは劣る。全体MSEは状態維持より22.1%大きい。差分化の改善と残る予測誤差の両方を記録し、OOD成功率の向上とは解釈しない。

再評価で全指標の保存report/checkpointとの差0.0を確認し、`absolute_vs_delta.json` と窓ごとの予測NPZへ保存した。元学習器・dataset・train_smoke・progressのsource hash、元モデルの重み・履歴hashは変更なし。入力正規化・全出力尺度は完全一致。lossグラフをPNG/PDF/SVG/CSVで保存し、表示を目視確認した。描画依存だけ一時PYTHONPATHを使用し、学習・再評価は元のNumPy 1.23.1環境で行った。

差分checkpointは `box_response_delta_best_predictor`。現在状態を加える復元が必要なため、既存のabsolute用guidanceは誤ロードを拒否する。今回の完了範囲は教師生成・新規学習・同条件の予測比較であり、このモデルを使う候補選択の接続・新しいOODロールアウトは未実施である。

**2026年11月11日 JST PPO・元モデル・差分モデルのnormal／OOD比較600試行を完了**

ユーザーの依頼により、補正強度・斜め方向の拡張は後回しとし、現在の7候補で3方式を比較した。出力は [THREEWAY_REVIEW.md](runs/comparison_seed202_threeway_v1/THREEWAY_REVIEW.md)、[summary.json](runs/comparison_seed202_threeway_v1/summary.json)、[数値CSV](runs/comparison_seed202_threeway_v1/metricstable.csv)。旧結果を流用せず、PPO単体と元モデルも含む全600試行を新規実行した。

実装・条件：

- `guidance_delta.py` に専用checkpointの検証と、角度・基部位置へ現在値を加える復元処理を追加。角度の基準は訓練と同じ右観測index273、位置は物体状態index122:125。復元後は元モデルと共通の採点処理へ渡す。
- `evaluate_threeway.py` に3方式の計画・実行・hash確認付き再開を追加。`report_threeway.py` で予定600枠を分母に維持した集計、`audit_threeway.py` で独立の軌跡・指標・候補選択・再推論監査を実装した。既存PPO・絶対モデル・差分モデル・収集データ・元の2方式評価ソースは変更していない。
- 新規のdelta adapter 7件、3方式計画／再開4件、集計9件の計20テスト成功。既存guidanceの7件も成功。公式train参照の4条件×3方式＝12試行の実smokeを経て本評価へ進んだ。
- 本評価は公式test、seed202、normal100条件・OOD100条件を各3方式。全1,560stepの複合タスクを実行し、最終開閉300stepだけ右手首並進へ補正。ゼロと各軸±1の7候補、20step（0.2秒）保持・再選択、元の採点尺度を維持した。
- normal目標角は0.5〜0.6rad、OODは[0.2,0.4]∪[1.6,3.1]rad（小角14／大角86件）。予測モデルの訓練目標角0.5〜1.5radは変更せず、追加収集・再学習・採点重みの調整は行っていない。
- 各方式の試行順・100並列・CUDA・物理24threadを統一。旧 `comparison_seed202_v1` の200件のepisode仕様、config全体、guidance、seedも完全一致した。

| 条件 | 方式 | 最終3条件成功 | PE平均 [m] | AAE平均 [rad] | AE平均 [rad] |
|---|---|---:|---:|---:|---:|
| normal | PPO単体 | 9/100 | 0.124144 | 0.623419 | 0.698447 |
| normal | PPO＋元モデル補正 | 11/100 | 0.119713 | 0.629018 | 0.648940 |
| normal | PPO＋差分モデル補正 | 9/100 | 0.120787 | 0.598671 | 0.722077 |
| OOD | PPO単体 | 19/100 | 0.101995 | 1.264346 | 0.659062 |
| OOD | PPO＋元モデル補正 | 19/100 | 0.095418 | 1.227065 | 0.626998 |
| OOD | PPO＋差分モデル補正 | 15/100 | 0.095528 | 1.300817 | 0.643278 |

成功は最終PE<0.05m、AAE<0.5rad、AE<0.2radの同時達成。PEは基部位置距離、AAEは蓋角度の絶対誤差、AEは基部姿勢のSO(3)回転角距離。表は失敗例を含む全100試行の未クリップ最終誤差の平均であり、予測モデルのRMSEではない。途中の持上げも要求する複合タスク成功 `ood_goal_success` はnormal／OODの全3方式で0/100。前段の持上げ達成は全方式でそれぞれ13/100で、補正はこの前段を変更しない。

normalでは差分化でAAE平均が0.623419→0.598671radとなったが、最終成功は9→9件。OODでは19→15件、AAE平均1.264346→1.300817radとなった。OODのPPO失敗→差分成功は4件、PPO成功→差分失敗は8件。元モデルとの比較でも差分は成功1件増／5件減で、19→15件。小角側は3/14で全方式共通、大角側はPPO16/86・元モデル16/86・差分12/86。角度のみ0.1rad以内はOODで11／7／9件だった。

差分のOOD AAE対応差（差分−PPO）の平均は+0.036471rad、試行単位の参考bootstrap95%区間は[-0.037776, 0.119727]rad。単一seedで参照の共有もあり、一般的・統計的な優劣は断定しない。今回は以前に評価したseed202の目標を再利用した探索的比較であり、独立した新規目標群による最終確認とは扱わない。normalとOODの目標分布および閾値への近さも異なるため、条件間の成功率の大小だけで汎化を判断しない。

この設定で差分モデルによるOOD成功率の改善は確認できなかった。学習内の参照検証で遷移予測誤差が改善しても、OOD状態で有効な候補を選べることを保証しない。今回の比較はその原因を特定する実験ではない。

独立監査：

- 全600試行、936,000 control step、9,000窓を照合。左右の実行action、最終開閉以外のゼロ補正、20step保持、7候補のargmin・fallback、PE/AAE/AE・lift込み指標を再計算した。
- PPOと各補正方式の400/400対で、補正前状態・行動・観測、手のreset後の開始状態、目標が完全一致した。数値異常は0件。
- 両補正方式の窓0/7/14を、元の100並列形状・順序・CUDAで独立に再推論。計1,200窓で予測値・採点値の最大差0.0、選択候補も全一致。資産・source・checkpoint・保存NPZのhashも照合した。
- ゼロfallbackは全方式0件。元モデルのゼロ選択はnormal8/1,500、OOD4/1,500、差分はnormal16/1,500、OOD7/1,500で、補正は実際に実行されている。
- 以前の2方式評価との再現確認では、PPO200試行・元モデル200試行の全保存配列が完全一致した。
- 監査結果は [independent_audit.json](runs/comparison_seed202_threeway_v1/independent_audit.json)。監査53.71秒。6workerの準備・実行・保存時間合計453.22秒（約7分33秒）、process起動・最終集計・監査は別。方式間の時間差には物理軌跡や機械負荷の差も含む。

README手順6bに計画・実行／再開・再集計・監査コマンドを追加し、差分学習レポートにも今回の閉ループ評価への追記リンクを加えた。

**2026年11月11日 JST 3方式・normal／OODの定性比較HTMLを作成**

ユーザーの依頼により、`runs/comparison_seed202_threeway_v1/qualitative/index.html` を作成した。定量評価で保存した18軌跡を6本の同期比較MP4として描画し、各ケースでPPO／元モデル補正／差分モデル補正を横並びにした。追加の学習・物理シミュレーション・方策推論は行っていない。

- 新規 `render_threeway.py` は既存 `scripts/render_ood_five.py` のURDF描画部品を再利用。1本15.6秒・20fps・312frame・1920×880、6本計1,872frame、合計7.77MB。描画487.4秒（GPU A5500）。拡大は同じ方向・倍率で物体追従、全体小窓は各ケース3方式共通の固定カメラ。
- 保存軌跡のstate1260はreset前のため、表示時は `final_states[0]` のreset後状態を使用。1261以降と終点は元の保存状態と一致。動画の最後のPTS15.55秒には真の終点15.60秒を表示し、各フレームのstate indexをmanifestへ保存。HTMLはこの対応で時刻と角度カーソルを表示する。
- 新規 `threeway_gallery.py` は外部サービスを使わないローカルHTMLを生成。normal／OODフィルタ、ケース選択、同期動画、開閉開始12.6秒への移動、開閉ループ、0.5／0.25倍、終点静止画、時刻連動の3方式角度曲線とグラフclick移動、ケースごとの終端指標、全600試行の集計を表示する。表のPE[m]、AAE/AE[rad]は元の評価単位を維持した。
- 選定はnormalのdelta成功獲得2件・喪失2件と、大角OODの獲得3件・喪失7件から、それぞれAAE改善量／悪化量の下側中央値。小角14件と大角共通失敗65件は三方式平均の選定用値 `PE/.05 + AAE/.5 + AE/.2` の下側中央値。同値はID順。
- 選定IDはnormal0017／0086、OOD0019（小角）／0041／0037／0053。改善2・悪化2・共通失敗2の説明用層別例であり、出現頻度を再現した標本ではない。各ケースに選定理由と注意点を記載した。特に小角の開始時角度条件達成、大角0037の0.5rad閾値との差0.000646radを明示。
- 全18軌跡のhash・共通prefix・post-reset state・目標を照合。終端stateからPE/AAE/AEと成功を独立再計算し保存値と一致、HTMLの角度系列も元stateと完全一致。全6MP4の長さ・解像度・312frameをffprobeで確認し、ffmpegで全フレームdecode成功。source・動画・画像・manifestをhashで記録した。
- 実Chromiumで6ケースの動画metadataと再生進行、12.6秒seek、0.5／0.25倍、終点静止画、開閉loop、graph click、最終frame物理15.60秒表示、normal2／OOD4フィルタ、全600件の表を確認。console/page/assetエラー0。幅1440と390の画面を目視確認し、モバイル横はみ出しなし。検証用Chromium・不足ライブラリ・日本語フォントは/tmpに隔離し、既存環境を変更していない。

成果物： [比較HTML](runs/comparison_seed202_threeway_v1/qualitative/index.html)、[manifest](runs/comparison_seed202_threeway_v1/qualitative/manifest.json)、[メディア検証](runs/comparison_seed202_threeway_v1/qualitative/media_verification.json)、[ブラウザ検証](runs/comparison_seed202_threeway_v1/qualitative/browser_verification.json)。READMEへ入口と再生成コマンドを追記した。600試行の定量結果やモデル・収集データは変更していない。

**2026年11月11日 適応終了判定の実装と少量検証**

ユーザーの「固定15.6秒や単なる延長ではなく、動きの停滞などで終了したい」という依頼に対応し、新規 `adaptive_stop.py`、`collect_adaptive.py`、`evaluate_adaptive.py`、`report_adaptive.py`、`audit_adaptive.py` を追加した。旧収集器・旧評価・方策・予測モデル・シミュレータは変更していない。

終了規則は開閉最低3秒、成功3閾値（PE<.05m、AAE<.5rad、AE<.2rad）を0.5秒連続維持、または直近2秒の蓋角度レンジ<.02rad・基部位置の窓開始からの最大変位<.002m・基部姿勢の窓開始からの最大回転<.02rad、最後に開閉10秒の上限。瞬間的な成功条件成立中は停滞を抑制し、保持確認を待つ。invalidを最優先し、最低時間を待たず失敗扱いとする。成功保持51状態と停滞201状態は両端を含む。

この停滞は物体の変化だけを使う。手の接触探索・非常に遅い進行が続いていても止める可能性はあるため、到達不能とは解釈しない。過去のval200軌跡に固定規則を適用した事前診断では、1無効を除き3秒時点で成功維持42・停滞32・継続125。モデル選択に使用したvalであり、未使用の独立性能評価ではない。代表例0037/0041は3方式とも継続、0053は3方式とも停滞だった。閾値をtest成功率で調整していない。

各試行の評価と保存範囲を最初の終了判定でラッチする。100並列の形状・試行順を変えないため、終了済み環境も最後の環境が終わるまで内部計算に残すが、その後の状態や異常は評価へ含めない。新しい収集器のテストで、終了済みslotが後でNaNになっても保存状態・評価履歴が保護されることを確認した。最終開閉の補正は従来の20step・7候補、最終窓だけ途中終了し得る。

- 終了判定17件＋並列収集1件の18 unittest PASS。
- smoke v1は12シミュレーション正常完了後、集計モジュールの並行実装が未配置だったため集計importのみ失敗した。別途生成済みの結果は保全した。その後、ちょうど3秒時点で観測が初めて非有限になった場合の固定時点比較に、その観測のinvalidを含める境界修正を行った。変更前ソース3ファイルはv1の `source_snapshot/` へ保存した。
- 修正版を `runs/adaptive_smoke_seed909_v2` で12試行再実行。独立監査は全12の元3秒prefix（状態、PPO、実行action、補正、観測、予測・score）のdtype/shape/bytes完全一致を確認。終了理由は停滞11・上限1、独立のrolling再判定・終端指標・固定3秒指標も一致。途中終了窓9件を確認した。
- 監査実装は別計算の8境界ケースと、prefix/終端PE/窓端点/action/reason/predictionの6改変検出を確認。smoke v2の `independent_audit.json` に結果を保存した。

本比較は `runs/comparison_seed202_adaptive_v1` に旧 `comparison_seed202_threeway_v1` のplanを複製し、normal100＋OOD100条件×3方式を実行する。終了時点の3条件成功、0.5秒維持成功、持上げ込み複合成功を分け、元の3秒時点も併記する。

**2026年11月11日 適応終了による本比較600試行を完了**

同じtest参照・seed202・200目標・100並列・物理24thread・CUDAを維持し、3方式を新規実行した。6workerの準備・実行・保存時間の合計は 672.68秒（約11分13秒）。追加学習は行っていない。

| 条件 | 方式 | 固定3秒成功 | 適応終端1時点成功 | 0.5秒維持成功 | 開閉時間平均 [s] | PE平均 [m] | AAE平均 [rad] | AE平均 [rad] |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| normal | PPO | 9/100 | 9/100 | 9/100 | 5.1155 | 0.159457 | 0.572852 | 0.939271 |
| normal | 元モデル補正 | 11/100 | 10/100 | 10/100 | 5.2273 | 0.153106 | 0.616246 | 0.891376 |
| normal | 差分モデル補正 | 9/100 | 9/100 | 9/100 | 4.9488 | 0.154448 | 0.610260 | 0.852048 |
| OOD | PPO | 19/100 | 20/100 | 20/100 | 5.0588 | 0.121013 | 1.419686 | 0.700709 |
| OOD | 元モデル補正 | 19/100 | 22/100 | 21/100 | 4.9658 | 0.101122 | 1.381819 | 0.624908 |
| OOD | 差分モデル補正 | 15/100 | 18/100 | 18/100 | 5.0473 | 0.110522 | 1.454386 | 0.652486 |

最終開閉の終了判定は途中のlift成功を含めず、持上げ込み複合成功は別に集計した。今回も全条件・全方式で0/100。延長により全指標が改善するわけではなく、目標角へ近づいても基部がずれる・回転する例がある。適応終端は方式ごとに時刻が異なるため、同じ時刻での誤差比較とは解釈しない。

問題になっていたOOD0037は、PPOが開閉3.20秒（全体15.80秒）、元モデルが3.45秒（16.05秒）、差分が3.51秒（16.11秒）で成功維持により終了。差分のAAEは固定3秒で0.500646rad→終端0.106596radとなり、この例では従来の時間制限が成否へ影響していた。0053は全3方式とも3秒時点で直近2秒の停滞条件により終了しており、10秒まで実行して到達不能を確かめたわけではない。

元モデルnormal0040は固定3秒では成功だが、成功域は2.90〜3.20秒の0.30秒だけで、維持0.5秒を満たさず5.31秒で停滞終了した。逆に元モデルOOD0073は9.58秒で初めて成功域に入り10秒時点では成功だが、維持0.42秒のため時間上限終了。これによりOOD元モデルの「終端1時点22件」と「成功維持21件」が異なる。

成果物: [比較HTML](runs/comparison_seed202_adaptive_v1/index.html)、[詳細レポート](runs/comparison_seed202_adaptive_v1/ADAPTIVE_REVIEW.md)、[CSV](runs/comparison_seed202_adaptive_v1/metricstable.csv)。HTMLは全200条件の保存角度系列・各方式の終了点・終了理由と、全600試行の数値を表示する。動画の再描画はしておらず、旧固定時間動画へリンクしている。

本評価の独立監査は全600試行でPASS。保存された1,059,635 control step、15,313補正窓（うち途中終了265窓）を照合した。旧固定時間版との先行3秒prefixは600/600でdtype・shape・bytesが完全一致し、補正前の方式間prefixも400対で一致。終了理由は独立のrolling再計算と一致し、成功維持87・観測上の停滞432・上限81・無効0。終端の物理指標、固定3秒の元row全キー、候補score/argmin、モデル・source・資産hashも確認した。監査79.05秒。レポートを別プロセスで再生成し、plan/resultsの両hashが一致する監査completed状態をHTML・summaryへ反映した。

OODでは、固定3秒未達から成功維持に至ったのはPPO1件・元モデル2件・差分3件。ただし全100試行の平均AAEとPEは全方式で増加した。閾値付近の少数例の救済と、分布全体の誤差改善は区別する。新しい評価でも差分の維持成功18/100はPPO20/100・元モデル21/100を上回らず、モデルの汎化改善が実証されたとは解釈しない。単一seed・既に観察した目標集合の探索的比較であり、方式の一般的な優劣も断定しない。

最終HTMLを実Chromiumで検証し、200条件すべての各方式の終了点・表示値・終了理由・時刻を埋込JSONと照合した。normal/OOD各100件の絞り込み、前後移動、終了後の未測定表示、旧動画等へのリンク、desktop1440/mobile390の表示を確認し、console/page/requestエラー0・横はみ出し0。監査は「完了（現在のplan/results hash一致）」と表示され、最終plan/results/summary/audit/HTMLのhash一致も確認済み。記録は [browser_verification.json](runs/comparison_seed202_adaptive_v1/browser_verification.json)。新しいレポートHTMLの最終SHA256は `d02268f88bea5a79a0cb12b3420b310c8f127861e150e829766760ca96dd6acc`。READMEと本ログに条件・結果・実行手順を保存した。

**2026年11月11日 適応終了の定量HTMLに比較動画を追加**

ユーザーの依頼により、既存 `runs/comparison_seed202_adaptive_v1/index.html` へ比較動画を組み込んだ。新しいシミュレーション・PPO推論・学習は行わず、評価時に保存された18軌跡を描画した。

- 新規 `render_adaptive.py` は旧 `render_threeway.py` の固定カメラ計算と `scripts/render_ood_five.py` のURDF描画部品を再利用。旧ソース・旧固定時間動画は変更しない。normal0017／0086、OOD0019／0041／0037／0053の以前と同じ6例を選び、新しい成否による再選抜はしていない。
- 3方式を1920×880の同一動画へ横並びにし、20fps・全6本2,355frame、合計9,278,076byte（約9.28MB）を保存。描画338.17秒。共通の状態が完全一致する開閉前区間では同じ描画画像を再利用し、物理や評価の状態は変更しない。
- 動画の共通時間軸はケース内で最も遅い終了時刻まで。各列の手・物体・追従カメラ・数値・角度曲線を、その方式の真の終端で止める。静止列には `STOPPED`・終了理由・その列の実時刻を表示する。内部の並列シミュレーションを物理的に止めたという意味ではなく、終了後の測定や動作を示さない保存軌跡の再生であることをHTMLに明記した。
- state1260は `final_states[0]` の手リセット後へ置換。共通frame indexと方式別のclamp済みindexをmanifestへ保存した。20fpsの刻みに合わない各終了stepも必ず真の終端を含める。最長物理時刻19.69／22.09／22.32／21.82／16.11／15.60秒に対し、MP4尺は19.70／22.10／22.35／21.85／16.15／15.60秒。HTMLはMP4の時刻から保存frame対応で物理時刻へ変換する。
- `report_adaptive.py` にoptional動画manifestの読込・source/asset hashと停止時刻の照合を追加。同じ200ケース選択を動画と共有し、「動画ありのみ」6例を初期表示する。チェックを外すと全200条件を選択できる。非動画ケースでは旧動画を解除して未作成の案内を表示する。全600試行の集計はフィルタされない。
- 再生、一時停止、12.6秒の開閉開始、1／0.5／0.25倍、開閉ループ、終点静止画、動画→角度グラフのカーソル連動、グラフクリック→動画seekを追加。前段では開閉前と表示し、方式の終了後に角度曲線を延ばさない。
- 新規 `verify_adaptive_media.py` が全6MP4をffprobeとffmpeg全decodeで検証。18軌跡のhash・終了理由・PE/AAE/AE・成功分類、各2,355frameの状態対応、18個の真の終端、共通prefix・post-reset state・clamp・素材/source/cache hashを照合しPASS。0017／0086／0037の終端画像も目視確認した。
- 追加前後で plan.json・results.json・independent_audit.json・metricstable.csv のファイルhashと、summary.groupsの正規化JSON hashがすべて完全一致した。定量結果は変更していない。reportの表示用source hashとHTMLのみ更新した。

成果物は同じ [定量・動画比較HTML](runs/comparison_seed202_adaptive_v1/index.html)、[動画manifest](runs/comparison_seed202_adaptive_v1/qualitative/manifest.json)、[メディア検証](runs/comparison_seed202_adaptive_v1/qualitative/media_verification.json)。README手順9に閲覧・再描画・レポート再生成方法を追記した。

最終HTMLの実Chromium検証もPASS。6本すべての再生進行・1920×880表示・12.6秒seek・最終PTSと真の物理終端時刻・終点画像・早期終了表示を確認した。normal2/OOD4の動画フィルタ、解除後200条件×3方式の数値と停止点、600試行集計の保持、角度グラフ2秒と動画14.6秒の対応、0.5/0.25倍、開閉ループ、動画なしケースへの切替時の古い映像解除も確認。console/asset error0、390px幅で横はみ出しなし。最終HTML SHA256は `ddad4e250e6b8765e0255bc8310cec2dbcc0169454eebc83664b752a946fcc82`。run直下の [browser_verification.json](runs/comparison_seed202_adaptive_v1/browser_verification.json) を動画対応版の検証記録へ更新した。
