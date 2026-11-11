# Ubuntuでの従来5条件のOOD目標状態評価

既定の実験はnormalと水平移動・持ち上げ・開閉OODの[全8通りの実験設計と実行手順](OOD_MULTIAXIS_DESIGN.md)です。このページは`--config configs/ood_eval.json`を明示して使う従来5条件の手順です。

論文と同じArtiGrasp・RaiSim・PyTorch環境が動いているUbuntu向けの追加実験です。ライセンス取得や環境の再構築は不要です。既存の学習済みモデルを使い、再学習しません。既にビルド済みの作業フォルダを`git pull`で更新する場合、今回の追加によるC++の再ビルドは不要です。

リポジトリのルートで、普段ArtiGraspを動かしているPython環境を有効にして実行してください。追加のPythonパッケージは必要ありません。動画検証には、既存のffmpegに含まれる`ffprobe`を使用します。

## まず実行するコマンド

**別の場所に新しくcloneした場合**は、そのcloneでシミュレータのPython拡張を一度ビルド・登録してください。元の作業フォルダでビルドした`.so`はGit管理外なので、新しいcloneには含まれません。同じconda環境・既存のRaiSim・ライセンスをそのまま使えます。

```bash
# 新しくcloneした artigrasp/ のルートで、既存のartigrasp環境を有効にした状態
cd raisimGymTorch
python setup.py develop
cd ..
```

既にこのcloneでビルド済みなら、上の手順は省略できます。

```bash
# artigrasp/ のルート。既存の論文用Python環境を有効にする。
python tools/ood_eval.py check

# 全10物体 × 5条件 × 各100サンプル = 5,000試行
python tools/ood_eval.py run --config configs/ood_eval.json --output results/ood_seed1
```

`check`は読み込んだパッケージの`raisimGymTorch_path`も表示します。正しい参照先は`…/artigrasp/raisimGymTorch/raisimGymTorch/__init__.py`です。`…/artigrasp/raisimGymTorch/__init__.py`という外側の同名ディレクトリを読む問題を避けるため、起動時に内側のパッケージの検索順位を先頭へ移します。`compose_eval`が未ビルドの場合は、上記のビルドコマンドをエラーと一緒に表示します。

正常終了後の比較表は`results/ood_seed1/report.md`です。計算中も完了した試行を順次保存します。計算を中断した場合は、同じ出力先を指定して再開します。

```bash
python tools/ood_eval.py run --config configs/ood_eval.json --output results/ood_seed1 --resume
```

最初に少数で接続確認する場合は、別の出力先で次を実行してください。この2例の結果を本番100例の結果と混ぜることはありません。

```bash
python tools/ood_eval.py run --config configs/ood_eval.json --output results/ood_smoke \
  --objects box ketchup --profiles id_control ood_articulation_180 \
  --samples 2 --batch-size 2
```

## 成功・失敗例をMP4にする

Ubuntuのデスクトップ／GUIセッションで、別ターミナルから既存のRaiSimUnityを起動し、Auto-connectを有効にします。カメラで物体と手、目標位置が見えるようにしてください。

```bash
cd raisimUnity/linux
./raisimUnity.x86_64
```

数値評価後、リポジトリのルートから実行します。

```bash
python tools/ood_record.py --results results/ood_seed1 --per-outcome 5
python tools/ood_eval.py report --results results/ood_seed1
```

各物体・各条件の評価済み100例から、**論文指標による成功5例・失敗5例まで**をサンプル番号順に選びます。保存済みの操作列と初期状態を再生し、全時刻の状態が元の数値評価と一致すること、および成功・失敗の判定が一致することを検証してから採用します。正常な失敗例を選び、数値異常による自動リセット試行は定性例から除外します。

成功が0例などの場合、存在しない成功動画を補充しません。取得数・不足数を動画一覧と結果表に記録します。既定では5条件すべてを録画するので、最大500本になります。条件を絞る場合は次のように指定できます。

```bash
python tools/ood_record.py --results results/ood_seed1 \
  --profiles ood_combined --per-outcome 5
```

Unityの保存場所が自動で見つからない場合は明示してください。バンドルされているUnityでは`raisimUnity/linux/Screenshot`や`screenshots`などです。

```bash
python tools/ood_record.py --results results/ood_seed1 \
  --unity-capture-dir raisimUnity/linux/Screenshot
```

数値評価と動画収集を続けて実行することもできます。先にUnityを起動してください。

```bash
python tools/ood_eval.py run --config configs/ood_eval.json --output results/ood_seed1 \
  --record-videos --unity-capture-dir raisimUnity/linux/Screenshot
```

公開実装の成功・失敗で選ぶ動画が必要なら、別の動画一覧に保存できます。

```bash
python tools/ood_record.py --results results/ood_seed1 \
  --metric public_success --per-outcome 5
```

実際のUnity録画を使い、再生は制御周期に合わせた実時間で進めます。MP4の映像ストリーム・長さ・フレーム数も検証します。Unity接続失敗、録画失敗、再生の不一致は成功扱いにせず、動画一覧とログに残して非ゼロの終了コードを返します。GUI接続のない数値評価は通常どおり実行できます。

## 目標条件

設定は[`configs/ood_eval.json`](../configs/ood_eval.json)です。単位は移動がm、設定の開閉角がdegree、保存される角度がradです。

| 条件名 | 持ち上げ目標 | 最終水平移動 | 開閉目標 |
| --- | --- | --- | --- |
| `id_control` | 公開評価どおり0.15 m、ketchupは0.30 m | x,yをそれぞれ−0.10〜−0.05 m | 公開評価の0.5〜0.6 radルール、ketchupは絶対角0.5〜1.0 rad |
| `ood_height` | 0.60〜0.90 m | 通常条件と同一 | 通常条件と同一 |
| `ood_translation` | 通常条件と同一 | 水平距離0.50〜0.80 m、方向は一様 | 通常条件と同一 |
| `ood_articulation_180` | 通常条件と同一 | 通常条件と同一 | 絶対関節角180° |
| `ood_combined` | 0.60〜0.90 m | 水平距離0.50〜0.80 m、方向は一様 | 絶対関節角180° |

姿勢目標の変化は公開コードに合わせ、通常物体ではEulerの第2成分に−0.4〜0.4 rad、ketchupでは−0.3〜0.3 radを加えます。論文の文章上のyawと公開コードの操作軸の差を埋める変更は加えていません。

`id_control`は公開評価の目標範囲に沿った対照条件です。論文の全ラベル組合せと同じ試行集合を再現するものではありません。初期ラベル対は復元抽出し、各物体・各サンプルの初期状態と乱数を条件間で揃えます。したがって同じ初期ラベルが繰り返される場合がありますが、目標は独立に抽出されます。`plan.json`に各ラベル・目標・seedをすべて保存します。

公開学習コード`general_two/runner.py`等のランダムな開閉目標は0.5〜1.5 radなので、180°はこのランダム目標範囲を超えます。空間移動には手続き的な制御が含まれ、全学習データの目標支持集合は公開されていません。高さ・水平移動のOODは、公開評価範囲を大きく超える外挿条件として解釈してください。

元の関節可動域を維持します。`espressomachine`の上限は1.5 rad、`ketchup`と`phone`は2.5 radなので、180°は可動域外です。値を勝手に上限へ丸めず、要求角と可動域を保存し、範囲外件数を表に出します。`target_reachable`は関節角だけの検査で、手の到達距離や机の支持範囲を保証する値ではありません。

高さ条件は移送前半の持ち上げ目標です。後半は元の実装どおり初期の高さに戻して机上に置き、開閉します。大きな高さに届かないまま置き直すケースも識別できるよう、最大上昇量・指定高さへの到達率を別に保存します。

## 成功率と誤差の定義

一次資料は[論文 §6.1 / Table 3 / Supplement B](https://arxiv.org/html/2309.03891v2)と、このリポジトリの[`compose_eval.py`](../raisimGymTorch/raisimGymTorch/env/envs/compose_eval/compose_eval.py)、[`ketchup_eval.py`](../raisimGymTorch/raisimGymTorch/env/envs/compose_eval/ketchup_eval.py)です。

| 指標 | 計算方法 |
| --- | --- |
| 論文の定義によるSuc.T | 最後の物理ステップ後に、位置PE < 0.05 m、姿勢AE < 0.2 rad、関節角AAE < 0.5 radを同時に満たす |
| PE | 実測した物体の最終基底位置と要求位置のユークリッド距離 |
| AE | 実測と要求のquaternion間のSO(3)測地距離。qと−qを同じ姿勢として扱う |
| AAE | 実測した有限可動域関節角と要求角の絶対差。2πで折り返さない |
| 公開実装互換の成功率 | 公開コードの到達・姿勢・開閉の独立した履歴フラグと終了時guardを再現する |
| 高さ目標到達率 | 持ち上げ区間中に、実測上昇量と指定上昇量の差が0.05 m未満になった割合 |
| OOD目標達成率 | 論文Suc.Tと高さ目標到達の両方を満たす本実験の補助指標 |

公開実装との相違は意図的に別々に扱っています。

- 公開コードは成功フラグを保持するため、位置・姿勢・関節角を別時刻に満たしても成功になります。
- 公開コードの`global_state[-5:-1]`は目標quaternionです。実測quaternionは`[125:129]`なので、公開の姿勢成功判定は通常、自分自身との比較になります。互換指標ではこの挙動を保持し、論文指標では実測値を使います。
- 標準の公開評価は最後に関節角`< 2 rad`を要求します。180°に正しく到達しても公開互換では失敗になり得ます。
- ketchupの公開開閉判定は目標角への到達ではなく、初期角から`0.3 rad`を超えて変化したかです。
- ketchupの元スクリプトは`set_obj_goal`だけを更新するため、表示の要求角は変わっても方策観測に使う`final_obj_angle_`は0のままです。本実験では開閉準備の段階で既存の`set_goals`にも要求角を渡します。通常条件とOOD条件の両方でこの修正を使い、`controller_changes`に記録します。公開実装互換の列は、こうして要求角を与えた同じ軌跡に対する**判定方法の互換性**を示します。元スクリプトと同一軌跡を意味しません。
- 元コードは`global_state`の更新順による1ステップ遅れがあります。公開互換ではこの順序も再現し、論文指標では最後のステップ後を使います。
- 公開コードは表示誤差をPE 0.1 m、AE 0.3 rad、AAE 1.0 radでクリップします。無加工値と表示互換のクリップ値を別の列に保存します。

`compose_eval`の138次元global stateのうち、`[122:130]`が実測物体状態、`[130:138]`が要求状態です。ネットワークの入力次元やC++状態の定義は変更していません。元評価と同じ正規化なしの観測、決定論的なactor出力を使います。ketchupの専用手順と`right_special`の接近、notebook/mixer等の右手モデル切替も保持しています。`mixer`のラベル列挙には`label_eval_compose`相当の選択を使い、元の`label_eval_floating`が開閉ラベルを除外してしまう問題を回避します。

## 論文との比較表

`report.md`は自動で次を出力します。

- 条件・物体別の論文Suc.T／公開実装互換成功率、成功数/100、Wilson 95%信頼区間。
- 実測のPE/AE/AAEと、公開表示のクリップ済み誤差。平均に使った測定数も併記。
- 論文の8物体の集約と、公開リポジトリの10物体の集約。物体平均と全試行平均。
- 同じ初期状態での対照条件とOOD条件の成功・失敗の変化。
- 持ち上げ到達・OOD目標達成、可動域外件数、異常試行、未実行数、動画リンクと不足数。

論文Table 3のArtiGrasp公開値はSuc.T **0.50**、PE **0.03 m**、AE **0.10 rad**、AAE **0.41 rad**です。D-Graspの値も比較用に収録しています。Table 3には物体別値が掲載されていないので、物体別の論文値は作っていません。

論文の8物体はbox / laptop / notebook / waffleiron / mixer / ketchup / espressomachine / microwaveです。capsulemachineとphoneも今回評価しますが、追加対象として分けます。これら2物体は既存の標準複合手順への拡張です。論文は約7,500試行の組合せであり、本実験の物体ごと均等100例とは試行の重み・対象目標が異なります。論文値との差は参考比較として読み、通常条件でも完全な再現値とはみなさないでください。

成功率の分母は予定した全100試行です。数値異常・環境の自動リセットは失敗に数え、黙って取り除きません。誤差の数値が測れない試行は誤差平均に0として混ぜず、測定数を示します。未実行がある表は暫定下限値であることを明記し、全試行完了後に最終評価してください。

## 出力と再現性

```text
results/ood_seed1/
  plan.json                 # 予定した全試行のラベル、seed、要求目標
  provenance.json           # コード・計画・モデル・入力データのSHA256、Git revision
  execution.json            # 実際のPython/NumPy/PyTorch、環境バイナリ等
  episodes.jsonl            # 各試行の判定、誤差、要求角、可動域、初期ラベル
  trajectories/*.npz        # 各時刻の実測状態、左右の操作列、初期状態、接触目標
  summary.json              # 集約結果
  report.md                 # 論文比較表・動画一覧へのリンク
  videos/paper_success/
    index.md                # MP4へのリンクと成功/失敗例の不足数
    index.json              # 録画検証の結果、MP4/軌跡SHA256
    <object>/<profile>/<success|failure>/*.mp4
```

録画のために全操作列・全状態を残すので、全5,000試行の軌跡は数GB程度以上の空き容量を見込んでください。動画は別途容量を使います。標準の出力先`results/`はGitの対象外です。

再開時は完了済みの試行を飛ばし、同じ計画の残りだけを実行します。計画・評価コード・入力データ・モデル・実行環境の変更を検出した場合、異なる実験を混ぜないように停止します。目標条件を変える場合は設定JSONを編集し、**新しい出力先**で実行してください。

先に計画だけを確認することもできます。このコマンドはRaiSimを起動せず、NumPyと既存データだけで実行できます。

```bash
python tools/ood_eval.py plan --config configs/ood_eval.json --output results/ood_seed2 --seed 2
python tools/ood_eval.py run --config configs/ood_eval.json --output results/ood_seed2 --resume
```

既定のバッチは100環境です。メモリ使用量を減らす場合も、計100例という評価数は変わりません。

```bash
python tools/ood_eval.py run --config configs/ood_eval.json --output results/ood_small_batch \
  --batch-size 10 --num-threads 4
```

既定は再現性を優先したCPU推論です。既存環境のCUDAを使う場合は新しい実験で`--device cuda`を指定できます。数値実験には実時間待機を入れず、動画再生だけ実時間で進めます。

```bash
python tools/ood_eval.py run --config configs/ood_eval.json --output results/ood_seed1_cuda --device cuda
```

GPUを使うのは左右の方策ネットワークの推論で、RaiSimの物理計算はCPUです。起動時に要求したデバイスを表示し、モデル読み込み後に実際のパラメータ配置とGPU名を確認して表示します。

```text
Requested policy inference device: cuda | RaiSim physics: CPU | threads: 4 | batch size: 100
Loading checkpoints on CPU; requested policy inference device: cuda
Policy inference device (verified parameters): cuda:0 (GPU名)
RaiSim physics device: CPU | environments: 100 | threads: 4
```

`torch.load(..., map_location="cpu")`はチェックポイントを最初にCPUへ読み込む指定です。その後モデルを指定デバイスへ移すため、この行やPyTorchの`FutureWarning`から推論先を判定することはできません。CUDAを指定して利用できなかった場合は停止し、CPUに自動で切り替わることはありません。

既に実行中の場合はコードを更新せず、別ターミナルで保存された設定を確認できます。実行フォルダの`execution.json`の`device`が推論先の要求設定です。完了した試行の`episodes.jsonl`には`policy_device`も保存します。

```bash
python -c "import json; print(json.load(open('results/ood_seed1_cuda/execution.json'))['device'])"
nvidia-smi
```

`nvidia-smi`で評価中のPythonプロセスを確認できます。GPU使用率は、CPU側の物理計算やCPU/GPU間転送を待って低くなる場合があります。表示だけを更新する場合も評価コードのハッシュが変わるため、実行中の実験は完了させてから更新し、次の実験は新しい出力先で開始してください。

保存した1例だけを録画なしで再生・照合する例です。

```bash
python tools/ood_eval.py replay --results results/ood_seed1 \
  --episode-id box__ood_combined__0000 \
  --replay-result /tmp/artigrasp_replay.json
```

集計し直す場合はシミュレータは不要です。

```bash
python tools/ood_eval.py report --results results/ood_seed1
```

## 実装の検証

```bash
python -m unittest discover -s tools/tests -v
```

論文／公開実装の判定差、180°と終了時guard、姿勢の参照位置、失敗・未実行の分母、条件間の初期状態対応、複合制御の段階切替、保存操作列の再生、動画の検証をテストしています。シミュレータを使わないテストはローカルで実行できます。Ubuntu実機での数値結果・Unity MP4は、上記コマンドを既存環境で実行した際に生成されます。
