# Ubuntuでのnormal＋3種類のOOD：全8通りの評価

既存の論文用Ubuntu環境・ライセンス・学習済みモデルを使用し、再学習しません。設定は[`configs/ood_multiaxis.json`](../configs/ood_multiaxis.json)、プロトコル名は`factorial_v1`です。CLIの既定設定もこの8条件になっています。

## 評価する8通り

水平移動・持ち上げ・開閉の3因子をそれぞれnormal/OODに切り替える`2³ = 8`通りです。向きのOODや追加強度の条件は含めません。向きは全条件でnormalと同じです。

| 条件名 | 水平移動OOD | 持ち上げOOD | 開閉OOD |
| --- | --- | --- | --- |
| `normal` | OFF | OFF | OFF |
| `ood_horizontal` | ON | OFF | OFF |
| `ood_lift` | OFF | ON | OFF |
| `ood_articulation` | OFF | OFF | ON |
| `ood_horizontal_lift` | ON | ON | OFF |
| `ood_horizontal_articulation` | ON | OFF | ON |
| `ood_lift_articulation` | OFF | ON | ON |
| `ood_all` | ON | ON | ON |

10物体 × 8通り × 各100サンプル = **8,000試行**です。論文対象8物体（box、laptop、notebook、waffleiron、mixer、ketchup、espressomachine、microwave）と追加2物体（capsulemachine、phone）は結果を分けて集約します。`--suite all`と従来の`--suite core`は、どちらもこの8通りを選びます。

## normalとランダムなOODの範囲

normalの最終目標は、[論文Supplement B.4](https://arxiv.org/html/2309.03891v2#A2.SS4)に記載された範囲に合わせます。目標は初期物体姿勢を基準に生成します。

| 要素 | normal | OODがONのとき |
| --- | --- | --- |
| 最終水平移動 | x,yをそれぞれ一様に−0.10〜−0.05 m | 水平距離を一様に0.30〜0.80 m。方向は対応するnormalと同じ |
| 移送途中の持ち上げ | 公開制御の0.15 m、ketchupは0.30 m | 上昇量を一様に0.60〜0.90 m |
| 最終開閉角 | 全物体で絶対角0.50〜0.60 radを一様抽出 | 下記の可動域内・学習ランダム目標範囲外の区間から一様抽出 |
| ベースの向き | 初期姿勢への世界Z軸yawノイズを一様に−0.40〜0.40 rad | normalと同一 |
| 最終位置のz | 初期の机上高さから変化なし | normalと同一 |

水平移動はnormalの方向を保ち、距離だけを外挿します。0.30 m以上は論文の把持評価で使われるxy範囲（各±0.15 m）の最大水平距離約0.212 mも超えます。持ち上げは同評価の高さ上限0.45 mを超えます（[Supplement B.3](https://arxiv.org/html/2309.03891v2#A2.SS3)）。全学習データの空間目標支持集合は公開されていないため、空間OODはこれらの公表範囲に対する外挿です。

開閉OODは公開学習コードのランダム目標範囲`[0.5, 1.5] rad`を基準に、候補区間`[0.20, 0.40] ∪ [1.60, 3.10] rad`を使います（約11〜23°、92〜178°）。各物体のURDF可動域から両端0.05 radを除いた範囲との共通部分を取り、**残った区間の合計長に比例して一様抽出**します。抽出後に値を丸める処理はありません。

- espressomachineは上限1.5 radなので、下側の0.20〜0.40 radから抽出します。
- ketchup・phoneは上限2.5 radなので、上側の候補は2.45 radまでです。
- その他の上限がπ radの物体では、上側の候補はπ−0.05 radまでです。
- 有効な区間がなくなる設定はエラーにします。分布内の角度を代わりに使いません。

毎試行で異なる乱数から値を選び、`--seed`で再現できます。同じ物体・sample_indexの8通りは参照ラベル・初期状態を共有し、各因子をONにしたときの値も共有します。例えば`ood_lift`と`ood_all`の持ち上げ量は同一です。抽出値、有効な角度区間、関節の可動域とOOD判定を`plan.json`へ保存し、実際に環境へ渡した目標を試行結果にも保存します。

normalは**論文の目標分布に合わせた対照条件**です。公開制御の段階・時間・途中の持ち上げ量を使い、参照ペアを100例抽出するため、論文の全評価手順やTable 3の数値を完全再現する設定とは区別します。従来版の初期角に応じた相対開閉目標やEuler Y回転は、この8条件のnormalには使いません。

## Ubuntuで実行

既存のartigrasp環境を有効にし、リポジトリのルートで実行します。新しくclone/worktreeを作った場合だけ、そのcheckoutで`cd raisimGymTorch`、`python setup.py develop`、`cd ..`を一度実行します。既存checkoutのC++再ビルドは不要です。

```bash
python tools/ood_eval.py check

# まず全8通りを少数で確認（2物体 × 8通り × 2例 = 32試行）
python tools/ood_eval.py run --objects box ketchup --samples 2 \
  --output results/ood8_smoke --device cuda

# 本番：10物体 × 8通り × 100例
python tools/ood_eval.py run --output results/ood8_seed1 --device cuda
```

複数行のコマンドは行末の`\`を含めて入力してください。1行にまとめても実行できます。GPUは方策推論に使い、起動ログの実際のactor parameter deviceが`cuda:0`等であることを確認します。RaiSimの物理計算はCPUです。

### GPU使用と所要時間の確認

`Policy inference device (verified parameters): cuda:0 (NVIDIA RTX A5500)`なら、モデルはGPUへ配置されています。実際の初回推論でも`Policy forward (verified tensors)`に入力・出力のデバイスとworker PIDを表示し、CUDAテンソルの使用メモリを表示します。CPUのNumPyへ戻した操作をRaiSimへ渡すため、GPU推論とCPU物理計算を各ステップで交互に行います。

GPU側のネットワークは128ユニット×2層と小さく、2環境など少数のバッチではGPU使用率が低くなります。`nvidia-smi`の利用率はサンプリング期間内にGPUカーネルが実行された時間の割合なので、0%だけではCPU推論と判断できません（[NVIDIAの説明](https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization)）。

データ読み込み、環境初期化、モデル読み込み、ロールアウト、集計・保存の所要時間を表示します。ロールアウトはさらにCPU物理計算＋状態取得、方策推論＋CPU/GPU転送、その他に分けます。方策の時間は転送や待ち時間を含む実時間であり、GPUカーネルだけの時間ではありません。記録は`performance/<バッチ先頭のepisode_id>.json`に保存します。

物理計算のスレッド設定を後から壊さないよう、PyTorchのCPUスレッド数も`--num-threads`と揃えます。以前の`torch.set_num_threads(1)`はOpenMPを共有する構成で物理計算まで1スレッドへ変更する可能性があり、修正しています（[PyTorch 2.4.1の実装](https://github.com/pytorch/pytorch/blob/v2.4.1/aten/src/ATen/ParallelOpenMP.cpp#L42)）。ログのRaiSimスレッド数は要求設定であり、実際に動作したCPUスレッド数の測定値ではありません。

時間を切り分けるには、まず**1条件だけ**を実行します。

```bash
python tools/ood_eval.py run --objects box --profiles normal --samples 2 \
  --num-threads 2 --device cuda --output results/ood8_timing_cuda
```

2物体×全8条件のsmoke実験では16回の環境・モデル初期化が発生します。少数例のsmokeにかかった総時間を、そのまま100例の本番へ比例換算しないでください。`--batch-size`だけを増やしても`--samples 2`なら同時実行は2例のままです。本番の同時実行数は最大100例で、CPUの並列度は`--num-threads`から調整できます。速度の改善幅はUbuntuでの計測結果を確認してください。

先に目標計画だけ確認する場合です。計画生成にはRaiSimやGPUは不要です。

```bash
python tools/ood_eval.py plan --output results/ood8_seed1
python tools/ood_eval.py run --output results/ood8_seed1 --resume --device cuda
```

中断からの再開も、同じ出力先の保存済み計画を使用します。

```bash
python tools/ood_eval.py run --output results/ood8_seed1 --resume
```

任意の条件だけを確認する場合は、例えば`--profiles normal ood_all`を指定できます。通常の本番では指定せず、全8通りを実行してください。新しい乱数での追加実験は`--seed 2 --output results/ood8_seed2`のように出力先も分けます。

## 定量評価とMarkdownの結果表

実行後は`results/ood8_seed1/report.md`と`summary.json`へ集計します。途中でも次のコマンドで集計し直せます。

```bash
python tools/ood_eval.py report --results results/ood8_seed1
```

- **論文Suc.T**：最後のstep後に位置誤差 < 0.05 m、SO(3)姿勢誤差 < 0.2 rad、関節角誤差 < 0.5 radを同時に満たす成功率。
- **公開実装互換**：元実装の時刻ごとの成功保持、1ステップ前の測定、姿勢の添字、終了時guardを含む判定式を同じ軌跡へ適用した成功率。
- **OOD目標達成**：論文Suc.Tに加え、持ち上げ区間に目標上昇量との絶対誤差 < 0.05 mを達成した割合。最終位置が机上なので、高さOODの評価に使います。
- **厳密目標達成**：OOD目標達成の関節角閾値を0.1 radへ狭めた補助指標。小角度目標で既に論文閾値内だった件数も併記します。

各物体・各条件の成功数/100、Wilson 95%区間、無加工誤差、公開表示のクリップ値、normalとの対応比較を表にします。8通りが揃った初期状態の組から、他因子について平均した各OODのON−OFF差と、2因子・3因子の差の差もppで出します。これらは記述統計で、有意差検定は行いません。

論文8物体と全10物体の集約を分け、[Table 3](https://arxiv.org/html/2309.03891v2#S6.T3)の公表値を併記します。論文は約7,500試行の集約で、本実験とはサンプル数・重み・目標条件が異なります。全計画試行を主表の分母にし、実行異常は失敗、未完了は暫定下限として明示します。

ketchupでは要求した開閉角を方策の観測にも渡す既存の補正をnormal/OOD両方に適用します。`controller_changes`とレポートに記録します。公開実装互換は判定式の互換性であり、元スクリプトと同一軌跡という意味ではありません。

## 成功・失敗例のMP4

UbuntuのGUIセッションで既存のRaiSimUnityを起動し、Auto-connectを有効にします。その後、定量評価した操作列を再生して録画します。

```bash
python tools/ood_record.py --results results/ood8_seed1 --per-outcome 5
python tools/ood_eval.py report --results results/ood8_seed1
```

既定では`ood_goal_success`に従い、各物体・各条件から成功5例・失敗5例までを選びます。最大800本です。成功0例の場合は成功動画の不足を記録します。元の状態軌跡と判定が一致する実MP4だけを採用します。動画の一覧は`videos/ood_goal_success/index.md`に保存します。

論文指標で選別したい場合は`--metric paper_success`、一部だけ録画する場合は`--profiles normal ood_all`を追加できます。録画の詳細やキャプチャ場所の指定は[録画の手順](OOD_EVALUATION_UBUNTU.md#成功失敗例をmp4にする)を参照してください。

## 旧実験との併用

既に実行中の実験は動画収集まで同じ版で完了してください。評価コードと入力のハッシュを検証するため、**実行中のcheckoutでgit pullすると、その実験の再開・録画で版の不一致になります**。新しい8条件は別の出力先を使います。

先に新しい版を準備するなら、別のworktreeを作れます。

```bash
cd ~/goto/artigrasp
git fetch origin
git worktree add --detach ../artigrasp-ood8 origin/main
cd ../artigrasp-ood8/raisimGymTorch
python setup.py develop
cd ..
```

元のコードは変更しません。新しいcheckoutで環境をビルド・登録すると同じconda環境の参照先が変わるため、旧実験の再開・録画は元のcheckoutから行い、そのcheckoutの拡張が読み込まれることを`check`で確認してください。

従来5条件は`--config configs/ood_eval.json`、以前の17条件等は`--config configs/archive/ood_multiaxis_legacy.json`で明示的に選べます。既定では実行されません。既存の結果に異なる版・異なる計画の試行を追加せず、保存済み計画と元の版で再開してください。
