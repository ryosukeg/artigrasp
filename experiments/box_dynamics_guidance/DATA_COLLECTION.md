**box の補正付き応答データの収集仕様**

この仕様を [collect.py](collect.py) に実装しました。既存の `tools/ood_eval.py` による補正なし軌跡と区別して扱います。実行計画は [planning.py](planning.py)、読み戻しは [dataset.py](dataset.py)、記録の監査は [audit_collection.py](audit_collection.py) が担当します。

**学習する関係**

固定 PPO が各 step の観測に反応しながら、一定の手首補正 z を H=20 step 加えた場合の物体応答を予測します。入力は開始時の左右観測・物体状態・元の目標・z、教師はシミュレータで実測した窓の終了状態です。任意の方策に共通する 1 step の物理遷移モデルとは異なり、固定 PPO の応答も含みます。

**補正を加える位置と単位**

- `Controller.phase(step)` と観測取得を済ませ、`policy.actions(obs_r, obs_l)` を呼び出した後に補正する。
- box の step 1,260〜1,559 かつ右手 mode 8 で、`right[:, 0:3]` へ z を加える。
- 左手、右手の残り 48 成分、最終目標、終了時刻は変えない。
- z は action 座標で表す。現在の root-guided 設定では z の 1 単位は手首並進の制御目標 0.001 m に対応する。世界座標の上下左右と同一視せず、action の座標変換を確認して記録する。
- 毎 step 更新される制御目標への補正であり、実現移動量・速度や、毎 step 1 mm ずつ加算する軌道指令ではない。
- action の並進座標は step 1,260 の `reset_right_hand` で確定する root 座標に固定される。保存する `action_frame_quaternion_wxyz` を R に変換すると、名目上の世界座標での PD 目標差は `R @ (0.001 * z)`。実際の手首移動と異なり、PD 目標には関節制限での clip もある。
- 初期較正ではゼロと各軸正負の action 0.5、1.0 単位を試す。大きさは本収集前に実測で決め、学習と評価で補正範囲をそろえる。
- 各窓で選んだ z を 20 step 保持する。PPO の行動自体は各 step で再計算する。窓をまたいで異なる補正を混ぜた教師を作らない。

**参照と目標の分離**

公式 `train_set/box.npy` から、学習用と検証用の参照元グループを先に分けます。把持参照と開閉参照の双方が対応する分割に収まるようにペアを抽出し、分割 manifest を保存します。参照元の識別子には subject、sequence、参照フレームと種類を記録します。公式 test split は最終比較まで使用しません。

初期の収集目標角案は 0.5〜1.5 rad、他の空間目標は通常条件です。これは論文 normal 評価の 0.5〜0.6 rad より広い収集専用分布なので、既存の `normal_goal.source=paper_supplement_b4` の値だけを書き換えて論文 normal と表示する実装は避けます。収集分布を明示する sampler と manifest が必要です。

**保存するデータ**

| 項目 | 内容 |
|---|---|
| 試行識別子 | object、episode ID、参照識別子、学習または検証の分割、seed |
| 実行条件 | ソース・拡張 binary・PPO checkpoint の hash、設定、control_dt、H |
| 時刻とフェーズ | 各 step の番号、制御 mode、窓の開始・終了番号 |
| 開閉開始時の状態 | 基部の位置・姿勢誤差、関節角、既に終了した持ち上げ区間の成否。全試行と開始状態別の補助集計に使う |
| 観測 | 最終開閉区間の各 step の未標準化左右観測と窓の終了観測 |
| 目標 | 元の要求角、基部の目標位置・姿勢。評価式と同じ真の目標 |
| 元の行動 | 各 step の PPO 左右 action |
| 補正と実行行動 | z と、実際に `env.step` へ渡した左右 action |
| 実測状態 | 各 step の global state。窓の終点の角度・角速度・基部位置姿勢も保存 |
| 有効性 | 非有限観測、非有限 action、環境の自動 reset、異常が最初に起きた step |

既存の 138 次元 global state だけでは、観測に含まれる速度や接触情報を復元できません。右観測の関節角・角速度は現在の実装で index 273・274、global state の実測物体位置は 122:125、quaternion は 125:129、関節角は 129 です。実装時に次元と対応を検証し、保存スキーマを version 管理します。終点の角速度には終点の観測が必要です。

数値異常や自動 reset が起きた窓と、それ以後の同一試行の窓は学習対象から外し、件数と理由を記録します。前段での異常も保持します。タスクに失敗したが数値的には正常な窓は残します。初期化・前段・実行 action の記録も残し、同じ前段から候補を比較できるようにします。

**保存スキーマ version 1**

`plan.json` に目標・参照分割・全窓の補正を先に保存します。`provenance.json` に収集ソース、既存 runtime、拡張 binary、checkpoint、参照資産、URDF と参照 mesh の hash を記録します。batch ごとに NPZ を atomic rename で保存し、全件を保存した後で `batches/batch_XXXX.json` を完了記録として作成します。全計画の完了を確認してから `dataset.json` を作成し、loader は完成した dataset のみ読みます。

| NPZ の配列 | 形状・意味 |
|---|---|
| `states` | 1,561 × 138。既存 evaluator と同じ切替前の保存時刻規約 |
| `actions_r/l`, `ppo_actions_r/l` | 各 1,560 × 51。実行 action と補正前 PPO 出力 |
| `residuals` | 1,560 × 3。前段はゼロ、最終区間は窓ごとに一定 |
| `final_obs_r/l` | 各 301 × 276。最終開閉の未標準化観測と最後の実行後観測 |
| `final_states` | 301 × 138。最初は phase 1,260 の手の切替を終えた直後 |
| `window_start_step/end_step` | 各15。開始1,260〜1,540、終了1,280〜1,560、終了はexclusive |
| `window_z`, `window_valid` | 15 × 3 と15。補正値と学習可能性 |
| `goal` | 8。真の物体基部位置3、wxyz姿勢4、蓋角度1 |
| `modes_r/l` | 各1,560。最終区間の右8・左7を検証 |
| `bad_observation`, `bad_action`, `bad_state`, `auto_reset` | 各1,560。数値異常と自動リセット |
| `initial_invalid`, `terminal_bad_observation` | 初期状態と最後の観測・状態の異常 |

step 1,260 では `Controller.phase` が手の状態を切り替えるため、最初の窓入力は `final_states[0]` を使います。`states[1260]` と取り違えません。窓の入力・教師はそれぞれ `final_*[start-1260]`、`final_*[end-1260]` です。300 action に対して観測は301個必要です。

loader の入力は左右 raw 観測552成分、物体状態8、目標8、補正3の計571成分。教師は終了時の蓋角度・角速度、基部位置3、wxyz姿勢4の計9成分です。姿勢は開始時に対して終点と目標の quaternion の符号を合わせます。[train_smoke.py](train_smoke.py) は128ユニット×2層、5 epochの読み込み・学習・検証確認用で、本学習の精度を確定するモデルではありません。

**少量収集の完了条件**

1. ゼロ補正で既存 baseline と実行 action・物体軌跡を比較し、差を確認する。
2. 最終開閉以外の action が変化しないことを確認する。
3. 1 窓につき H=20 回の実行と、開始・終了観測が対応することを確認する。
4. 読み戻したデータから学習用 batch を作れ、短い学習と検証が完走する。
5. 本収集に使う補正量、参照分割、目標分布、再現用 seed を確定し、WORK_LOG に記録する。

保存は batch ごとに完了させ、中断後に完了済み batch を再利用できる構成にします。未完了ファイルを完成した dataset として読ませず、計画と完了記録を照合します。
