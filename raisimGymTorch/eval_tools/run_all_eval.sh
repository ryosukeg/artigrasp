#!/usr/bin/env bash
# 全物体 x train/test で ArtiGrasp の評価スクリプトを順に実行し、ログと終了コードを保存する。
# 使い方: eval_tools/run_all_eval.sh [-t "tasks"] [-b "objs"] [-s "splits"] [-o out_dir]
#   tasks : floating_arti floating_grasp fixed compose
#   例    : eval_tools/run_all_eval.sh -t "fixed" -b "box laptop" -s "test"
# 同じ -o を指定して再実行すると、成功済みの実行はスキップする。
set -u
# stdout をバッファすると stderr の警告が値の行に割り込み、集計で値を取りこぼす
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TASKS="floating_arti floating_grasp fixed compose"
OBJS="box capsulemachine espressomachine ketchup laptop microwave mixer notebook phone waffleiron"
SPLITS="test train"
OUT=""

while getopts "t:b:s:o:h" opt; do
  case $opt in
    t) TASKS=$OPTARG ;;
    b) OBJS=$OPTARG ;;
    s) SPLITS=$OPTARG ;;
    o) OUT=$OPTARG ;;
    *) sed -n '2,7p' "$0"; exit 1 ;;
  esac
done
OUT=${OUT:-eval_results/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"

if ! python -c "from raisimGymTorch.env.bin import general_two" 2>/dev/null; then
  echo "raisimGymTorch をimportできません。conda activate artigrasp を確認してください。" >&2
  exit 1
fi

# compose_eval は general_two の full_400 と var_l400.csv を要求する
COMPOSE_VAR=data_all/general_two/2023-05-15-10-18-54/var_l400.csv

run_one() {
  local task=$1 obj=$2 split=$3
  local name="${task}_${obj}_${split}"
  local log="$OUT/$name.log" status="$OUT/$name.status"
  local script args=(-obj "$obj")

  if [[ -f $status && $(cat "$status") == 0 ]]; then
    echo "skip (done): $name"
    return
  fi

  case $task in
    floating_arti)
      # 関節を持たない物体のラベルは空で、評価が成立しない
      case $obj in mixer|ketchup|phone) echo "skip (no arti labels): $name"; return ;; esac
      script=floating_evaluation/runner_eval.py ;;
    floating_grasp)
      script=floating_evaluation/runner_eval.py; args+=(-grasp) ;;
    fixed)
      script=fixed_arti_evaluation/runner_eval.py ;;
    compose)
      if [[ ! -f $COMPOSE_VAR && -z ${FORCE_COMPOSE:-} ]]; then
        echo "skip (missing $COMPOSE_VAR): $name"; return
      fi
      if [[ $obj == ketchup ]]; then script=compose_eval/ketchup_eval.py
      else script=compose_eval/compose_eval.py; fi ;;
    *) echo "unknown task: $task" >&2; return ;;
  esac
  [[ $split == test ]] && args+=(-test)

  echo "run: $name"
  python "raisimGymTorch/env/envs/$script" "${args[@]}" >"$log" 2>&1
  echo $? >"$status"
  echo "  exit=$(cat "$status")"
}

for task in $TASKS; do
  for obj in $OBJS; do
    for split in $SPLITS; do
      run_one "$task" "$obj" "$split"
    done
  done
done

echo "logs: $OUT"
python eval_tools/summarize_eval.py "$OUT"
