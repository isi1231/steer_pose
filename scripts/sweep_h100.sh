#!/usr/bin/env bash
# 小规模超参扫描：因为单个模型只有 ~69K 参数，在 H100 上每次训练只要几分钟，
# 扫描比反复调参更划算。挑 val Lkp 最低的配置再跑长训练。
#
# 用法:
#   bash scripts/sweep_h100.sh data/poses_bamapig_train.npz
set -euo pipefail

POSES="${1:?用法: bash scripts/sweep_h100.sh <poses.npz>}"
TAG="$(basename "${POSES%.npz}")"
mkdir -p ckpt log
SUMMARY="ckpt/sweep_${TAG}_summary.txt"
: > "$SUMMARY"

# 视角对数量 x batch（模型太小，显存永远够，重点看这对组合）
for PAIRS in 3000 10000 20000 50000; do
  for BATCH in 2048 4096; do
    NAME="sweep_p${PAIRS}_b${BATCH}_${TAG}"
    echo "=== 跑 $NAME ==="
    python -m steerpose.train \
        --poses3d "$POSES" \
        --epochs 300 --batch "$BATCH" --lr 1e-3 --weight-decay 1e-5 \
        --workers 12 --num-pairs "$PAIRS" --amp --seed 0 \
        --log-every 50 --save-every 0 --out "ckpt/${NAME}.pt" \
        > "log/${NAME}.out" 2>&1 || { echo "  [失败] 见 log/${NAME}.out"; continue; }

    BEST=$(grep -oE 'best val Lkp [0-9.]+' "ckpt/${NAME}_train.log" | tail -1 | awk '{print $4}')
    FINAL=$(grep -oE 'val Lkp [0-9.]+' "ckpt/${NAME}_train.log" | tail -1 | awk '{print $3}')
    echo "pairs=${PAIRS} batch=${BATCH} best_val_Lkp=${BEST} final_val_Lkp=${FINAL}" | tee -a "$SUMMARY"
  done
done

echo
echo "=== 汇总（按 best val Lkp 升序）==="
sort -t= -k4 -g "$SUMMARY" | tee "${SUMMARY%.txt}.sorted.txt"
echo
echo "挑最好的那组，用 scripts/run_h100.sh 跑长训练（把 --num-pairs 换成对应值）"
