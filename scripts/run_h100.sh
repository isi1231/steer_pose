#!/usr/bin/env bash
# H100 上跑 SteerPose 的推荐配置（BamaPig3D 真实规模）
#
# 用法:
#   bash scripts/run_h100.sh data/poses_bamapig_train.npz            # 默认档位 B
#   bash scripts/run_h100.sh data/poses_bamapig_train.npz quick      # 只验证链路
#   bash scripts/run_h100.sh data/poses_animal3d.npz agnostic        # class-agnostic
#
# 参数取舍说明见 docs/h100_tuning.md
#
# ⚠️ 为什么按"目标总步数"反推 epoch，而不是直接写 epoch 数：
#   这个模型只有 ~69K 参数，**epoch 极其便宜**；决定成败的是优化器总共走了多少步
#   （= 每轮批数 × epoch = ceil(训练对数 / batch) × epoch）。
#   曾经 quick 档是 epochs=30 / batch=512 / pairs=3000 → 每轮只有 5 步，总计 150 步，
#   模型会停在"输出平均姿态"的平凡基线上（Lkp ≈ 0.648），从日志上看就像数据或实现有问题。
#   实测同一份数据：250 步 → Lkp 0.636（=基线），2000 步 → 0.133。
#   train.py 现在会打印总步数与平凡基线，val Lkp 接近基线时会显式警告。
set -euo pipefail

POSES="${1:?用法: bash scripts/run_h100.sh <poses.npz> [quick|full|agnostic]}"
MODE="${2:-full}"
TAG="$(basename "${POSES%.npz}")"
mkdir -p ckpt log

case "$MODE" in
  quick)
    # 档位 A：跑通链路 + 产出一个"虽然不是论文水平、但至少学到了东西"的权重
    BATCH=256;   PAIRS=3000;  STEPS=12000; WORKERS=4;  LR=1e-3;  NAME="quick"
    ;;
  agnostic)
    # 档位 C：class-agnostic（Animal3D，40 物种）
    BATCH=2048;  PAIRS=50000; STEPS=40000; WORKERS=16; LR=1e-3;  NAME="agnostic"
    ;;
  full|*)
    # 档位 B：BamaPig3D 正式训练（推荐）
    BATCH=1024;  PAIRS=20000; STEPS=40000; WORKERS=12; LR=1e-3;  NAME="bamapig"
    ;;
esac

# 由目标步数反推 epoch：每轮步数 = ceil(训练对数 / batch)，训练对数 ≈ 0.7 * PAIRS
N_TRAIN=$(( PAIRS * 7 / 10 ))
SPE=$(( (N_TRAIN + BATCH - 1) / BATCH ))
[ "$SPE" -lt 1 ] && SPE=1
EPOCHS=$(( (STEPS + SPE - 1) / SPE ))
LOG_EVERY=$(( EPOCHS / 20 ))
[ "$LOG_EVERY" -lt 1 ] && LOG_EVERY=1
SAVE_EVERY=$(( EPOCHS / 4 ))
[ "$SAVE_EVERY" -lt 1 ] && SAVE_EVERY=1

echo "=== SteerPose 训练（$MODE）==="
echo "  poses   : $POSES"
echo "  batch   : $BATCH   pairs: $PAIRS   workers: $WORKERS"
echo "  每轮步数: $SPE   ×   epoch: $EPOCHS   =   总步数 ≈ $((SPE * EPOCHS))"

nohup python -m steerpose.train \
    --poses3d "$POSES" \
    --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR" --weight-decay 1e-5 \
    --workers "$WORKERS" --num-pairs "$PAIRS" --num-views 100 --num-rolls 20 \
    --amp --seed 0 --log-every "$LOG_EVERY" --save-every "$SAVE_EVERY" \
    --out "ckpt/${NAME}_${TAG}.pt" > "log/train_${NAME}_${TAG}.out" 2>&1 &

echo "  已在后台启动，PID $!"
echo
echo "查看进度:   tail -f log/train_${NAME}_${TAG}.out"
echo "结构化日志: tail -f ckpt/${NAME}_${TAG}_train.log"
echo
echo "!! 先看日志里这两行，判断训练是否真的在学："
echo "     [i] 优化规模：每轮 N 步 × M epoch = 共 K 步"
echo "     [i] 平凡基线（输出平均姿态）val Lkp = X"
echo "   val Lkp 必须明显低于 X。若结束时报'与平凡基线几乎相同'，说明步数不够，"
echo "   再加大 --epochs 即可 —— 不要去怀疑数据或实现（用 scripts/diag_lkp_plateau.py 可复核）。"
echo
if [ "$MODE" = "quick" ]; then
  echo "!! quick 档位只用于验证链路，模型区分度仍不足，标定结果仅供流程自测。"
  echo "   正式训练请跑: bash scripts/run_h100.sh $POSES"
  echo
fi
echo "训练完成后，标定前请依次确认（都很快）:"
echo "  0) 数据体检（3D GT 重投影 vs 官方 2D 标注，秒级，不需要权重）"
echo "     python scripts/check_mammal_data.py --root /data/BamaPig3D_pure_pickle"
echo "  1) 推断链实现自检（5 秒）"
echo "     python scripts/selftest_infer.py"
echo
echo "完整标定流程（两视角，相机 0 <-> 6）:"
echo "  a) 造数据（用官方 2D 标注 + 真值外参，可顺便算论文的 ER/Et）"
echo "     --root 会自动读到内参 newcameramtx 并写进 npz，标定时无需再传 --focal"
echo "     python scripts/prepare_mammal_2d.py \\"
echo "         --root /data/BamaPig3D_pure_pickle \\"
echo "         --cam-a 0 --cam-b 6 --min-frame 1400 \\"
echo "         --out data/two_view_pig_c0_c6.npz"
echo "  b) 标定（用 _best.pt；npz 里的 K 会被自动采用）"
echo "     python -m steerpose.calibrate --ckpt ckpt/${NAME}_${TAG}_best.pt \\"
echo "         --poses-npz data/two_view_pig_c0_c6.npz --out out/two_view_c0_c6.npz"
echo "  c) 看输出里的 ER / Et / 匹配置信度 / 几何项用了几对匹配"
echo "     （置信度 < 0.5 或几何项标了'静默失效'，则结果不可信）"
