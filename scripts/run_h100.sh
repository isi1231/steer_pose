#!/usr/bin/env bash
# H100 上跑 SteerPose 的推荐配置（BamaPig3D 真实规模）
#
# 用法:
#   bash scripts/run_h100.sh data/poses_bamapig_train.npz            # 默认档位 B
#   bash scripts/run_h100.sh data/poses_bamapig_train.npz quick      # 只验证链路
#   bash scripts/run_h100.sh data/poses_animal3d.npz agnostic        # class-agnostic
#
# 参数取舍说明见 docs/h100_tuning.md
set -euo pipefail

POSES="${1:?用法: bash scripts/run_h100.sh <poses.npz> [quick|full|agnostic]}"
MODE="${2:-full}"
TAG="$(basename "${POSES%.npz}")"
mkdir -p ckpt log

case "$MODE" in
  quick)
    # 档位 A：5 分钟内确认环境/数据/显存都没问题
    EPOCHS=30;  BATCH=512;   PAIRS=3000;  WORKERS=4;  LR=1e-3;  NAME="quick"
    ;;
  agnostic)
    # 档位 C：class-agnostic（Animal3D，40 物种）
    EPOCHS=800; BATCH=8192;  PAIRS=50000; WORKERS=16; LR=1e-3;  NAME="agnostic"
    ;;
  full|*)
    # 档位 B：BamaPig3D 正式训练（推荐）
    EPOCHS=400; BATCH=4096;  PAIRS=20000; WORKERS=12; LR=1e-3;  NAME="bamapig"
    ;;
esac

echo "=== SteerPose 训练（$MODE）==="
echo "  poses   : $POSES"
echo "  epochs  : $EPOCHS   batch: $BATCH   pairs: $PAIRS   workers: $WORKERS"

nohup python -m steerpose.train \
    --poses3d "$POSES" \
    --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR" --weight-decay 1e-5 \
    --workers "$WORKERS" --num-pairs "$PAIRS" --num-views 100 --num-rolls 20 \
    --amp --seed 0 --log-every 20 --save-every 100 \
    --out "ckpt/${NAME}_${TAG}.pt" > "log/train_${NAME}_${TAG}.out" 2>&1 &

echo "  已在后台启动，PID $!"
echo
echo "查看进度:   tail -f log/train_${NAME}_${TAG}.out"
echo "结构化日志: tail -f ckpt/${NAME}_${TAG}_train.log"
echo
if [ "$MODE" = "quick" ]; then
  echo "!! quick 档位只用于验证链路能否跑通，模型区分度不足，标定结果不可用。"
  echo "   正式训练请跑: bash scripts/run_h100.sh $POSES"
  echo
fi
echo "标定前请先确认推断链实现无误（5 秒）:"
echo "  python scripts/selftest_infer.py"
echo
echo "完整标定流程（两视角，相机 0 <-> 6）:"
echo "  1) 造数据（用官方 2D 标注 + 真值外参，可顺便算论文的 ER/Et）"
echo "     python scripts/prepare_mammal_2d.py \\"
echo "         --kp2d /data/BamaPig3D_pure_pickle/label_keypoints2d.pkl \\"
echo "         --extrinsics /data/BamaPig3D/extrinsic_camera_params \\"
echo "         --cam-a 0 --cam-b 6 --min-frame 1400 \\"
echo "         --out data/two_view_pig_c0_c6.npz"
echo "  2) 标定（--focal 传像素焦距，估计值可用 0.9 x 图像长边）"
echo "     python -m steerpose.calibrate --ckpt ckpt/${NAME}_${TAG}_best.pt \\"
echo "         --poses-npz data/two_view_pig_c0_c6.npz --out out/two_view_c0_c6.npz"
echo "  3) 看输出里的 ER / Et / 匹配置信度（置信度 < 0.5 则结果不可信）"
