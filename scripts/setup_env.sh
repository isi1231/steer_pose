#!/usr/bin/env bash
# SteerPose 复现 —— 服务器环境一键配置（Ubuntu + H100 / CUDA 12.x）
#
# 用法:
#   bash scripts/setup_env.sh              # 用 conda 建环境 steerpose
#   bash scripts/setup_env.sh --venv       # 用 python venv 建环境
#   bash scripts/setup_env.sh --cpu        # 装 CPU 版 PyTorch（无 GPU 机器）
#
# 国内服务器加速: 脚本已默认使用清华 PyPI 镜像，如需官方源加 --no-mirror
set -euo pipefail

ENV_NAME=steerpose
PY_VER=3.10
USE_VENV=0
CPU=0
MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

for arg in "$@"; do
  case "$arg" in
    --venv) USE_VENV=1 ;;
    --cpu) CPU=1 ;;
    --no-mirror) MIRROR="" ;;
    *) echo "未知参数: $arg"; exit 1 ;;
  esac
done

if [ "$CPU" = "1" ]; then
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
  TORCH_SPEC="torch"
else
  TORCH_SPEC="torch"
fi

echo "=== 1/4 创建 Python 环境 ==="
if [ "$USE_VENV" = "1" ]; then
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PIP="pip"
else
  if ! command -v conda >/dev/null 2>&1; then
    echo "未找到 conda，请改用: bash scripts/setup_env.sh --venv"; exit 1
  fi
  if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "环境 ${ENV_NAME} 已存在，跳过创建"
  else
    conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
  fi
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
  conda activate "${ENV_NAME}"
  PIP="pip"
fi

echo "=== 2/4 安装 PyTorch（CUDA 版）==="
if [ -n "$MIRROR" ] && [ "$CPU" = "0" ]; then
  # 清华 pytorch-wheels 镜像（国内快）
  ${PIP} install ${TORCH_SPEC} --index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu124 \
    || ${PIP} install ${TORCH_SPEC} --index-url "${TORCH_INDEX}"
else
  ${PIP} install ${TORCH_SPEC} --index-url "${TORCH_INDEX}"
fi

echo "=== 3/4 安装其余依赖 ==="
if [ -n "$MIRROR" ]; then
  ${PIP} install -r requirements.txt ${MIRROR}
else
  ${PIP} install -r requirements.txt
fi

echo "=== 4/4 验证环境 ==="
python - <<'PY'
import torch, numpy, scipy
print("torch      :", torch.__version__)
print("numpy      :", numpy.__version__)
print("scipy      :", scipy.__version__)
print("cuda 可用  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU        :", torch.cuda.get_device_name(0))
    print("算力       :", torch.cuda.get_device_capability(0))
    x = torch.randn(1024, 1024, device="cuda")
    print("矩阵乘法测试:", (x @ x).sum().item() is not None)
else:
    print("警告: 未检测到 CUDA，将只能跑 CPU（演示流程仍可跑通）")
PY

echo
echo "环境就绪。接下来:"
echo "  conda activate ${ENV_NAME}   # venv 用户: source .venv/bin/activate"
echo "  python scripts/prepare_data.py --demo --out data/poses_demo.npz   # 生成演示数据"
echo "  python -m steerpose.train --poses3d data/poses_demo.npz --epochs 50 --batch 512 \\"
echo "      --workers 4 --device cuda --out ckpt/demo.pt"
echo "  python -m steerpose.calibrate --ckpt ckpt/demo.pt --demo"
