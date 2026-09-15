# -*- coding: utf-8 -*-
"""SteerPose 网络（论文 Fig.2 + 附录 D.2）.

输入:
  - N 个关节的 2D 坐标 (B, N, 2)
  - 目标视角的相对旋转 R，Rodrigues 向量 (B, 3)
  - 可选: 关节可见性掩码 (B, N)，1=可见 0=缺失（模拟遮挡）
结构:
  关节坐标 / 旋转向量 --MLP--> 32 维 token（共 N+1 个）
  + 可学习位置编码（look-up embedding，N+1 个 token）
  --Transformer 编码器 x5（多头自注意力 + FFN）-->
  均值池化 --MLP--> N 个关节的 2D 坐标 q(R)

注：官方代码（github.com/kcvl-public/steerpose）截至 2026-09 尚未公开，
本实现依据论文正文与附录 D.2 独立实现；未注明的细节（MLP 层数、注意力头数、
FFN 维度）为合理假设。
"""
import torch
import torch.nn as nn


class SteerPose(nn.Module):
    def __init__(
        self,
        num_joints: int = 20,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 5,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.d_model = d_model

        self.joint_mlp = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.rot_mlp = nn.Sequential(
            nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

        # 可学习位置编码：N 个关节 token + 1 个旋转 token
        self.pos_embedding = nn.Parameter(torch.zeros(num_joints + 1, d_model))
        # 被掩码关节的可学习替换 token
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, num_joints * 2)
        )

    def forward(
        self,
        joints2d: torch.Tensor,           # (B, N, 2) 归一化 2D 关节坐标
        rotvec: torch.Tensor,             # (B, 3) 或 (3,)
        joint_mask: torch.Tensor = None,  # (B, N) 1=可见 0=缺失
    ) -> torch.Tensor:
        B = joints2d.shape[0]
        if rotvec.dim() == 1:
            rotvec = rotvec.expand(B, 3)
        if joint_mask is None:
            joint_mask = torch.ones(
                B, self.num_joints, device=joints2d.device, dtype=joints2d.dtype
            )
        mask = joint_mask.unsqueeze(-1)

        joints_in = joints2d * mask
        joint_tok = (
            self.joint_mlp(joints_in)
            + (1.0 - mask) * self.mask_token
            + self.pos_embedding[: self.num_joints]
        )
        rot_tok = self.rot_mlp(rotvec) + self.pos_embedding[self.num_joints]

        tokens = torch.cat([joint_tok, rot_tok.unsqueeze(1)], dim=1)
        out = self.encoder(tokens)
        return self.head(out.mean(dim=1)).view(B, self.num_joints, 2)
