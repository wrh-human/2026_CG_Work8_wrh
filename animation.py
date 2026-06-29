# -*- coding: utf-8 -*-
"""
选做内容：LBS 姿态动画
固定 shape 参数，让右肩关节从 0 逐渐旋转到某个角度，
观察权重区域如何随骨骼运动被平滑带动，生成 GIF 动画。

输出：animation/lbs_animation.gif
"""

import os
import warnings
import torch
import numpy as np
import smplx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.ticker import NullFormatter
import imageio

warnings.filterwarnings('ignore')

# ========================== 配置 ==========================
MODEL_PATH = '/Users/ruohanwang/Downloads/SMPL_NEUTRAL.pkl'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'animation')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32


def to_numpy(t):
    return t.detach().cpu().numpy()


def setup_ax(ax, verts_np, title='', elev=20, azim=-75):
    """设置 3D 坐标轴"""
    center = verts_np.mean(axis=0)
    max_radius = np.max(np.linalg.norm(verts_np - center, axis=1))
    ax.set_xlim(center[0] - max_radius * 1.2, center[0] + max_radius * 1.2)
    ax.set_ylim(center[1] - max_radius * 1.2, center[1] + max_radius * 1.2)
    ax.set_zlim(center[2] - max_radius * 1.2, center[2] + max_radius * 1.2)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_axis_off()
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.zaxis.set_major_formatter(NullFormatter())
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.set_box_aspect([1, 1, 1], zoom=0.85)
    return ax


def render_frame(vertices, faces, title='', elev=20, azim=-75, figsize=(8, 7)):
    """渲染单帧网格"""
    verts_np = to_numpy(vertices) if torch.is_tensor(vertices) else vertices
    if verts_np.ndim == 3:
        verts_np = verts_np[0]
    faces_np = to_numpy(faces) if torch.is_tensor(faces) else faces

    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')

    mesh = Poly3DCollection(
        verts_np[faces_np],
        facecolors='lightblue',
        alpha=0.95, edgecolor='none', linewidth=0, antialiased=True,
    )
    ax.add_collection3d(mesh)
    setup_ax(ax, verts_np, title=title, elev=elev, azim=azim)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return buf


def main():
    print("=" * 50)
    print("  LBS 蒙皮 — 姿态动画（选做）")
    print("=" * 50)

    # ── 加载模型 ──
    print("\n[加载 SMPL 模型...]")
    model = smplx.create(MODEL_PATH, model_type='smpl', gender='neutral', batch_size=1)
    model = model.to(DEVICE)

    v_template = model.v_template
    shapedirs = model.shapedirs
    posedirs = model.posedirs
    J_regressor = model.J_regressor
    parents = model.parents
    lbs_weights = model.lbs_weights
    faces = model.faces_tensor

    if v_template.dim() == 2:
        v_template = v_template.unsqueeze(0)

    # ── 固定 shape 参数 ──
    betas = torch.zeros(1, model.num_betas, dtype=DTYPE, device=DEVICE)
    betas[0, 0] = 1.0
    betas[0, 1] = 0.5

    # ── 从 smplx 引入 LBS 函数 ──
    from smplx.lbs import lbs as smplx_lbs

    # ── 逐帧生成 ──
    # 旋转右肩 (joint 17)：从 0 到 -1.8 rad (~ -103°)
    num_frames = 45
    angles = np.linspace(0, -1.8, num_frames)

    frames = []
    for i, angle in enumerate(angles):
        global_orient = torch.zeros(1, 3, dtype=DTYPE, device=DEVICE)
        global_orient[0] = torch.tensor([0.15, 0.0, 0.0])

        body_pose = torch.zeros(1, 69, dtype=DTYPE, device=DEVICE)
        rs_idx = 17 - 1
        body_pose[0, rs_idx*3:rs_idx*3+3] = torch.tensor([0.0, -0.2, angle])

        full_pose = torch.cat([global_orient, body_pose], dim=1)

        verts, J_transformed = smplx_lbs(
            betas, full_pose, v_template, shapedirs, posedirs,
            J_regressor, parents, lbs_weights, pose2rot=True,
        )

        progress = angle / (-1.8) * 100
        buf = render_frame(
            verts, faces,
            title=f'Right Shoulder Rotation: {angle*180/np.pi:.0f}°',
            elev=15, azim=-70,
        )
        frames.append(buf)
        print(f"  帧 {i+1}/{num_frames} ({progress:.0f}%)", end='\r')

    print("\n\n[保存 GIF...]")
    gif_path = os.path.join(OUTPUT_DIR, 'lbs_animation.gif')
    imageio.mimsave(gif_path, frames, fps=15, loop=0)
    file_size = os.path.getsize(gif_path) / 1024
    print(f"  ✓ 已保存 {gif_path} ({file_size:.0f} KB)")

    print("\n" + "=" * 50)
    print("  动画生成完成！")
    print("=" * 50)


if __name__ == '__main__':
    main()
