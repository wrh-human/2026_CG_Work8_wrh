# -*- coding: utf-8 -*-
"""
计算机图形学 实验八 —— LBS蒙皮
基于 SMPL 模型的 Linear Blend Skinning 完整可视化

本脚本实现：
  - Task 1: 加载 SMPL 模型并输出基础信息
  - Task 2: 可视化模板网格与蒙皮权重 (Stage a)
  - Task 3: 可视化形状校正与关节回归 (Stage b)
  - Task 4: 可视化姿态校正 B_P(θ)  (Stage c)
  - Task 5: 可视化完整 LBS 结果   (Stage d)
  - Task 6: 总对比图
  - Task 7: 手写 LBS 与官方前向结果一致性验证
"""

import os
import sys
import warnings
import torch
import numpy as np
import smplx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.ticker import NullFormatter
from typing import Optional, Tuple

warnings.filterwarnings('ignore')

# ========================== 配置 ==========================
MODEL_PATH = '/Users/ruohanwang/Downloads/SMPL_NEUTRAL.pkl'
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32

# 关节名称（SMPL 前 24 个身体关节）
JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1',
    'left_knee', 'right_knee', 'spine2',
    'left_ankle', 'right_ankle', 'spine3',
    'left_foot', 'right_foot', 'neck',
    'left_collar', 'right_collar', 'head',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hand', 'right_hand',
]

print(f"[INFO] Using device: {DEVICE}")

# ====================== 工具函数 ==========================

def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def blend_shapes(betas: torch.Tensor, shapedirs: torch.Tensor) -> torch.Tensor:
    """
    blend_shape = einsum('bl,mkl->bmk', [betas, shapedirs])
    返回 B x V x 3 的位移量
    """
    return torch.einsum('bl,mkl->bmk', [betas, shapedirs])


def vertices2joints(J_regressor: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    """
    从顶点回归关节：J[b,j,k] = sum_i J_regressor[j,i] * vertices[b,i,k]
    """
    return torch.einsum('ji,bik->bjk', [J_regressor, vertices])


def batch_rodrigues(rot_vecs: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """批量 Rodrigues 变换：轴角 -> 旋转矩阵。"""
    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)

    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    创建 4x4 变换矩阵。
    R: Bx3x3, t: Bx3x1 -> T: Bx4x4
    """
    import torch.nn.functional as F
    R_pad = F.pad(R, [0, 0, 0, 1])          # Bx4x3
    t_pad = F.pad(t, [0, 0, 0, 1], value=1) # Bx4x1
    return torch.cat([R_pad, t_pad], dim=2)  # Bx4x4


def batch_rigid_transform(
    rot_mats: torch.Tensor,
    joints: torch.Tensor,
    parents: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算运动学链上的刚性变换。
    完全匹配 smplx 的官方实现逻辑。

    Parameters
    ----------
    rot_mats : B x J x 3 x 3
    joints   : B x J x 3
    parents  : J (parents[0] = -1)

    Returns
    -------
    posed_joints : B x J x 3 (全局关节位置)
    rel_transforms : B x J x 4 x 4 (相对于静止姿态的混合变换)
    """
    import torch.nn.functional as F
    batch_size = rot_mats.shape[0]
    device, dtype = rot_mats.device, rot_mats.dtype
    num_joints = parents.shape[0]

    # 关节位置扩展为 BxJx3x1
    joints_unsq = joints.unsqueeze(dim=-1)

    # 相对关节位置（相对于父关节在 rest-pose 下的偏移）
    rel_joints = joints_unsq.clone()
    rel_joints[:, 1:] -= joints_unsq[:, parents[1:]]

    # 构建局部变换矩阵 T_local = [R | t_rel; 0 | 1]
    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1),
    ).reshape(-1, num_joints, 4, 4)

    # 沿运动学链累积（前向遍历）
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, num_joints):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)  # B x J x 4 x 4

    # 变换后关节位置就是每列最后一个元素（平移部分）
    posed_joints = transforms[:, :, :3, 3]

    # 计算相对变换（消除 rest-pose 中关节位移的贡献）
    joints_homogen = F.pad(joints_unsq, [0, 0, 0, 1])  # BxJx4x1
    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0]
    )

    return posed_joints, rel_transforms


def lbs_manual(
    betas: torch.Tensor,
    pose: torch.Tensor,
    v_template: torch.Tensor,
    shapedirs: torch.Tensor,
    posedirs: torch.Tensor,
    J_regressor: torch.Tensor,
    parents: torch.Tensor,
    lbs_weights: torch.Tensor,
) -> dict:
    """
    完整的手写 LBS 实现。

    返回包含所有中间量的字典：
        verts:          B x V x 3  最终蒙皮顶点
        J:              B x J x 3  形状变形后的关节位置
        v_shaped:       B x V x 3  形状变形后网格
        v_posed:        B x V x 3  姿态校正后网格
        J_transformed:  B x J x 3  变换后关节位置
        pose_offsets:   B x V x 3  姿态偏移量
        rot_mats:       B x J x 3 x 3  旋转矩阵
    """
    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype

    # ── (a) 模板网格 ──────────────────────────────
    # v_template 在外部设定

    # ── (b) 形状变形 ──────────────────────────────
    v_shaped = v_template + blend_shapes(betas, shapedirs)  # B x V x 3

    # 从变形后顶点回归关节
    J = vertices2joints(J_regressor, v_shaped)  # B x J x 3

    # ── (c) 姿态校正 ──────────────────────────────
    ident = torch.eye(3, dtype=dtype, device=device)

    # 轴角 -> 旋转矩阵
    rot_mats = batch_rodrigues(pose.reshape(-1, 3)).reshape(batch_size, -1, 3, 3)

    # pose_feature: 排除根节点（全局旋转）
    pose_feature = (rot_mats[:, 1:, :, :] - ident).reshape(batch_size, -1)
    # posedirs 形状: 207 x 20670 (已经转置)
    pose_offsets = torch.matmul(pose_feature, posedirs).reshape(batch_size, -1, 3)

    v_posed = v_shaped + pose_offsets  # B x V x 3

    # ── (d) 线性混合蒙皮 ──────────────────────────
    # 计算全局关节变换（返回相对变换）
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents)
    # A: B x J x 4 x 4 — 每个关节相对于 rest-pose 的变换

    num_joints = J_regressor.shape[0]

    # 蒙皮权重加权
    W = lbs_weights.unsqueeze(dim=0).expand(batch_size, -1, -1)  # B x V x J
    # A 展平为 B x J x 16
    T = torch.matmul(W, A.reshape(batch_size, num_joints, 16)).reshape(
        batch_size, -1, 4, 4
    )

    # 齐次坐标
    ones = torch.ones(batch_size, v_posed.shape[1], 1, dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, ones], dim=2)  # B x V x 4

    # 应用混合变换
    v_homo = torch.matmul(T, v_posed_homo.unsqueeze(dim=-1))  # B x V x 4 x 1
    verts = v_homo[:, :, :3, 0]  # B x V x 3

    return {
        'verts': verts,
        'J': J,
        'v_shaped': v_shaped,
        'v_posed': v_posed,
        'J_transformed': J_transformed,
        'pose_offsets': pose_offsets,
        'rot_mats': rot_mats,
    }


# ====================== 3D 渲染工具 ==========================

def setup_3d_ax(ax, vertices, elev=20, azim=-75, title='',
                title_size=14, axis_off=True):
    """
    设置 3D 坐标轴的基本属性。
    """
    verts_np = to_numpy(vertices) if torch.is_tensor(vertices) else vertices
    if verts_np.ndim == 3:
        verts_np = verts_np[0]

    center = verts_np.mean(axis=0)
    max_radius = np.max(np.linalg.norm(verts_np - center, axis=1))

    ax.set_xlim(center[0] - max_radius * 1.2, center[0] + max_radius * 1.2)
    ax.set_ylim(center[1] - max_radius * 1.2, center[1] + max_radius * 1.2)
    ax.set_zlim(center[2] - max_radius * 1.2, center[2] + max_radius * 1.2)

    # 翻转 Z 轴使向上为正
    ax.set_zticklabels([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=title_size, pad=10)

    if axis_off:
        ax.set_axis_off()
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.zaxis.set_major_formatter(NullFormatter())
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')
        ax.grid(False)

    ax.set_box_aspect([1, 1, 1], zoom=0.85)

    return ax


def _vertex_colors_to_face_colors(vertices, faces, vertex_colors, cmap_name='viridis',
                                   vmin=None, vmax=None):
    """
    将顶点颜色转换为面片颜色。
    vertex_colors 可以是:
      - 1D 标量数组 (V,) — 使用 cmap 映射
      - 2D RGB 数组 (V, 3)
      - 2D RGBA 数组 (V, 4)
    返回 (face_colors_rgba, colormap_norm) 其中 norm 用于 colorbar。
    """
    vc = to_numpy(vertex_colors) if torch.is_tensor(vertex_colors) else vertex_colors
    faces_np = to_numpy(faces) if torch.is_tensor(faces) else faces

    # 确保是 (V, C) 格式
    if vc.ndim == 1:
        vc_2d = vc[:, None]  # (V, 1)
    else:
        vc_2d = vc

    n_verts = vc_2d.shape[0]
    n_channels = vc_2d.shape[1]

    if n_channels == 1:
        # 标量：通过 colormap
        vmin_ = vmin if vmin is not None else float(vc_2d.min())
        vmax_ = vmax if vmax is not None else float(vc_2d.max())
        norm = Normalize(vmin=vmin_, vmax=vmax_)
        cmap = cm.get_cmap(cmap_name)
        # 每个面片的颜色 = 其三个顶点颜色的平均
        face_vals = vc_2d[faces_np].mean(axis=1)  # (F, 1)
        face_colors_rgba = cmap(norm(face_vals[:, 0]))
        return face_colors_rgba, norm

    elif n_channels in (3, 4):
        # RGB 或 RGBA
        vc_rgba = np.clip(vc_2d, 0, 1)
        if vc_rgba.max() > 1.0:
            vc_rgba = vc_rgba / 255.0
        if n_channels == 3:
            # 加 alpha 通道
            alpha_vals = np.ones((n_verts, 1))
            vc_rgba = np.concatenate([vc_rgba, alpha_vals], axis=1)
        # 插值到面片
        face_colors_rgba = vc_rgba[faces_np].mean(axis=1)
        return face_colors_rgba, None


def render_mesh(vertices, faces, vertex_colors=None,
                joints=None, joint_colors='red', joint_sizes=50,
                title='', elev=20, azim=-75,
                figsize=(10, 8), axis_off=True,
                show_colorbar=False, colorbar_label='',
                cmap_name='viridis', vmin=None, vmax=None,
                alpha=0.95):
    """
    使用 matplotlib 3D 渲染网格 + 关节。
    返回 (fig, ax)
    """
    verts_np = to_numpy(vertices) if torch.is_tensor(vertices) else vertices
    faces_np = to_numpy(faces) if torch.is_tensor(faces) else faces
    if verts_np.ndim == 3:
        verts_np = verts_np[0]

    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')

    # 顶点 -> 面片颜色
    norm_cb = None
    if vertex_colors is not None:
        face_colors_rgba, norm_cb = _vertex_colors_to_face_colors(
            verts_np, faces_np, vertex_colors,
            cmap_name=cmap_name, vmin=vmin, vmax=vmax,
        )
    else:
        face_colors_rgba = 'lightblue'

    mesh = Poly3DCollection(
        verts_np[faces_np],
        facecolors=face_colors_rgba,
        alpha=alpha,
        edgecolor='none',
        linewidth=0,
        antialiased=True,
    )
    ax.add_collection3d(mesh)

    # 关节
    if joints is not None:
        jnts_np = to_numpy(joints) if torch.is_tensor(joints) else joints
        if jnts_np.ndim == 3:
            jnts_np = jnts_np[0]
        ax.scatter(
            jnts_np[:, 0], jnts_np[:, 1], jnts_np[:, 2],
            c=joint_colors, s=joint_sizes, marker='o',
            edgecolors='black', linewidths=0.5,
            alpha=0.9, zorder=10,
        )

    # 颜色条
    if show_colorbar and norm_cb is not None:
        sm = cm.ScalarMappable(norm=norm_cb, cmap=cmap_name)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=15, pad=0.05)
        cbar.set_label(colorbar_label, fontsize=10)

    setup_3d_ax(ax, verts_np, title=title, axis_off=axis_off)

    return fig, ax


def render_mesh_on_ax(ax, vertices, faces, vertex_colors=None,
                       joints=None, joint_colors='red', title='',
                       cmap_name='viridis', alpha=0.95,
                       vmin=None, vmax=None):
    """
    在已有的 subplot ax 上绘制网格。
    """
    verts_np = to_numpy(vertices) if torch.is_tensor(vertices) else vertices
    faces_np = to_numpy(faces) if torch.is_tensor(faces) else faces
    if verts_np.ndim == 3:
        verts_np = verts_np[0]

    if vertex_colors is not None:
        face_colors_rgba, _ = _vertex_colors_to_face_colors(
            verts_np, faces_np, vertex_colors,
            cmap_name=cmap_name, vmin=vmin, vmax=vmax,
        )
    else:
        face_colors_rgba = 'lightblue'

    mesh = Poly3DCollection(
        verts_np[faces_np],
        facecolors=face_colors_rgba,
        alpha=alpha,
        edgecolor='none',
        linewidth=0,
        antialiased=True,
    )
    ax.add_collection3d(mesh)

    if joints is not None:
        jnts_np = to_numpy(joints) if torch.is_tensor(joints) else joints
        if jnts_np.ndim == 3:
            jnts_np = jnts_np[0]
        ax.scatter(
            jnts_np[:, 0], jnts_np[:, 1], jnts_np[:, 2],
            c=joint_colors, s=50, marker='o',
            edgecolors='black', linewidths=0.5,
            alpha=0.9, zorder=10,
        )

    setup_3d_ax(ax, verts_np, title=title, axis_off=True)
    return ax


# ====================== 主程序 ==========================

def main():
    print("=" * 60)
    print("      计算机图形学 实验八 —— LBS 蒙皮可视化")
    print("=" * 60)

    # ╔══════════════════════════════════════════════════╗
    # ║   Task 1: 加载 SMPL 模型                        ║
    # ╚══════════════════════════════════════════════════╝
    print("\n[Task 1] 加载 SMPL 模型...")
    model = smplx.create(
        MODEL_PATH,
        model_type='smpl',
        gender='neutral',
        batch_size=1,
    )
    model = model.to(DEVICE)

    # 提取关键缓冲
    v_template = model.v_template               # V x 3
    shapedirs = model.shapedirs                 # V x 3 x num_betas
    posedirs = model.posedirs                   # 207 x 20670 (已转置)
    J_regressor = model.J_regressor             # J x V
    parents = model.parents                     # J
    lbs_weights = model.lbs_weights             # V x J
    faces = model.faces_tensor                  # F x 3

    num_verts = model.get_num_verts()
    num_faces = model.get_num_faces()
    num_joints = J_regressor.shape[0]
    num_betas = model.num_betas

    print(f"  顶点数:          {num_verts}")
    print(f"  面片数:          {num_faces}")
    print(f"  关节数:          {num_joints}")
    print(f"  Betas 维度:      {num_betas}")
    print(f"  蒙皮权重形状:    {tuple(lbs_weights.shape)}")

    # 检查 parents
    print(f"  运动学树 parents: {to_numpy(parents).tolist()}")

    # 确保 v_template 是 batch 格式
    if v_template.dim() == 2:
        v_template = v_template.unsqueeze(0)    # 1 x V x 3

    # ================ 设置参数 =================
    # 形状参数：让前几个 betas 非零
    betas = torch.zeros(1, num_betas, dtype=DTYPE, device=DEVICE)
    betas[0, 0] = 1.5
    betas[0, 1] = 0.8
    betas[0, 2] = -0.6
    betas[0, 3] = 0.4
    betas[0, 4] = 0.3

    # 姿态参数：右手抬起 + 肘部弯曲
    global_orient = torch.zeros(1, 3, dtype=DTYPE, device=DEVICE)
    global_orient[0] = torch.tensor([0.2, 0.0, 0.0])

    body_pose = torch.zeros(1, 69, dtype=DTYPE, device=DEVICE)

    # joint 17 = right_shoulder -> body_pose index 16 (body_pose 起始于 joint 1)
    rs_idx = 17 - 1
    body_pose[0, rs_idx*3:rs_idx*3+3] = torch.tensor([0.0, -0.3, -1.2])

    # joint 19 = right_elbow -> body_pose index 18
    re_idx = 19 - 1
    body_pose[0, re_idx*3:re_idx*3+3] = torch.tensor([0.0, 0.3, -0.8])

    # joint 18 = left_elbow
    le_idx = 18 - 1
    body_pose[0, le_idx*3:le_idx*3+3] = torch.tensor([0.0, 0.0, 0.3])

    full_pose = torch.cat([global_orient, body_pose], dim=1)

    print(f"\n  形状参数 β: {to_numpy(betas[0]).tolist()}")
    print(f"  全局旋转:    {to_numpy(global_orient[0]).tolist()}")
    print(f"  右肩旋转:    {to_numpy(body_pose[0, rs_idx*3:rs_idx*3+3]).tolist()}")
    print(f"  右肘旋转:    {to_numpy(body_pose[0, re_idx*3:re_idx*3+3]).tolist()}")

    # ╔══════════════════════════════════════════════════╗
    # ║   Task 2: Stage (a) - 模板 + 权重               ║
    # ╚══════════════════════════════════════════════════╝
    print("\n[Task 2] 可视化模板网格与蒙皮权重...")

    joint_idx_a = 17  # right_shoulder
    joint_weight = lbs_weights[:, joint_idx_a]

    fig_a, _ = render_mesh(
        v_template, faces,
        vertex_colors=joint_weight,
        title=f'Stage (a): Template + {JOINT_NAMES[joint_idx_a]} Weight',
        cmap_name='plasma',
        show_colorbar=True,
        colorbar_label=f'{JOINT_NAMES[joint_idx_a]} weight',
        elev=15, azim=-75,
    )
    fig_a.savefig(os.path.join(OUTPUT_DIR, 'stage_a_template_weights.png'),
                  dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig_a)
    print("  ✓ 已保存 stage_a_template_weights.png")

    # 全关节主导权重图
    dominant_joints = torch.argmax(lbs_weights, dim=1)
    max_weights = torch.max(lbs_weights, dim=1).values

    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
    num_colors = min(num_joints, 20)
    tab20 = cm.get_cmap('tab20', num_colors)
    dom_colors = np.zeros((num_verts, 3))
    for v in range(num_verts):
        c = tab20(dominant_joints[v].item() % num_colors)[:3]
        hsv = rgb_to_hsv(np.array(c[:3]))
        hsv[2] = 0.4 + 0.6 * max_weights[v].item()
        dom_colors[v] = hsv_to_rgb(hsv)

    fig_aj, _ = render_mesh(
        v_template, faces,
        vertex_colors=dom_colors,
        title='All Joints: Dominant Weight Map',
        elev=15, azim=-75,
    )
    fig_aj.savefig(os.path.join(OUTPUT_DIR, 'all_joint_weights.png'),
                   dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig_aj)
    print("  ✓ 已保存 all_joint_weights.png")

    # ╔══════════════════════════════════════════════════╗
    # ║   Task 3: Stage (b) - 形状 + 关节               ║
    # ╚══════════════════════════════════════════════════╝
    print("\n[Task 3] 可视化形状校正与关节回归...")

    v_shaped = v_template + blend_shapes(betas, shapedirs)
    J = vertices2joints(J_regressor, v_shaped)

    fig_b, _ = render_mesh(
        v_shaped, faces,
        joints=J, joint_colors='red',
        title='Stage (b): Shape + Joints',
        elev=15, azim=-75,
    )
    fig_b.savefig(os.path.join(OUTPUT_DIR, 'stage_b_shaped_joints.png'),
                  dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig_b)
    print("  ✓ 已保存 stage_b_shaped_joints.png")

    # ╔══════════════════════════════════════════════════╗
    # ║   Task 4: Stage (c) - 姿态校正                   ║
    # ╚══════════════════════════════════════════════════╝
    print("\n[Task 4] 可视化姿态校正...")

    ident = torch.eye(3, dtype=DTYPE, device=DEVICE)
    rot_mats = batch_rodrigues(full_pose.reshape(-1, 3)).reshape(1, -1, 3, 3)
    pose_feature = (rot_mats[:, 1:, :, :] - ident).reshape(1, -1)
    pose_offsets = torch.matmul(pose_feature, posedirs).reshape(1, -1, 3)

    v_posed = v_shaped + pose_offsets

    pose_offset_magnitudes = torch.norm(pose_offsets[0], dim=1)
    print(f"  姿态偏移范围: {pose_offset_magnitudes.min():.6f} ~ {pose_offset_magnitudes.max():.6f}")

    fig_c, _ = render_mesh(
        v_posed, faces,
        vertex_colors=pose_offset_magnitudes,
        title='Stage (c): Pose Corrective Offsets',
        cmap_name='coolwarm',
        show_colorbar=True,
        colorbar_label='Offset magnitude',
        elev=15, azim=-75,
    )
    fig_c.savefig(os.path.join(OUTPUT_DIR, 'stage_c_pose_offsets.png'),
                  dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig_c)
    print("  ✓ 已保存 stage_c_pose_offsets.png")

    # ╔══════════════════════════════════════════════════╗
    # ║   Task 5 & 7: Stage (d) LBS + 手写验证          ║
    # ╚══════════════════════════════════════════════════╝
    print("\n[Task 5] 可视化完整 LBS 结果...")
    print("[Task 7] 手写 LBS 验证...")

    # 手写 LBS
    result = lbs_manual(
        betas, full_pose, v_template, shapedirs, posedirs,
        J_regressor, parents, lbs_weights,
    )

    # 官方前向
    output = model(
        betas=betas,
        body_pose=body_pose,
        global_orient=global_orient,
        return_verts=True,
    )
    verts_official = output.vertices
    joints_official = output.joints

    # 可视化 LBS 结果
    fig_d, _ = render_mesh(
        result['verts'], faces,
        joints=result['J_transformed'], joint_colors='red',
        title='Stage (d): Final LBS Result',
        elev=15, azim=-75,
    )
    fig_d.savefig(os.path.join(OUTPUT_DIR, 'stage_d_lbs_result.png'),
                  dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig_d)
    print("  ✓ 已保存 stage_d_lbs_result.png")

    # ── 误差计算 ──
    error = result['verts'] - verts_official
    mean_abs_error = error.abs().mean().item()
    max_abs_error = error.abs().max().item()
    mean_sq_error = (error ** 2).mean().item()
    rmse = torch.sqrt((error ** 2).mean()).item()

    print(f"\n  ┌─── 误差验证 ──────────────────────────┐")
    print(f"  │  平均绝对误差 (MAE):   {mean_abs_error:.3e}")
    print(f"  │  最大绝对误差 (MaxAE): {max_abs_error:.3e}")
    print(f"  │  MSE:                  {mean_sq_error:.3e}")
    print(f"  │  RMSE:                 {rmse:.3e}")
    print(f"  └────────────────────────────────────────┘")

    if mean_abs_error < 1e-5 and max_abs_error < 1e-4:
        print("  ✓ 手写 LBS 完全匹配官方结果！")
    elif mean_abs_error < 1e-3:
        print("  ✓ 手写 LBS 与官方结果高度一致（误差在可接受范围）")
    else:
        print(f"  ⚠ 误差偏大 (MAE={mean_abs_error:.3e})，请检查实现")

    # ╔══════════════════════════════════════════════════╗
    # ║   Task 6: 对比图                                ║
    # ╚══════════════════════════════════════════════════╝
    print("\n[Task 6] 生成总对比图...")

    fig_grid = plt.figure(figsize=(18, 16), facecolor='white')

    # (a) 模板 + 权重
    ax1 = fig_grid.add_subplot(2, 2, 1, projection='3d', facecolor='white')
    render_mesh_on_ax(
        ax1, v_template, faces,
        vertex_colors=lbs_weights[:, joint_idx_a],
        title=f'(a) Template + {JOINT_NAMES[joint_idx_a]} Weight',
        cmap_name='plasma',
    )

    # (b) 形状 + 关节
    ax2 = fig_grid.add_subplot(2, 2, 2, projection='3d', facecolor='white')
    render_mesh_on_ax(
        ax2, v_shaped, faces,
        joints=J, joint_colors='red',
        title='(b) Shape + Joints',
    )

    # (c) 姿态偏移
    ax3 = fig_grid.add_subplot(2, 2, 3, projection='3d', facecolor='white')
    render_mesh_on_ax(
        ax3, v_posed, faces,
        vertex_colors=pose_offset_magnitudes,
        cmap_name='coolwarm',
        title='(c) Pose Offsets',
    )

    # (d) LBS 结果
    ax4 = fig_grid.add_subplot(2, 2, 4, projection='3d', facecolor='white')
    render_mesh_on_ax(
        ax4, result['verts'], faces,
        joints=result['J_transformed'], joint_colors='red',
        title='(d) Final LBS',
    )

    fig_grid.suptitle(
        'SMPL LBS Pipeline — Four Stages',
        fontsize=18, fontweight='bold', y=0.98,
    )
    fig_grid.savefig(os.path.join(OUTPUT_DIR, 'comparison_grid.png'),
                     dpi=150, bbox_inches='tight', pad_inches=0.2)
    plt.close(fig_grid)
    print("  ✓ 已保存 comparison_grid.png")

    # ╔══════════════════════════════════════════════════╗
    # ║   Error Map                                     ║
    # ╚══════════════════════════════════════════════════╝
    fig_err, ax_err = plt.subplots(figsize=(10, 8), subplot_kw={'projection': '3d'}, facecolor='white')
    error_per_vertex = error.abs().mean(dim=2)[0]
    render_mesh_on_ax(
        ax_err, result['verts'], faces,
        vertex_colors=error_per_vertex,
        cmap_name='hot',
        title='Error: Hand-written vs Official LBS',
    )
    fig_err.savefig(os.path.join(OUTPUT_DIR, 'error_map.png'),
                    dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig_err)
    print("  ✓ 已保存 error_map.png")

    # ╔══════════════════════════════════════════════════╗
    # ║   Summary                                       ║
    # ╚══════════════════════════════════════════════════╝
    summary_path = os.path.join(OUTPUT_DIR, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + '\n')
        f.write("  计算机图形学 实验八 —— LBS 蒙皮  Summary\n")
        f.write("=" * 60 + '\n\n')

        f.write("── Task 1: 模型基础信息 ──\n")
        f.write(f"  顶点数:       {num_verts}\n")
        f.write(f"  面片数:       {num_faces}\n")
        f.write(f"  关节数:       {num_joints}\n")
        f.write(f"  Betas 维度:   {num_betas}\n")
        f.write(f"  蒙皮权重:     {list(lbs_weights.shape)}\n\n")

        f.write("── 所用参数 ──\n")
        f.write(f"  betas: {to_numpy(betas[0]).tolist()}\n")
        f.write(f"  global_orient: {to_numpy(global_orient[0]).tolist()}\n")
        f.write(f"  right_shoulder: {to_numpy(body_pose[0, rs_idx*3:rs_idx*3+3]).tolist()}\n")
        f.write(f"  right_elbow:    {to_numpy(body_pose[0, re_idx*3:re_idx*3+3]).tolist()}\n\n")

        f.write("── Task 7: 手写 LBS 误差 ──\n")
        f.write(f"  MAE:   {mean_abs_error:.3e}\n")
        f.write(f"  MaxAE: {max_abs_error:.3e}\n")
        f.write(f"  MSE:   {mean_sq_error:.3e}\n")
        f.write(f"  RMSE:  {rmse:.3e}\n")

        if mean_abs_error < 1e-5 and max_abs_error < 1e-4:
            f.write("  结论: 手写 LBS 与官方结果完全一致 ✓\n")
        elif mean_abs_error < 1e-3:
            f.write("  结论: 手写 LBS 与官方结果高度一致 ✓\n")
        else:
            f.write("  结论: 手写 LBS 与官方结果存在差异，请检查 ⚠\n")

        f.write("\n── 输出文件 ──\n")
        for fname in ['stage_a_template_weights.png', 'all_joint_weights.png',
                      'stage_b_shaped_joints.png', 'stage_c_pose_offsets.png',
                      'stage_d_lbs_result.png', 'comparison_grid.png',
                      'error_map.png', 'summary.txt']:
            fpath = os.path.join(OUTPUT_DIR, fname)
            if os.path.exists(fpath):
                size = os.path.getsize(fpath) / 1024
                f.write(f"  ✓ {fname}  ({size:.1f} KB)\n")
            else:
                f.write(f"  ✗ {fname}  (未生成)\n")

        f.write("\n" + "=" * 60 + '\n')

    print(f"\n  ✓ 已保存 summary.txt")
    print("\n" + "=" * 60)
    print("  所有任务完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
