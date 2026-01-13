"""
磁控铰链机器人静态仿真器
Static Simulator for Magnetic Hinge-Controlled Robot
作者：AI Assistant
功能：计算6节铰链式机器人在匀强磁场中的静态平衡姿态，优化0/180°磁矩配置
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Rectangle
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# 1. 机器人物理模型类
# ============================================================================

class MagneticHingeRobot:
    """
    磁控铰链机器人物理模型
    6节连杆，5个铰链关节，每个连杆具有轴向磁矩（0°或180°）
    """

    def __init__(self,
                 link_length_mm=1.5,
                 width_mm=0.4,
                 height_mm=0.4,
                 magnetization_T=1.2,
                 density_kg_m3=2000.0,
                 max_angle_deg=70.0,
                 joint_stiffness=1e3):
        """
        初始化机器人参数

        参数：
        link_length_mm: 连杆长度(mm)
        width_mm: 连杆宽度(mm)
        height_mm: 连杆高度(mm)
        magnetization_T: 磁化强度(T)
        density_kg_m3: 材料密度(kg/m³)
        max_angle_deg: 最大关节角度(度)
        joint_stiffness: 关节刚度(N·m/rad)
        """
        # 机器人结构参数
        self.N_links = 6  # 连杆数量
        self.N_joints = self.N_links - 1  # 关节数量
        self.N_dof = 3 + self.N_joints  # 自由度数量: [x, y, θ_base, θ1, ..., θ5]

        # 几何参数（转换为国际单位）
        self.L = link_length_mm * 1e-3  # 连杆长度(m)
        self.w = width_mm * 1e-3  # 连杆宽度(m)
        self.h = height_mm * 1e-3  # 连杆高度(m)

        # 磁矩计算
        mu0 = 4 * np.pi * 1e-7  # 真空磁导率
        M = magnetization_T / mu0  # 磁化强度(A/m)
        vol = self.L * self.w * self.h  # 体积(m³)
        self.m_mag = M * vol  # 磁矩大小(A·m²)

        # 质量参数（仅用于重力计算）
        self.density = density_kg_m3
        self.mass = self.density * vol  # 每个连杆质量(kg)

        # 关节约束
        self.max_angle = np.deg2rad(max_angle_deg)  # 最大关节角(rad)
        self.k_joint = joint_stiffness  # 关节刚度(N·m/rad)

        # 物理常数
        self.g = 9.81  # 重力加速度(m/s²)

        # 缩放因子
        self.scale_factor = 1e3  # 米到毫米的转换

        # 地面参数
        self.ground_height = 0.0  # 地面高度
        self.ground_stiffness = 1e6  # 地面接触刚度(N/m)

        print(f"机器人初始化完成:")
        print(f"  连杆数: {self.N_links}, 关节数: {self.N_joints}, 自由度: {self.N_dof}")
        print(f"  连杆尺寸: {link_length_mm}×{width_mm}×{height_mm} mm")
        print(f"  磁矩大小: {self.m_mag:.2e} A·m²")
        print(f"  最大关节角: {max_angle_deg}°")

    def forward_kinematics(self, q):
        """
        正向运动学计算

        参数：
        q: 状态向量 [x0, y0, φ0, θ1, θ2, θ3, θ4, θ5]

        返回：
        centers: 连杆质心位置 (N_links, 2) [m]
        joints: 关节位置 (N_links+1, 2) [m]
        abs_angles: 绝对角度 (N_links) [rad]
        """
        # 提取状态变量
        x0, y0, phi0 = q[0], q[1], q[2]
        joint_angles = q[3:]  # 相对关节角度

        # 计算绝对角度（累积和）
        abs_angles = np.cumsum(np.concatenate(([phi0], joint_angles)))

        # 初始化位置数组
        joints = np.zeros((self.N_links + 1, 2))
        centers = np.zeros((self.N_links, 2))

        # 基座位置
        joints[0] = np.array([x0, y0])

        # 计算每个连杆的位置
        for i in range(self.N_links):
            # 当前连杆的方向向量
            dir_vec = np.array([np.cos(abs_angles[i]), np.sin(abs_angles[i])])

            # 下一个关节位置
            joints[i + 1] = joints[i] + self.L * dir_vec

            # 连杆质心位置（中点）
            centers[i] = joints[i] + 0.5 * self.L * dir_vec

        return centers, joints, abs_angles

    def compute_potential_energy(self, q, B_field, mpc_config, include_gravity=False):
        """
        计算总势能

        参数：
        q: 状态向量
        B_field: 磁场向量 [Bx, By] (T)
        mpc_config: 磁矩配置 [α0, α1, ..., α5] (0或π)
        include_gravity: 是否包含重力

        返回：
        总势能 (J)
        """
        # 运动学计算
        centers, joints, abs_angles = self.forward_kinematics(q)

        total_energy = 0.0

        # 1. 磁势能: U_mag = -Σ (m_i · B)
        for i in range(self.N_links):
            # 全局坐标系中的磁矩方向
            m_angle_global = abs_angles[i] + mpc_config[i]
            m_vector = self.m_mag * np.array([np.cos(m_angle_global), np.sin(m_angle_global)])

            # 磁势能
            total_energy += -np.dot(m_vector, B_field)

        # 2. 关节约束能（惩罚项）
        joint_angles = q[3:]
        for angle in joint_angles:
            if angle > self.max_angle:
                total_energy += 0.5 * self.k_joint * (angle - self.max_angle) ** 2
            elif angle < -self.max_angle:
                total_energy += 0.5 * self.k_joint * (angle + self.max_angle) ** 2

        # 3. 重力势能（可选）
        if include_gravity:
            for i in range(self.N_links):
                # U_grav = m * g * y
                total_energy += self.mass * self.g * centers[i, 1]

        # 4. 地面接触惩罚（防止穿透地面）
        for i in range(self.N_links + 1):
            y = joints[i, 1]
            if y < self.ground_height:
                penetration = self.ground_height - y
                total_energy += 0.5 * self.ground_stiffness * penetration ** 2

        return total_energy

    def find_equilibrium(self, B_field, mpc_config,
                         include_gravity=False,
                         initial_guess=None,
                         fixed_base=False,
                         base_position=None):
        """
        寻找静态平衡姿态（能量最小化）

        参数：
        B_field: 磁场向量 (T)
        mpc_config: 磁矩配置
        include_gravity: 是否包含重力
        initial_guess: 初始猜测
        fixed_base: 是否固定基座
        base_position: 固定基座的位置 [x, y, angle]

        返回：
        q_eq: 平衡状态
        """
        # 默认初始猜测：伸直状态
        if initial_guess is None:
            initial_guess = np.zeros(self.N_dof)
            initial_guess[1] = self.h / 2  # 初始高度为厚度一半
            # 微小随机扰动避免对称性
            initial_guess[3:] = np.random.uniform(-0.001, 0.001, self.N_joints)

        # 定义优化边界
        bounds = []

        # 基座位置和角度
        if fixed_base and base_position is not None:
            # 固定基座
            bounds.append((base_position[0], base_position[0]))  # x
            bounds.append((base_position[1], base_position[1]))  # y
            bounds.append((base_position[2], base_position[2]))  # φ0
        else:
            # 自由基座
            bounds.append((-10 * self.L, 10 * self.L))  # x
            bounds.append((0, 10 * self.L))  # y（不能低于地面）
            bounds.append((-np.pi, np.pi))  # φ0

        # 关节角度约束
        for _ in range(self.N_joints):
            bounds.append((-self.max_angle, self.max_angle))

        # 目标函数
        def objective(q):
            return self.compute_potential_energy(q, B_field, mpc_config, include_gravity)

        # 优化求解
        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 1000,
                'ftol': 1e-8,
                'disp': False
            }
        )

        if not result.success:
            print(f"优化警告: {result.message}")

        return result.x


# ============================================================================
# 2. 静态仿真器类
# ============================================================================

class StaticSimulator:
    """
    静态仿真器：计算并分析机器人的静态性能指标
    """

    def __init__(self, robot=None, B_magnitude=0.02):
        """
        初始化仿真器

        参数：
        robot: 机器人实例
        B_magnitude: 默认磁场大小(T)
        """
        if robot is None:
            self.robot = MagneticHingeRobot()
        else:
            self.robot = robot

        self.B_magnitude = B_magnitude

        # 常见的磁矩配置（0°或180°）
        self.common_configs = {
            'alternating': [0, np.pi, 0, np.pi, 0, np.pi],
            'uniform_0': [0, 0, 0, 0, 0, 0],
            'uniform_180': [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi],
            'front_back': [0, 0, 0, np.pi, np.pi, np.pi],
            'back_front': [np.pi, np.pi, np.pi, 0, 0, 0],
            'gradient': [0, 0, np.pi, np.pi, 0, 0],
            'middle_reverse': [0, 0, np.pi, 0, 0, 0],
        }

        # 性能权重
        self.weights = {
            'bending': 0.4,  # 弯曲能力
            'isotropy': 0.3,  # 各向同性
            'compactness': 0.2,  # 紧凑度
            'symmetry': 0.1  # 对称性
        }

        print(f"静态仿真器初始化完成")
        print(f"默认磁场强度: {B_magnitude * 1000:.1f} mT")

    def compute_static_metrics(self, q, mpc_config):
        """
        计算静态性能指标

        参数：
        q: 状态向量
        mpc_config: 磁矩配置

        返回：
        metrics: 指标字典
        """
        # 运动学计算
        centers, joints, abs_angles = self.robot.forward_kinematics(q)

        # 转换为毫米
        joints_mm = joints * self.robot.scale_factor
        centers_mm = centers * self.robot.scale_factor

        # 关节角度（度）
        joint_angles_rad = q[3:]
        joint_angles_deg = np.rad2deg(joint_angles_rad)

        # 1. 抬升高度：末端相对于最低点的高度
        min_y = np.min(joints_mm[:, 1])
        end_y = joints_mm[-1, 1]
        lift_height = max(0, end_y - min_y)  # 确保非负

        # 2. 弯曲幅度指标
        max_joint_angle = np.max(np.abs(joint_angles_deg))  # 最大关节角度
        mean_abs_joint_angle = np.mean(np.abs(joint_angles_deg))  # 平均绝对角度
        bending_energy = np.sum(joint_angles_rad ** 2)  # 弯曲能量

        # 3. 形状紧凑度：首尾距离与总长度之比
        total_length = self.robot.N_links * self.robot.L * self.robot.scale_factor
        head_tail_distance = np.linalg.norm(joints_mm[-1] - joints_mm[0])
        compactness = head_tail_distance / total_length

        # 4. 弯曲对称性
        half = self.robot.N_joints // 2
        front_bending = np.sum(np.abs(joint_angles_rad[:half]))
        rear_bending = np.sum(np.abs(joint_angles_rad[half:]))
        total_bending = front_bending + rear_bending
        symmetry = 1.0 if total_bending == 0 else 1.0 - np.abs(front_bending - rear_bending) / total_bending

        # 5. 弯曲模式识别
        signs = np.sign(joint_angles_rad)
        sign_changes = np.sum(np.abs(np.diff(signs)))
        if sign_changes >= 2:
            bending_mode = "S形"
        elif sign_changes == 1:
            bending_mode = "C形"
        else:
            bending_mode = "直线" if max_joint_angle < 1.0 else "同向弯曲"

        return {
            # 抬升性能
            'lift_height_mm': lift_height,
            'end_height_mm': end_y,
            'min_height_mm': min_y,

            # 弯曲性能
            'max_joint_angle_deg': max_joint_angle,
            'mean_joint_angle_deg': mean_abs_joint_angle,
            'bending_energy': bending_energy,
            'compactness': compactness,
            'symmetry': symmetry,
            'bending_mode': bending_mode,
            'sign_changes': sign_changes,

            # 详细数据
            'joint_angles_deg': joint_angles_deg,
            'joints_pos_mm': joints_mm,
            'centers_pos_mm': centers_mm,
            'abs_angles_deg': np.rad2deg(abs_angles)
        }

    def evaluate_magnet_config(self, mpc_config,
                               B_magnitude=None,
                               test_directions=8,
                               include_gravity_for_lift=True):
        """
        综合评价磁矩配置

        参数：
        mpc_config: 磁矩配置
        B_magnitude: 磁场大小(T)
        test_directions: 测试方向数量
        include_gravity_for_lift: 计算抬升时是否包含重力

        返回：
        results: 评估结果字典
        """
        if B_magnitude is None:
            B_magnitude = self.B_magnitude

        print(f"评估磁矩配置: {self.config_to_string(mpc_config)}")

        # 存储不同方向的结果
        bending_results = []
        lift_height = 0.0

        # 测试多个磁场方向
        for i in range(test_directions):
            phi = i * 2 * np.pi / test_directions
            B_field = B_magnitude * np.array([np.cos(phi), np.sin(phi)])

            # 寻找平衡（无重力）
            q_eq = self.robot.find_equilibrium(
                B_field, mpc_config,
                include_gravity=False
            )

            # 计算指标
            metrics = self.compute_static_metrics(q_eq, mpc_config)

            bending_results.append({
                'direction_deg': np.rad2deg(phi),
                'B_field': B_field,
                'state': q_eq,
                'max_angle': metrics['max_joint_angle_deg'],
                'bending_energy': metrics['bending_energy'],
                'compactness': metrics['compactness'],
                'symmetry': metrics['symmetry'],
                'bending_mode': metrics['bending_mode']
            })

            # 如果是垂直向上方向，计算抬升高度（有重力）
            if np.abs(phi - np.pi / 2) < 0.01:  # 垂直向上
                q_eq_lift = self.robot.find_equilibrium(
                    B_field, mpc_config,
                    include_gravity=include_gravity_for_lift
                )
                metrics_lift = self.compute_static_metrics(q_eq_lift, mpc_config)
                lift_height = metrics_lift['lift_height_mm']

        # 统计性能
        max_angles = [r['max_angle'] for r in bending_results]
        bending_energies = [r['bending_energy'] for r in bending_results]

        mean_max_angle = np.mean(max_angles)
        std_max_angle = np.std(max_angles)
        max_max_angle = np.max(max_angles)

        mean_bending_energy = np.mean(bending_energies)
        std_bending_energy = np.std(bending_energies)

        # 方向各向同性
        if mean_max_angle > 0:
            isotropy_angle = 1.0 - min(std_max_angle / mean_max_angle, 1.0)
        else:
            isotropy_angle = 0.0

        if mean_bending_energy > 0:
            isotropy_energy = 1.0 - min(std_bending_energy / mean_bending_energy, 1.0)
        else:
            isotropy_energy = 0.0

        isotropy = (isotropy_angle + isotropy_energy) / 2

        # 最佳弯曲方向
        best_bending_idx = np.argmax(max_angles)
        best_bending_dir = bending_results[best_bending_idx]['direction_deg']

        # 平均紧凑度
        avg_compactness = np.mean([r['compactness'] for r in bending_results])

        # 使用最佳方向的姿态计算对称性
        B_best = bending_results[best_bending_idx]['B_field']
        q_best = self.robot.find_equilibrium(B_best, mpc_config, include_gravity=False)
        metrics_best = self.compute_static_metrics(q_best, mpc_config)
        symmetry = metrics_best['symmetry']

        # 归一化评分
        max_angle_norm = min(mean_max_angle / 70.0, 1.0)  # 相对于70°归一化
        isotropy_norm = isotropy
        compactness_norm = avg_compactness  # 本身就在0-1之间
        symmetry_norm = symmetry  # 本身就在0-1之间

        # 综合评分
        composite_score = (
                self.weights['bending'] * max_angle_norm +
                self.weights['isotropy'] * isotropy_norm +
                self.weights['compactness'] * compactness_norm +
                self.weights['symmetry'] * symmetry_norm
        )

        # 整理结果
        results = {
            'mpc_config': mpc_config,
            'config_string': self.config_to_string(mpc_config),

            # 性能指标
            'mean_max_angle': mean_max_angle,
            'max_max_angle': max_max_angle,
            'isotropy': isotropy,
            'lift_height_mm': lift_height,
            'best_bending_direction': best_bending_dir,
            'composite_score': composite_score,

            # 归一化评分
            'normalized_scores': {
                'bending': max_angle_norm,
                'isotropy': isotropy_norm,
                'compactness': compactness_norm,
                'symmetry': symmetry_norm
            },

            # 详细数据
            'detailed_results': bending_results,
            'best_state': q_best,
            'best_metrics': metrics_best
        }

        return results

    def config_to_string(self, mpc_config):
        """将磁矩配置转换为字符串表示"""
        angles_deg = np.rad2deg(mpc_config)
        return '[' + ', '.join([f'{a:.0f}°' for a in angles_deg]) + ']'

    def evaluate_all_configs(self, configs=None, B_magnitude=None):
        """
        评估所有常见磁矩配置

        参数：
        configs: 要评估的配置字典
        B_magnitude: 磁场大小

        返回：
        results: 所有配置的评估结果
        ranked: 按综合评分排序的配置
        """
        if configs is None:
            configs = self.common_configs

        if B_magnitude is None:
            B_magnitude = self.B_magnitude

        print("=" * 60)
        print("开始评估所有磁矩配置")
        print("=" * 60)

        results = {}

        for name, config in configs.items():
            print(f"\n评估配置: {name:15s} {self.config_to_string(config)}")

            result = self.evaluate_magnet_config(
                config,
                B_magnitude=B_magnitude,
                test_directions=8,
                include_gravity_for_lift=True
            )

            results[name] = result

            print(f"  平均最大关节角: {result['mean_max_angle']:6.1f}°")
            print(f"  方向各向同性: {result['isotropy']:6.3f}")
            print(f"  抬升高度: {result['lift_height_mm']:6.2f} mm")
            print(f"  综合评分: {result['composite_score']:6.3f}")

        # 按综合评分排序
        ranked = sorted(results.items(), key=lambda x: x[1]['composite_score'], reverse=True)

        # 打印排名
        print("\n" + "=" * 60)
        print("性能排名")
        print("=" * 60)

        for i, (name, data) in enumerate(ranked):
            print(f"{i + 1:2d}. {name:15s} 评分: {data['composite_score']:.3f}  "
                  f"弯曲: {data['mean_max_angle']:5.1f}°  "
                  f"抬升: {data['lift_height_mm']:5.2f} mm")

        return results, ranked

    def visualize_config(self, mpc_config, config_name="Custom Configuration", save_path=None):
        """
        可视化磁矩配置的完整性能

        参数：
        mpc_config: 磁矩配置
        config_name: 配置名称
        save_path: 保存路径（可选）
        """
        # 评估配置
        results = self.evaluate_magnet_config(mpc_config)

        # 创建图形
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(f'Magnet Configuration Analysis: {config_name}\n{self.config_to_string(mpc_config)}',
                     fontsize=16, fontweight='bold')

        # 使用GridSpec创建子图布局
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # 1. 磁矩配置示意图
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_magnet_config(mpc_config, ax1)

        # 2. 垂直磁场下的姿态（有重力）
        ax2 = fig.add_subplot(gs[0, 1])
        B_vertical = np.array([0, self.B_magnitude])
        q_vertical = self.robot.find_equilibrium(B_vertical, mpc_config, include_gravity=True)
        self._plot_robot_state(q_vertical, mpc_config, B_vertical, ax2)
        metrics_v = self.compute_static_metrics(q_vertical, mpc_config)
        ax2.set_title(f'Vertical Magnetic Field (Lifting)', fontsize=12, fontweight='bold')

        # 添加抬升高度信息
        ax2.text(0.05, 0.95, f"Lift Height: {metrics_v['lift_height_mm']:.2f} mm\n"
                             f"End Height: {metrics_v['end_height_mm']:.2f} mm",
                 transform=ax2.transAxes, fontsize=10,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        # 3. 水平磁场下的姿态（无重力）
        ax3 = fig.add_subplot(gs[0, 2])
        B_horizontal = np.array([self.B_magnitude, 0])
        q_horizontal = self.robot.find_equilibrium(B_horizontal, mpc_config, include_gravity=False)
        self._plot_robot_state(q_horizontal, mpc_config, B_horizontal, ax3)
        metrics_h = self.compute_static_metrics(q_horizontal, mpc_config)
        ax3.set_title(f'Horizontal Magnetic Field (Bending)', fontsize=12, fontweight='bold')

        # 添加弯曲信息
        ax3.text(0.05, 0.95, f"Max Joint Angle: {metrics_h['max_joint_angle_deg']:.1f}°\n"
                             f"Bending Mode: {metrics_h['bending_mode']}",
                 transform=ax3.transAxes, fontsize=10,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # 4. 最佳弯曲方向的姿态
        ax4 = fig.add_subplot(gs[0, 3])
        best_dir = results['best_bending_direction']
        B_best = self.B_magnitude * np.array([np.cos(np.deg2rad(best_dir)),
                                              np.sin(np.deg2rad(best_dir))])
        q_best = self.robot.find_equilibrium(B_best, mpc_config, include_gravity=False)
        self._plot_robot_state(q_best, mpc_config, B_best, ax4)
        ax4.set_title(f'Best Bending Direction ({best_dir:.0f}°)', fontsize=12, fontweight='bold')

        # 5. 方向性能极坐标图
        ax5 = fig.add_subplot(gs[1, 0:2], projection='polar')
        self._plot_directional_performance(results, ax5)

        # 6. 关节角度分布
        ax6 = fig.add_subplot(gs[1, 2:4])
        self._plot_joint_angles_distribution(results, ax6)

        # 7. 综合评分雷达图
        ax7 = fig.add_subplot(gs[2, 0:2], projection='polar')
        self._plot_radar_chart(results, ax7)

        # 8. 性能摘要
        ax8 = fig.add_subplot(gs[2, 2:4])
        self._plot_performance_summary(results, ax8)

        # 调整布局
        plt.tight_layout()

        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图形已保存到: {save_path}")

        plt.show()

        return fig

    def _plot_magnet_config(self, mpc_config, ax):
        """绘制磁矩配置图"""
        # 设置背景
        ax.set_facecolor('#f8f9fa')

        for i in range(self.robot.N_links):
            # 连杆位置
            x_start = i * 1.2
            x_end = x_start + 1.0
            y = 0

            # 绘制连杆
            ax.plot([x_start, x_end], [y, y], 'k-', linewidth=4, alpha=0.8, solid_capstyle='round')

            # 绘制连杆编号
            ax.text((x_start + x_end) / 2, y + 0.15, f'L{i}',
                    ha='center', va='center', fontsize=9, fontweight='bold')

            # 磁矩角度
            angle_deg = np.rad2deg(mpc_config[i])

            # 磁矩箭头
            if angle_deg == 0:
                # 向前（右）
                dx = 0.4
                dy = 0
                color = 'red'
            else:
                # 向后（左）
                dx = -0.4
                dy = 0
                color = 'blue'

            # 绘制箭头
            arrow = FancyArrowPatch(
                ((x_start + x_end) / 2, y),
                ((x_start + x_end) / 2 + dx, y + dy),
                arrowstyle='->', color=color, linewidth=2,
                mutation_scale=15
            )
            ax.add_patch(arrow)

            # 磁矩角度标签
            ax.text((x_start + x_end) / 2, y - 0.15, f'{angle_deg:.0f}°',
                    ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        # 设置图形属性
        ax.set_xlim(-0.5, self.robot.N_links * 1.2 - 0.2)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect('equal')
        ax.set_title('Magnet Configuration Diagram', fontsize=12, fontweight='bold')
        ax.set_xlabel('Link Position')
        ax.grid(True, alpha=0.3, linestyle='--')

    def _plot_robot_state(self, q, mpc_config, B_field, ax):
        """绘制机器人姿态"""
        # 计算运动学
        centers, joints, abs_angles = self.robot.forward_kinematics(q)

        # 转换为毫米
        joints_mm = joints * self.robot.scale_factor
        centers_mm = centers * self.robot.scale_factor

        # 设置背景
        ax.set_facecolor('#f8f9fa')

        # 绘制地面
        ax.axhline(y=0, color='black', linestyle='-', linewidth=2, alpha=0.7)
        ax.fill_between([np.min(joints_mm[:, 0]) - 2, np.max(joints_mm[:, 0]) + 2],
                        -1, 0, color='gray', alpha=0.3)

        # 绘制机器人连杆
        for i in range(self.robot.N_links):
            # 连杆
            ax.plot([joints_mm[i, 0], joints_mm[i + 1, 0]],
                    [joints_mm[i, 1], joints_mm[i + 1, 1]],
                    color='navy', linewidth=3, alpha=0.8, solid_capstyle='round')

            # 关节
            ax.plot(joints_mm[i, 0], joints_mm[i, 1], 'o',
                    color='darkred', markersize=8, alpha=0.8)

            # 磁矩箭头
            center_x = (joints_mm[i, 0] + joints_mm[i + 1, 0]) / 2
            center_y = (joints_mm[i, 1] + joints_mm[i + 1, 1]) / 2

            m_angle = abs_angles[i] + mpc_config[i]
            arrow_length = 0.3 * self.robot.L * self.robot.scale_factor
            dx = arrow_length * np.cos(m_angle)
            dy = arrow_length * np.sin(m_angle)

            arrow = FancyArrowPatch(
                (center_x, center_y),
                (center_x + dx, center_y + dy),
                arrowstyle='->', color='red', linewidth=2,
                mutation_scale=10
            )
            ax.add_patch(arrow)

        # 绘制末端
        ax.plot(joints_mm[-1, 0], joints_mm[-1, 1], 'o',
                color='green', markersize=10, alpha=0.8, label='End Effector')

        # 绘制磁场方向
        B_angle = np.arctan2(B_field[1], B_field[0])
        B_length = 3.0
        B_origin = [joints_mm[0, 0], joints_mm[0, 1] + 3]

        arrow = FancyArrowPatch(
            B_origin,
            [B_origin[0] + B_length * np.cos(B_angle), B_origin[1] + B_length * np.sin(B_angle)],
            arrowstyle='->', color='green', linewidth=3,
            mutation_scale=15
        )
        ax.add_patch(arrow)

        ax.text(B_origin[0], B_origin[1] + 1.5,
                f'B: {np.linalg.norm(B_field) * 1000:.1f} mT',
                fontsize=10, color='green', fontweight='bold')

        # 设置图形属性
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')

        # 自动调整坐标范围
        margin = 2.0
        x_min, x_max = np.min(joints_mm[:, 0]), np.max(joints_mm[:, 0])
        y_min, y_max = np.min(joints_mm[:, 1]), np.max(joints_mm[:, 1])
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(max(-1, y_min - margin), y_max + margin)

        # 添加图例
        ax.legend(loc='upper right')

    def _plot_directional_performance(self, results, ax):
        """绘制方向性能极坐标图"""
        # 提取数据
        detailed_results = results['detailed_results']
        directions = [r['direction_deg'] for r in detailed_results]
        max_angles = [r['max_angle'] for r in detailed_results]

        # 转换为弧度
        directions_rad = np.deg2rad(directions)

        # 闭合图形
        directions_rad_closed = np.append(directions_rad, directions_rad[0])
        max_angles_closed = np.append(max_angles, max_angles[0])

        # 绘制极坐标图
        ax.plot(directions_rad_closed, max_angles_closed, 'b-', linewidth=2.5, alpha=0.8)
        ax.fill(directions_rad_closed, max_angles_closed, alpha=0.2, color='blue')

        # 标记最佳方向
        best_dir = results['best_bending_direction']
        best_idx = np.argmin(np.abs(np.array(directions) - best_dir))
        ax.plot(directions_rad[best_idx], max_angles[best_idx], 'ro', markersize=10)

        # 设置极坐标图属性
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim([0, 80])
        ax.set_yticks([20, 40, 60, 80])
        ax.set_yticklabels(['20°', '40°', '60°', '80°'])

        # 添加方向标签
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'])

        ax.set_title('Directional Performance Polar Plot', fontsize=12, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.5)

        # 添加说明
        ax.text(0.5, 1.05, f"Best Direction: {best_dir:.0f}°",
                transform=ax.transAxes, ha='center', fontsize=10)

    def _plot_joint_angles_distribution(self, results, ax):
        """绘制关节角度分布"""
        # 提取关节角度
        joint_angles_deg = results['best_metrics']['joint_angles_deg']
        joint_indices = np.arange(1, len(joint_angles_deg) + 1)

        # 创建条形图
        bars = ax.bar(joint_indices, joint_angles_deg,
                      color='steelblue', alpha=0.7, edgecolor='navy', linewidth=1.5)

        # 添加数值标签
        for i, (idx, angle) in enumerate(zip(joint_indices, joint_angles_deg)):
            color = 'red' if abs(angle) > 60 else ('orange' if abs(angle) > 30 else 'green')
            ax.text(idx, angle + (1 if angle >= 0 else -3), f'{angle:.1f}°',
                    ha='center', va='bottom' if angle >= 0 else 'top',
                    fontsize=9, fontweight='bold', color=color)

        # 添加关节约束线
        max_angle = np.rad2deg(self.robot.max_angle)
        ax.axhline(y=max_angle, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.axhline(y=-max_angle, color='red', linestyle='--', alpha=0.7, linewidth=1.5)

        # 设置图形属性
        ax.set_xlabel('Joint Number', fontsize=11)
        ax.set_ylabel('Joint Angle (°)', fontsize=11)
        ax.set_title('Joint Angle Distribution', fontsize=12, fontweight='bold')
        ax.set_xticks(joint_indices)
        ax.set_xticklabels([f'J{i}' for i in joint_indices])
        ax.set_ylim([-80, 80])
        ax.grid(True, alpha=0.3, axis='y')

        # 添加统计信息
        mean_angle = np.mean(np.abs(joint_angles_deg))
        max_angle_val = np.max(np.abs(joint_angles_deg))

        ax.text(0.02, 0.98, f"Mean Absolute Angle: {mean_angle:.1f}°\nMax Angle: {max_angle_val:.1f}°",
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    def _plot_radar_chart(self, results, ax):
        """绘制综合评分雷达图"""
        # 提取归一化评分
        scores = results['normalized_scores']
        categories = list(scores.keys())
        values = list(scores.values())

        # 确保有足够的数据
        if len(categories) < 3:
            print("警告：类别数量不足，无法绘制雷达图")
            return

        N = len(categories)

        # 角度
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形

        # 数值
        values += values[:1]

        # 绘制雷达图
        ax.plot(angles, values, 'o-', linewidth=2.5, color='darkblue', alpha=0.8)
        ax.fill(angles, values, alpha=0.25, color='steelblue')

        # 设置角度标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)

        # 设置径向网格
        ax.set_ylim([0, 1.0])
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, alpha=0.5)

        ax.set_title('Comprehensive Performance Radar Chart', fontsize=12, fontweight='bold', pad=20)

        # 添加综合评分
        composite_score = results['composite_score']
        ax.text(0.5, 1.05, f"Composite Score: {composite_score:.3f}",
                transform=ax.transAxes, ha='center', fontsize=11, fontweight='bold')

    def _plot_performance_summary(self, results, ax):
        """绘制性能摘要"""
        # 清空坐标轴
        ax.clear()
        ax.axis('off')

        # 创建性能摘要文本
        summary_text = "Performance Summary\n\n"
        summary_text += f"Magnet Configuration: {results['config_string']}\n\n"

        summary_text += f"Bending Performance:\n"
        summary_text += f"  Mean Max Joint Angle: {results['mean_max_angle']:.1f}°\n"
        summary_text += f"  Max Joint Angle: {results['max_max_angle']:.1f}°\n"
        summary_text += f"  Best Bending Direction: {results['best_bending_direction']:.0f}°\n\n"

        summary_text += f"Lifting Performance:\n"
        summary_text += f"  Lift Height: {results['lift_height_mm']:.2f} mm\n\n"

        summary_text += f"Isotropy:\n"
        summary_text += f"  Directional Isotropy: {results['isotropy']:.3f}\n\n"

        summary_text += f"Composite Score:\n"
        summary_text += f"  {results['composite_score']:.3f}\n\n"

        summary_text += f"Bending Mode:\n"
        summary_text += f"  {results['best_metrics']['bending_mode']}\n"
        summary_text += f"  Sign Changes: {results['best_metrics']['sign_changes']}"

        # 显示文本
        ax.text(0.1, 0.95, summary_text,
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        ax.set_title('Performance Summary', fontsize=12, fontweight='bold')



# ============================================================================
# 3. 优化器类
# ============================================================================

class MagnetConfigOptimizer:
    """
    磁矩配置优化器
    通过搜索算法寻找最优的0/180°磁矩配置
    """

    def __init__(self, simulator):
        """
        初始化优化器

        参数：
        simulator: 静态仿真器实例
        """
        self.simulator = simulator
        self.robot = simulator.robot

    def exhaustive_search(self, max_configs=None):
        """
        穷举搜索所有可能的0/180°配置

        参数：
        max_configs: 最大搜索配置数（None表示全部搜索）

        返回：
        best_config: 最佳配置
        best_score: 最佳评分
        all_results: 所有配置的结果
        """
        n_links = self.robot.N_links
        total_configs = 2 ** n_links

        if max_configs is not None and max_configs < total_configs:
            print(f"随机搜索 {max_configs} 个配置（共 {total_configs} 个）")
            return self.random_search(max_configs)

        print(f"穷举搜索所有 {total_configs} 个配置...")

        best_score = -float('inf')
        best_config = None
        all_results = []

        # 遍历所有可能的二进制组合
        for i in range(total_configs):
            # 生成二进制表示的磁矩配置
            binary = format(i, f'0{n_links}b')
            config = [0.0 if bit == '0' else np.pi for bit in binary]

            # 评估配置
            result = self.simulator.evaluate_magnet_config(config)
            score = result['composite_score']

            all_results.append({
                'config': config,
                'score': score,
                'mean_max_angle': result['mean_max_angle'],
                'lift_height': result['lift_height_mm'],
                'isotropy': result['isotropy']
            })

            # 更新最佳配置
            if score > best_score:
                best_score = score
                best_config = config

            # 进度显示
            if (i + 1) % 10 == 0 or i == total_configs - 1:
                print(f"  进度: {i + 1}/{total_configs} ({100 * (i + 1) / total_configs:.1f}%)")

        # 按评分排序
        all_results.sort(key=lambda x: x['score'], reverse=True)

        print(f"\n搜索完成！最佳评分: {best_score:.3f}")
        print(f"最佳配置: {self.simulator.config_to_string(best_config)}")

        return best_config, best_score, all_results

    def random_search(self, n_samples=100):
        """
        随机搜索磁矩配置

        参数：
        n_samples: 随机采样数量

        返回：
        best_config: 最佳配置
        best_score: 最佳评分
        all_results: 所有采样结果
        """
        print(f"随机搜索 {n_samples} 个配置...")

        best_score = -float('inf')
        best_config = None
        all_results = []

        for i in range(n_samples):
            # 随机生成0/180°配置
            config = [np.random.choice([0.0, np.pi]) for _ in range(self.robot.N_links)]

            # 评估配置
            result = self.simulator.evaluate_magnet_config(config)
            score = result['composite_score']

            all_results.append({
                'config': config,
                'score': score,
                'mean_max_angle': result['mean_max_angle'],
                'lift_height': result['lift_height_mm'],
                'isotropy': result['isotropy']
            })

            # 更新最佳配置
            if score > best_score:
                best_score = score
                best_config = config

            # 进度显示
            if (i + 1) % 10 == 0 or i == n_samples - 1:
                print(f"  进度: {i + 1}/{n_samples} ({100 * (i + 1) / n_samples:.1f}%)")

        # 按评分排序
        all_results.sort(key=lambda x: x['score'], reverse=True)

        print(f"\n随机搜索完成！最佳评分: {best_score:.3f}")
        print(f"最佳配置: {self.simulator.config_to_string(best_config)}")

        return best_config, best_score, all_results

    def genetic_algorithm(self, population_size=20, generations=10, mutation_rate=0.1):
        """
        遗传算法优化磁矩配置

        参数：
        population_size: 种群大小
        generations: 迭代代数
        mutation_rate: 变异率

        返回：
        best_config: 最佳配置
        best_score: 最佳评分
        history: 进化历史
        """
        print(f"遗传算法优化: 种群大小={population_size}, 代数={generations}")

        # 初始化种群
        population = []
        for _ in range(population_size):
            config = [np.random.choice([0.0, np.pi]) for _ in range(self.robot.N_links)]
            population.append(config)

        best_config = None
        best_score = -float('inf')
        history = {'scores': [], 'configs': []}

        # 进化循环
        for gen in range(generations):
            print(f"\n第 {gen + 1}/{generations} 代:")

            # 评估种群
            scores = []
            for config in population:
                result = self.simulator.evaluate_magnet_config(config)
                score = result['composite_score']
                scores.append(score)

                # 更新全局最佳
                if score > best_score:
                    best_score = score
                    best_config = config

            # 记录历史
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            history['scores'].append((avg_score, max_score))
            history['configs'].append(population[np.argmax(scores)])

            print(f"  平均评分: {avg_score:.3f}, 最高评分: {max_score:.3f}")

            # 选择（轮盘赌选择）
            if gen < generations - 1:  # 最后一代不进行选择交叉变异
                # 归一化适应度
                scores_array = np.array(scores)
                min_score = np.min(scores_array)
                if min_score < 0:
                    scores_array = scores_array - min_score + 0.001
                fitness = scores_array / np.sum(scores_array)

                # 选择父代
                selected_indices = np.random.choice(
                    range(population_size),
                    size=population_size,
                    p=fitness
                )

                # 交叉和变异生成新种群
                new_population = []
                for i in range(0, population_size, 2):
                    if i + 1 < population_size:
                        parent1 = population[selected_indices[i]]
                        parent2 = population[selected_indices[i + 1]]

                        # 单点交叉
                        crossover_point = np.random.randint(1, self.robot.N_links - 1)
                        child1 = parent1[:crossover_point] + parent2[crossover_point:]
                        child2 = parent2[:crossover_point] + parent1[crossover_point:]

                        # 变异
                        for j in range(self.robot.N_links):
                            if np.random.random() < mutation_rate:
                                child1[j] = 0.0 if child1[j] == np.pi else np.pi
                            if np.random.random() < mutation_rate:
                                child2[j] = 0.0 if child2[j] == np.pi else np.pi

                        new_population.extend([child1, child2])

                # 确保种群大小不变
                if len(new_population) > population_size:
                    new_population = new_population[:population_size]
                elif len(new_population) < population_size:
                    # 补充随机个体
                    while len(new_population) < population_size:
                        config = [np.random.choice([0.0, np.pi]) for _ in range(self.robot.N_links)]
                        new_population.append(config)

                population = new_population

        print(f"\n遗传算法优化完成！最佳评分: {best_score:.3f}")
        print(f"最佳配置: {self.simulator.config_to_string(best_config)}")

        return best_config, best_score, history


# ============================================================================
# 4. 主程序
# ============================================================================

def main():
    """
    主程序：演示静态仿真器的完整功能
    """
    print("=" * 70)
    print("磁控铰链机器人静态仿真器")
    print("=" * 70)

    # 步骤1：创建机器人模型
    print("\n1. 创建机器人模型...")
    robot = MagneticHingeRobot(
        link_length_mm=1.5,
        width_mm=0.4,
        height_mm=0.4,
        magnetization_T=1.2,
        max_angle_deg=70.0
    )

    # 步骤2：创建仿真器
    print("\n2. 创建静态仿真器...")
    simulator = StaticSimulator(robot=robot, B_magnitude=0.02)

    # 步骤3：评估常见配置
    print("\n3. 评估常见磁矩配置...")
    results, ranked = simulator.evaluate_all_configs()

    # 步骤4：可视化最佳配置
    print("\n4. 可视化最佳配置...")
    best_name, best_data = ranked[0]
    simulator.visualize_config(best_data['mpc_config'], config_name=best_name, save_path="best_config.png")

    # 步骤5：创建优化器
    print("\n5. 创建优化器...")
    optimizer = MagnetConfigOptimizer(simulator)

    # 步骤6：执行优化
    print("\n6. 执行优化搜索...")

    # 方法选择
    print("\n选择优化方法:")
    print("1. 穷举搜索 (64种配置)")
    print("2. 随机搜索 (100种配置)")
    print("3. 遗传算法 (20种群, 10代)")

    choice = input("\n请输入选择 (1-3, 默认1): ").strip()

    if choice == '2':
        best_config, best_score, all_results = optimizer.random_search(n_samples=100)
    elif choice == '3':
        best_config, best_score, history = optimizer.genetic_algorithm(
            population_size=20,
            generations=10
        )
    else:
        best_config, best_score, all_results = optimizer.exhaustive_search()

    # 步骤7：可视化优化结果
    print("\n7. 可视化优化结果...")
    simulator.visualize_config(best_config, config_name="优化结果", save_path="optimized_config.png")

    # 步骤8：生成报告
    print("\n8. 生成优化报告...")
    generate_report(simulator, best_config, best_score)

    print("\n" + "=" * 70)
    print("仿真完成！")
    print("=" * 70)


def generate_report(simulator, best_config, best_score):
    """
    生成优化报告
    """
    # 评估最佳配置
    result = simulator.evaluate_magnet_config(best_config)

    print("\n" + "=" * 70)
    print("优化报告")
    print("=" * 70)

    print(f"\n最佳磁矩配置: {simulator.config_to_string(best_config)}")
    print(f"综合评分: {best_score:.3f}")

    print("\n性能指标:")
    print(f"  平均最大关节角: {result['mean_max_angle']:.1f}°")
    print(f"  最大关节角: {result['max_max_angle']:.1f}°")
    print(f"  方向各向同性: {result['isotropy']:.3f}")
    print(f"  抬升高度: {result['lift_height_mm']:.2f} mm")
    print(f"  最佳弯曲方向: {result['best_bending_direction']:.0f}°")

    print("\n归一化评分:")
    for category, score in result['normalized_scores'].items():
        print(f"  {category:12s}: {score:.3f}")

    print("\n建议:")
    if result['mean_max_angle'] > 50:
        print("  ✓ 弯曲性能优秀，适合需要大范围弯曲的应用")
    elif result['mean_max_angle'] > 30:
        print("  ✓ 弯曲性能良好，适合一般运动需求")
    else:
        print("  ⚠ 弯曲性能一般，可能需要优化")

    if result['isotropy'] > 0.7:
        print("  ✓ 各向同性优秀，对磁场方向不敏感，易于控制")
    elif result['isotropy'] > 0.5:
        print("  ✓ 各向同性良好，在大多数方向表现一致")
    else:
        print("  ⚠ 各向同性较差，对磁场方向敏感")

    if result['lift_height_mm'] > 0.5:
        print(f"  ✓ 抬升性能良好 ({result['lift_height_mm']:.2f} mm)，适合爬坡/越障")
    else:
        print("  ⚠ 抬升性能有限，可能不适合爬坡应用")

    print("\n" + "=" * 70)


# ============================================================================
# 5. 快速使用示例
# ============================================================================

def quick_example():
    """
    快速使用示例
    """
    print("快速示例: 评估交替磁化配置")

    # 创建仿真器
    robot = MagneticHingeRobot()
    simulator = StaticSimulator(robot)

    # 评估交替磁化配置
    alternating_config = [0, np.pi, 0, np.pi, 0, np.pi]
    result = simulator.evaluate_magnet_config(alternating_config)

    print(f"\n配置: {simulator.config_to_string(alternating_config)}")
    print(f"综合评分: {result['composite_score']:.3f}")
    print(f"平均最大关节角: {result['mean_max_angle']:.1f}°")
    print(f"抬升高度: {result['lift_height_mm']:.2f} mm")

    # 可视化
    simulator.visualize_config(alternating_config, config_name="交替磁化")


# ============================================================================
# 6. 测试自定义配置
# ============================================================================

def test_custom_config():
    """
    测试自定义磁矩配置
    """
    print("测试自定义配置")

    # 创建仿真器
    robot = MagneticHingeRobot()
    simulator = StaticSimulator(robot)

    # 自定义配置（例如：前半向前，后半向后）
    custom_config = [0, 0, 0, np.pi, np.pi, np.pi]

    # 评估
    result = simulator.evaluate_magnet_config(custom_config)

    print(f"\n自定义配置: {simulator.config_to_string(custom_config)}")
    print(f"综合评分: {result['composite_score']:.3f}")

    # 可视化
    simulator.visualize_config(custom_config, config_name="自定义配置")


# ============================================================================
# 运行程序
# ============================================================================

if __name__ == "__main__":
    # 运行完整仿真
    main()

    # 或者运行快速示例
    # quick_example()

    # 或者测试自定义配置
    # test_custom_config()
