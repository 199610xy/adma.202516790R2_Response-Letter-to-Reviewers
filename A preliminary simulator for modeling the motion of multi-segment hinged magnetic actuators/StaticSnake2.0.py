import numpy as np
from scipy.optimize import minimize
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class MagneticRobotAnalyzer:
    """磁控铰链机器人性能分析器 - 修正版（区分XY和XZ平面）"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("磁控铰链机器人性能分析器")
        self.root.geometry("1400x800")

        # 分析模式：XY平面（水平）或 XZ平面（垂直）
        self.analysis_mode = tk.StringVar(value="XY")  # 默认XY平面

        # 机器人参数
        self.link_length_mm = 1.5
        self.magnetization_T = 1.2
        self.density_kg_m3 = 2000.0
        self.max_angle_deg = 70.0
        self.B_magnitude = 0.02  # 20 mT

        # 磁矩配置 (默认交替磁化)
        self.mpc_config = [0, np.pi, 0, np.pi, 0, np.pi]

        # 创建GUI
        self.create_widgets()

        # 初始化机器人参数
        self.init_robot_params()

    def init_robot_params(self):
        """初始化机器人物理参数"""
        self.N_links = 6
        self.N_joints = self.N_links - 1
        self.L = self.link_length_mm * 1e-3  # 转换为米
        self.w = 0.4e-3
        self.h = 0.4e-3

        # 磁矩计算
        mu0 = 4 * np.pi * 1e-7
        M = self.magnetization_T / mu0
        vol = self.L * self.w * self.h
        self.m_mag = M * vol

        # 质量
        self.mass = self.density_kg_m3 * vol

        # 关节约束
        self.max_angle = np.deg2rad(self.max_angle_deg)
        self.k_joint = 1e3

        # 物理常数
        self.g = 9.81
        self.scale_factor = 1e3  # 米到毫米

    def forward_kinematics_2d(self, q, mode="XY"):
        """二维正向运动学"""
        if mode == "XY":
            # XY平面：x0, y0, phi0
            x0, y0, phi0 = q[0], q[1], q[2]
        else:  # XZ平面
            # XZ平面：x0, z0, phi0
            x0, z0, phi0 = q[0], q[1], q[2]

        joint_angles = q[3:]

        # 计算绝对角度
        abs_angles = np.cumsum(np.concatenate(([phi0], joint_angles)))

        # 计算关节位置
        if mode == "XY":
            joints = np.zeros((self.N_links + 1, 2))
            joints[0] = np.array([x0, y0])

            for i in range(self.N_links):
                dir_vec = np.array([np.cos(abs_angles[i]), np.sin(abs_angles[i])])
                joints[i + 1] = joints[i] + self.L * dir_vec

            # 计算质心位置
            centers = np.zeros((self.N_links, 2))
            for i in range(self.N_links):
                dir_vec = np.array([np.cos(abs_angles[i]), np.sin(abs_angles[i])])
                centers[i] = joints[i] + 0.5 * self.L * dir_vec
        else:  # XZ平面
            joints = np.zeros((self.N_links + 1, 2))
            joints[0] = np.array([x0, z0])

            for i in range(self.N_links):
                dir_vec = np.array([np.cos(abs_angles[i]), np.sin(abs_angles[i])])
                joints[i + 1] = joints[i] + self.L * dir_vec

            centers = np.zeros((self.N_links, 2))
            for i in range(self.N_links):
                dir_vec = np.array([np.cos(abs_angles[i]), np.sin(abs_angles[i])])
                centers[i] = joints[i] + 0.5 * self.L * dir_vec

        return centers, joints, abs_angles

    def compute_energy_2d(self, q, B_field, mpc_config, mode="XY", include_gravity=False):
        """计算二维总势能"""
        centers, joints, abs_angles = self.forward_kinematics_2d(q, mode)
        total_energy = 0.0

        # 磁势能
        for i in range(self.N_links):
            m_angle_global = abs_angles[i] + mpc_config[i]
            m_vector = self.m_mag * np.array([np.cos(m_angle_global), np.sin(m_angle_global)])
            total_energy += -np.dot(m_vector, B_field)

        # 关节约束
        joint_angles = q[3:]
        for angle in joint_angles:
            if angle > self.max_angle:
                total_energy += 0.5 * self.k_joint * (angle - self.max_angle) ** 2
            elif angle < -self.max_angle:
                total_energy += 0.5 * self.k_joint * (angle + self.max_angle) ** 2

        # 重力（只在XZ模式中考虑）
        if include_gravity and mode == "XZ":
            for i in range(self.N_links):
                # 注意：在XZ模式中，重力方向是-Z方向
                # 所以重力势能是 +mass * g * z（z越高势能越大）
                total_energy += self.mass * self.g * centers[i, 1]  # centers[i, 1] 是Z坐标

        # 地面约束
        if mode == "XZ":
            min_z = np.min(joints[:, 1])  # 第二个坐标是Z
            if min_z < 0:  # Z=0是地面
                total_energy += 0.5 * 1e6 * min_z ** 2
        elif mode == "XY":
            # XY平面中没有地面约束，但我们可以限制机器人不能进入负Y区域
            min_y = np.min(joints[:, 1])
            if min_y < -10 * self.L:  # 一个宽松的约束
                total_energy += 0.5 * 1e3 * (min_y + 10 * self.L) ** 2

        return total_energy

    def find_equilibrium_2d(self, B_field, mpc_config, mode="XY", fixed_base=False, base_position=None):
        """寻找二维平衡姿态"""
        initial_guess = np.zeros(3 + self.N_joints)

        if mode == "XY":
            # XY平面：初始猜测在原点附近
            initial_guess[0] = 0.0
            initial_guess[1] = 0.0  # Y坐标
            initial_guess[2] = 0.0  # 初始角度
        else:  # XZ平面
            # XZ平面：初始猜测在Z=0.1mm处
            initial_guess[0] = 0.0
            initial_guess[1] = 0.1e-3  # Z坐标（离地面0.1mm）
            initial_guess[2] = 0.0  # 初始角度

        initial_guess[3:] = np.random.uniform(-0.001, 0.001, self.N_joints)

        bounds = []

        if fixed_base and base_position is not None:
            bounds.append((base_position[0], base_position[0]))
            bounds.append((base_position[1], base_position[1]))
            bounds.append((base_position[2], base_position[2]))
        else:
            if mode == "XY":
                bounds.append((-10 * self.L, 10 * self.L))  # X
                bounds.append((-10 * self.L, 10 * self.L))  # Y
                bounds.append((-np.pi, np.pi))  # phi
            else:  # XZ平面
                bounds.append((-10 * self.L, 10 * self.L))  # X
                bounds.append((0, 10 * self.L))  # Z（不能低于地面）
                bounds.append((-np.pi, np.pi))  # phi

        for _ in range(self.N_joints):
            bounds.append((-self.max_angle, self.max_angle))

        def objective(q):
            if mode == "XY":
                return self.compute_energy_2d(q, B_field, mpc_config, "XY", include_gravity=False)
            else:
                return self.compute_energy_2d(q, B_field, mpc_config, "XZ", include_gravity=True)

        result = minimize(
            objective,
            initial_guess,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-8, 'disp': False}
        )

        return result.x

    def analyze_horizontal_xy(self):
        """分析XY平面（水平面）性能"""
        mode = "XY"
        self.xy_results = {}

        # 更新磁矩配置
        self.update_magnet_config()

        # 采样磁场方向（在XY平面内）
        n_samples = 36
        reachable_points = []

        # 基座固定在原点
        base_position = [0, 0, 0]

        for i in range(n_samples):
            phi = i * 2 * np.pi / n_samples
            B_field = self.B_magnitude * np.array([np.cos(phi), np.sin(phi)])

            q_eq = self.find_equilibrium_2d(
                B_field, self.mpc_config,
                mode=mode,
                fixed_base=True,
                base_position=base_position
            )

            _, joints, _ = self.forward_kinematics_2d(q_eq, mode)
            head_position = joints[-1] * self.scale_factor

            reachable_points.append(head_position)

        reachable_points = np.array(reachable_points)

        # 计算XY平面指标
        self.xy_results["reachable_points"] = reachable_points
        self.xy_results["min_x"] = np.min(reachable_points[:, 0])
        self.xy_results["max_x"] = np.max(reachable_points[:, 0])
        self.xy_results["min_y"] = np.min(reachable_points[:, 1])
        self.xy_results["max_y"] = np.max(reachable_points[:, 1])
        self.xy_results["area_width"] = self.xy_results["max_x"] - self.xy_results["min_x"]
        self.xy_results["area_height"] = self.xy_results["max_y"] - self.xy_results["min_y"]
        self.xy_results["area_size"] = self.xy_results["area_width"] * self.xy_results["area_height"]

        # 计算覆盖半径
        distances = np.sqrt(reachable_points[:, 0] ** 2 + reachable_points[:, 1] ** 2)
        self.xy_results["max_radius"] = np.max(distances)
        self.xy_results["avg_radius"] = np.mean(distances)

        return self.xy_results

    def analyze_vertical_xz(self):
        """分析XZ平面（垂直面）性能"""
        mode = "XZ"
        self.xz_results = {}

        # 更新磁矩配置
        self.update_magnet_config()

        # 1. 计算最大抬升高度（磁场垂直向上）
        B_field_up = np.array([0, self.B_magnitude])  # [Bx, Bz]，这里Bz向上

        q_eq_up = self.find_equilibrium_2d(
            B_field_up, self.mpc_config,
            mode=mode,
            fixed_base=False
        )

        _, joints_up, _ = self.forward_kinematics_2d(q_eq_up, mode)
        joints_up_mm = joints_up * self.scale_factor

        # 计算抬升高度
        min_z = np.min(joints_up_mm[:, 1])  # Z坐标
        end_z = joints_up_mm[-1, 1]
        lift_height = max(0, end_z - min_z)

        self.xz_results["max_lift_height"] = lift_height
        self.xz_results["head_height"] = end_z
        self.xz_results["min_height"] = min_z
        self.xz_results["lift_state"] = q_eq_up
        self.xz_results["lift_joints"] = joints_up_mm

        # 2. 计算最大爬坡角度（磁场在XZ平面内倾斜）
        # 我们尝试不同的磁场方向，找到能维持稳定姿态的最大角度
        max_climb_angle = 0
        optimal_climb_state = None

        for angle_deg in np.linspace(0, 80, 17):  # 0°到80°
            angle_rad = np.deg2rad(angle_deg)

            # 磁场方向：倾斜angle_deg度（从水平方向算起）
            B_field_climb = self.B_magnitude * np.array([np.cos(angle_rad), np.sin(angle_rad)])

            q_eq_climb = self.find_equilibrium_2d(
                B_field_climb, self.mpc_config,
                mode=mode,
                fixed_base=False
            )

            _, joints_climb, _ = self.forward_kinematics_2d(q_eq_climb, mode)
            joints_climb_mm = joints_climb * self.scale_factor

            # 检查所有关节是否在地面以上
            if np.min(joints_climb_mm[:, 1]) > 0.1:  # 所有点离地面至少0.1mm
                max_climb_angle = angle_deg
                optimal_climb_state = q_eq_climb
                optimal_climb_joints = joints_climb_mm
            else:
                break

        self.xz_results["max_climb_angle"] = max_climb_angle
        self.xz_results["climb_state"] = optimal_climb_state
        self.xz_results["climb_joints"] = optimal_climb_joints if optimal_climb_state is not None else None

        # 3. 计算越障能力（粗略估计）
        # 假设机器人头部能抬起的最低点高度
        clearance_height = max(0, min_z - 0)  # 最低点离地面的高度
        self.xz_results["clearance_height"] = clearance_height

        return self.xz_results

    def update_magnet_config(self):
        """更新磁矩配置"""
        self.mpc_config = []
        for i in range(6):
            var = getattr(self, f'magnet_var_{i}')
            self.mpc_config.append(0 if var.get() == "向前" else np.pi)

        config_text = f"当前配置: {['向前' if a == 0 else '向后' for a in self.mpc_config]}"
        self.config_label.config(text=config_text)

    def analyze(self):
        """执行分析"""
        # 禁用按钮
        self.analyze_button.config(state=tk.DISABLED, text="分析中...")
        self.root.update()

        try:
            # 获取分析模式
            mode = self.analysis_mode.get()

            if mode == "XY":
                # 分析XY平面（水平面）
                self.analyze_horizontal_xy()
                self.update_xy_results()
                self.update_xy_plot()
            else:  # XZ
                # 分析XZ平面（垂直面）
                self.analyze_vertical_xz()
                self.update_xz_results()
                self.update_xz_plot()

        except Exception as e:
            messagebox.showerror("错误", f"分析过程中出现错误:\n{str(e)}")
            import traceback
            traceback.print_exc()

        # 启用按钮
        self.analyze_button.config(state=tk.NORMAL, text="开始分析")

    def update_xy_results(self):
        """更新XY平面分析结果"""
        results_text = f"磁矩配置: {['向前' if a == 0 else '向后' for a in self.mpc_config]}\n"
        results_text += f"分析模式: XY平面（水平面）\n\n"

        results_text += "=== XY平面性能指标 ===\n\n"
        results_text += "1. 可达范围分析:\n"
        results_text += f"   X方向范围: [{self.xy_results['min_x']:.2f}, {self.xy_results['max_x']:.2f}] mm\n"
        results_text += f"   Y方向范围: [{self.xy_results['min_y']:.2f}, {self.xy_results['max_y']:.2f}] mm\n"
        results_text += f"   范围宽度: {self.xy_results['area_width']:.2f} mm\n"
        results_text += f"   范围高度: {self.xy_results['area_height']:.2f} mm\n"
        results_text += f"   覆盖面积: {self.xy_results['area_size']:.2f} mm²\n\n"

        results_text += "2. 覆盖半径:\n"
        results_text += f"   最大半径: {self.xy_results['max_radius']:.2f} mm\n"
        results_text += f"   平均半径: {self.xy_results['avg_radius']:.2f} mm\n\n"

        results_text += "3. 备注:\n"
        results_text += "   - XY平面分析不考虑重力\n"
        results_text += "   - 磁场在XY平面内旋转\n"
        results_text += "   - 基座固定在原点\n"

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results_text)

    def update_xz_results(self):
        """更新XZ平面分析结果"""
        results_text = f"磁矩配置: {['向前' if a == 0 else '向后' for a in self.mpc_config]}\n"
        results_text += f"分析模式: XZ平面（垂直面）\n\n"

        results_text += "=== XZ平面性能指标 ===\n\n"
        results_text += "1. 抬升能力:\n"
        results_text += f"   头部高度: {self.xz_results['head_height']:.2f} mm\n"
        results_text += f"   最低点高度: {self.xz_results['min_height']:.2f} mm\n"
        results_text += f"   抬升高度: {self.xz_results['max_lift_height']:.2f} mm\n"
        results_text += f"   抬升效率: {self.xz_results['max_lift_height'] / (self.N_links * self.link_length_mm) * 100:.1f}%\n\n"

        results_text += "2. 爬坡能力:\n"
        results_text += f"   最大爬坡角度: {self.xz_results['max_climb_angle']:.1f}°\n\n"

        results_text += "3. 越障能力:\n"
        results_text += f"   最低点离地高度: {self.xz_results['clearance_height']:.2f} mm\n"

        if self.xz_results['clearance_height'] > 0.5:
            results_text += "   可通过小型障碍物\n\n"
        else:
            results_text += "   仅能在平坦表面移动\n\n"

        results_text += "4. 备注:\n"
        results_text += "   - XZ平面分析考虑重力\n"
        results_text += "   - 重力方向为-Z方向\n"
        results_text += "   - 地面在Z=0位置\n"

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results_text)

    def update_xy_plot(self):
        """更新XY平面图形"""
        self.ax.clear()

        if hasattr(self, 'xy_results'):
            points = self.xy_results['reachable_points']
            self.ax.scatter(points[:, 0], points[:, 1],
                            c='blue', alpha=0.6, s=20, label='可达点')

            # 绘制凸包
            from scipy.spatial import ConvexHull
            if len(points) >= 3:
                try:
                    hull = ConvexHull(points)
                    for simplex in hull.simplices:
                        self.ax.plot(points[simplex, 0], points[simplex, 1],
                                     'r-', alpha=0.5, linewidth=2)

                    # 填充凸包区域
                    hull_points = points[hull.vertices]
                    self.ax.fill(hull_points[:, 0], hull_points[:, 1],
                                 alpha=0.1, color='red', label='可达区域')
                except:
                    pass

            # 标记原点（基座位置）
            self.ax.plot(0, 0, 'ro', markersize=8, label='基座')

            self.ax.set_xlabel('X (mm)')
            self.ax.set_ylabel('Y (mm)')
            self.ax.set_title('XY平面：头部可达范围')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_aspect('equal')
            self.ax.legend()

        self.canvas.draw()

    def update_xz_plot(self):
        """更新XZ平面图形"""
        self.ax.clear()

        if hasattr(self, 'xz_results'):
            # 绘制地面
            self.ax.axhline(y=0, color='black', linestyle='-',
                            linewidth=2, alpha=0.7)
            self.ax.fill_between([-10, 10], -1, 0,
                                 color='gray', alpha=0.3, label='地面')

            # 绘制抬升姿态
            if hasattr(self.xz_results, 'lift_joints') or 'lift_joints' in self.xz_results:
                joints = self.xz_results.get('lift_joints', None)
                if joints is not None:
                    # 绘制机器人
                    for i in range(self.N_links):
                        self.ax.plot([joints[i, 0], joints[i + 1, 0]],
                                     [joints[i, 1], joints[i + 1, 1]],
                                     color='navy', linewidth=3, alpha=0.8)
                        self.ax.plot(joints[i, 0], joints[i, 1], 'o',
                                     color='darkred', markersize=6, alpha=0.8)

                    # 绘制头部
                    self.ax.plot(joints[-1, 0], joints[-1, 1], 'o',
                                 color='green', markersize=8, label='头部')

                    # 标记最低点
                    min_idx = np.argmin(joints[:, 1])
                    self.ax.plot(joints[min_idx, 0], joints[min_idx, 1], 'o',
                                 color='red', markersize=8, label='最低点')

                    # 绘制抬升高度线
                    if min_idx != len(joints) - 1:  # 最低点不是头部
                        self.ax.plot([joints[-1, 0], joints[-1, 0]],
                                     [joints[min_idx, 1], joints[-1, 1]],
                                     'r--', linewidth=2,
                                     label=f'抬升高度: {self.xz_results["max_lift_height"]:.2f} mm')

                    # 绘制重力方向箭头
                    self.ax.arrow(5, 2, 0, -1.5, head_width=0.3, head_length=0.5,
                                  fc='red', ec='red', label='重力方向')
                    self.ax.text(5.5, 1.5, '重力', color='red', fontsize=10)

            # 绘制爬坡姿态（如果有）
            if self.xz_results.get('climb_joints') is not None:
                joints_climb = self.xz_results['climb_joints']
                # 用虚线绘制爬坡姿态
                for i in range(self.N_links):
                    self.ax.plot([joints_climb[i, 0], joints_climb[i + 1, 0]],
                                 [joints_climb[i, 1], joints_climb[i + 1, 1]],
                                 '--', color='orange', linewidth=2, alpha=0.6)

                self.ax.plot(joints_climb[-1, 0], joints_climb[-1, 1], 'o',
                             color='orange', markersize=6,
                             label=f'爬坡 {self.xz_results["max_climb_angle"]:.1f}°')

            self.ax.set_xlabel('X (mm)')
            self.ax.set_ylabel('Z (mm)')
            self.ax.set_title('XZ平面：考虑重力的姿态')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_aspect('equal')
            self.ax.legend()
            self.ax.set_xlim([-5, 5])
            self.ax.set_ylim([-1, 8])

        self.canvas.draw()

    def create_widgets(self):
        """创建GUI控件"""
        # 左侧面板 - 配置和控制
        left_frame = ttk.Frame(self.root, padding="10")
        left_frame.grid(row=0, column=0, sticky="nsew")

        # 分析模式选择
        mode_frame = ttk.LabelFrame(left_frame, text="分析模式", padding="10")
        mode_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        ttk.Radiobutton(mode_frame, text="XY平面分析（水平面）",
                        variable=self.analysis_mode, value="XY").pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="XZ平面分析（垂直面）",
                        variable=self.analysis_mode, value="XZ").pack(anchor=tk.W)

        # 磁矩配置区域
        config_frame = ttk.LabelFrame(left_frame, text="磁矩配置 (6个连杆)", padding="10")
        config_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        # 创建6个磁矩选择器
        for i in range(6):
            frame = ttk.Frame(config_frame)
            frame.grid(row=i // 2, column=i % 2, sticky="w", padx=5, pady=5)

            ttk.Label(frame, text=f"连杆 {i + 1}:").pack(side=tk.LEFT)

            var = tk.StringVar(value="向前")
            setattr(self, f'magnet_var_{i}', var)

            combo = ttk.Combobox(frame, textvariable=var,
                                 values=["向前", "向后"],
                                 state="readonly", width=8)
            combo.pack(side=tk.LEFT, padx=5)

        # 配置显示
        self.config_label = ttk.Label(left_frame, text="当前配置: [向前, 向后, 向前, 向后, 向前, 向后]")
        self.config_label.grid(row=2, column=0, pady=(0, 10))

        # 预设配置
        preset_frame = ttk.LabelFrame(left_frame, text="预设配置", padding="10")
        preset_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))

        ttk.Button(preset_frame, text="交替磁化",
                   command=lambda: self.set_preset([0, np.pi, 0, np.pi, 0, np.pi])).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="全向前",
                   command=lambda: self.set_preset([0, 0, 0, 0, 0, 0])).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="全向后",
                   command=lambda: self.set_preset([np.pi] * 6)).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="前3后3",
                   command=lambda: self.set_preset([0, 0, 0, np.pi, np.pi, np.pi])).pack(side=tk.LEFT, padx=2)

        # 参数设置
        param_frame = ttk.LabelFrame(left_frame, text="机器人参数", padding="10")
        param_frame.grid(row=4, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(param_frame, text="连杆长度 (mm):").grid(row=0, column=0, sticky="w")
        self.length_var = tk.DoubleVar(value=1.5)
        ttk.Entry(param_frame, textvariable=self.length_var, width=10).grid(row=0, column=1, padx=5)

        ttk.Label(param_frame, text="磁场强度 (mT):").grid(row=1, column=0, sticky="w")
        self.B_var = tk.DoubleVar(value=20.0)
        ttk.Entry(param_frame, textvariable=self.B_var, width=10).grid(row=1, column=1, padx=5)

        # 控制按钮
        self.analyze_button = ttk.Button(left_frame, text="开始分析", command=self.analyze)
        self.analyze_button.grid(row=5, column=0, pady=(10, 5))

        # 结果文本框
        result_frame = ttk.LabelFrame(left_frame, text="分析结果", padding="10")
        result_frame.grid(row=6, column=0, sticky="nsew", pady=(0, 10))

        self.results_text = tk.Text(result_frame, height=20, width=40, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(result_frame, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)

        self.results_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # 右侧面板 - 图形显示
        right_frame = ttk.Frame(self.root, padding="10")
        right_frame.grid(row=0, column=1, sticky="nsew")

        # 创建matplotlib图形
        self.fig = Figure(figsize=(8, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, right_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)

        # 添加模式说明标签
        mode_info = ttk.Label(right_frame, text="XY平面：水平面，不考虑重力\nXZ平面：垂直面，考虑重力",
                              justify=tk.LEFT)
        mode_info.grid(row=1, column=0, pady=(5, 0))

        # 配置网格权重
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        left_frame.grid_columnconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

    def set_preset(self, config):
        """设置预设配置"""
        for i, angle in enumerate(config):
            var = getattr(self, f'magnet_var_{i}')
            var.set("向前" if angle == 0 else "向后")

        self.update_magnet_config()
        messagebox.showinfo("提示", "预设配置已应用，请点击'开始分析'进行计算")

    def run(self):
        """运行GUI"""
        self.root.mainloop()


# 快速使用函数
def quick_analysis():
    """快速分析函数"""
    print("启动磁控铰链机器人GUI分析器...")
    app = MagneticRobotAnalyzer()
    app.run()


if __name__ == "__main__":
    quick_analysis()
