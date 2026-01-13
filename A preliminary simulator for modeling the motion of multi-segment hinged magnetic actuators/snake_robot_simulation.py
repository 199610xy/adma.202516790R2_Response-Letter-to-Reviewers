import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv
import tkinter as tk
from tkinter import ttk
import threading
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class RobustMicroSnake:
    def __init__(self,
                 n_links=6,
                 link_length_mm=1.5,
                 width_mm=0.4,
                 height_mm=0.4,
                 magnetization_T=1.2,
                 fluid_viscosity=1e-3):

        self.N_links = n_links
        self.N_joints = n_links - 1
        self.N_dof = 3 + self.N_joints

        self.L = link_length_mm
        self.w = width_mm
        self.h = height_mm
        self.viscosity = fluid_viscosity

        self.sim_factor = 50e-6

        mu0 = 4 * np.pi * 1e-7
        M_SI = magnetization_T / mu0
        vol_SI = (self.L * 1e-3) * (self.w * 1e-3) * (self.h * 1e-3)
        m_SI = M_SI * vol_SI

        self.magnetic_moment_scaled = (m_SI * 1e9) * self.sim_factor

        self.max_angle = np.deg2rad(70)
        self.barrier_stiffness = 1.0

        eff_radius = np.sqrt((self.w * self.h) / np.pi)
        ratio = max(self.L / eff_radius, 2.0)
        epsilon = np.log(ratio)

        self.c_perp = (4 * np.pi * self.viscosity) / (epsilon + 0.5) * self.L
        self.c_para = (2 * np.pi * self.viscosity) / (epsilon - 0.5) * self.L
        self.c_rot = 0.05 * self.c_perp * (self.L ** 2)

    def _get_kinematics(self, state):
        x_b, y_b, th_b = state[0], state[1], state[2]
        joint_angles = np.clip(state[3:], -np.pi / 2, np.pi / 2)

        abs_angles = np.cumsum(np.concatenate(([th_b], joint_angles)))

        joints_pos = [np.array([x_b, y_b])]
        link_centers = []
        curr_x, curr_y = x_b, y_b

        for i in range(self.N_links):
            dx = self.L * np.cos(abs_angles[i])
            dy = self.L * np.sin(abs_angles[i])
            link_centers.append(np.array([curr_x + 0.5 * dx, curr_y + 0.5 * dy]))
            curr_x += dx;
            curr_y += dy
            joints_pos.append(np.array([curr_x, curr_y]))

        return np.array(link_centers), np.array(joints_pos), abs_angles

    def _compute_dynamics(self, state, B_field_mT, mpc_config):
        B_field = B_field_mT * 1e-3
        link_centers, joints, abs_angles = self._get_kinematics(state)

        R_mat = np.zeros((self.N_dof, self.N_dof))
        F_vec = np.zeros(self.N_dof)

        for i in range(self.N_links):
            J_v = np.zeros((2, self.N_dof))
            J_w = np.zeros((1, self.N_dof))

            J_v[0, 0] = 1;
            J_v[1, 1] = 1
            rx, ry = link_centers[i] - joints[0]
            J_v[0, 2] = -ry;
            J_v[1, 2] = rx;
            J_w[0, 2] = 1

            for k in range(self.N_joints):
                if i >= (k + 1):
                    j_pos = joints[k + 1]
                    rx_j, ry_j = link_centers[i] - j_pos
                    J_v[0, 3 + k] = -ry_j
                    J_v[1, 3 + k] = rx_j
                    J_w[0, 3 + k] = 1

            theta = abs_angles[i]
            c, s = np.cos(theta), np.sin(theta)
            Rot = np.array([[c, -s], [s, c]])
            D_local = np.diag([self.c_para, self.c_perp])
            D_global = Rot @ D_local @ Rot.T

            R_mat += J_v.T @ D_global @ J_v
            R_mat += J_w.T * self.c_rot * J_w

            m_angle = theta + mpc_config[i]
            mx = self.magnetic_moment_scaled * np.cos(m_angle)
            my = self.magnetic_moment_scaled * np.sin(m_angle)
            tau_mag = mx * B_field[1] - my * B_field[0]
            F_vec += J_w.flatten() * tau_mag

        curr_joints = state[3:]
        for k in range(self.N_joints):
            angle = curr_joints[k]
            if angle > self.max_angle:
                F_vec[3 + k] -= self.barrier_stiffness * (angle - self.max_angle)
            elif angle < -self.max_angle:
                F_vec[3 + k] -= self.barrier_stiffness * (angle + self.max_angle)

        if np.isnan(R_mat).any():
            return np.zeros(self.N_dof)

        R_mat += np.eye(self.N_dof) * 1e-6

        velocity = np.dot(pinv(R_mat), F_vec)

        max_vel = 50.0
        velocity = np.clip(velocity, -max_vel, max_vel)

        return velocity

    def step_rk4(self, state, B_field_mT, mpc_config, dt):
        k1 = self._compute_dynamics(state, B_field_mT, mpc_config)
        k2 = self._compute_dynamics(state + 0.5 * dt * k1, B_field_mT, mpc_config)
        k3 = self._compute_dynamics(state + 0.5 * dt * k2, B_field_mT, mpc_config)
        k4 = self._compute_dynamics(state + dt * k3, B_field_mT, mpc_config)

        new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        new_state[3:] = np.clip(new_state[3:], -np.pi / 1.8, np.pi / 1.8)
        return new_state


class MicroSnakeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Micro Snake 仿真控制面板 - 增强版")
        self.root.geometry("1300x900")

        # 默认参数
        self.params = {
            'n_links': tk.IntVar(value=6),
            'link_length_mm': tk.DoubleVar(value=1.5),
            'width_mm': tk.DoubleVar(value=0.4),
            'height_mm': tk.DoubleVar(value=0.4),
            'magnetization_T': tk.DoubleVar(value=1.2),
            'fluid_viscosity': tk.DoubleVar(value=1e-3),
            'B_amp_mT': tk.DoubleVar(value=15.0),
            'freq_Hz': tk.DoubleVar(value=0.1),
            'sim_time': tk.DoubleVar(value=10.0),
            'dt': tk.DoubleVar(value=0.005),
            'max_angle_deg': tk.DoubleVar(value=70.0),
            'barrier_stiffness': tk.DoubleVar(value=1.0),  # 这里添加了逗号
            'field_mode': tk.StringVar(value="full_rotation"),  # 'full_rotation' 或 'limited_swing'
            'swing_start_deg': tk.DoubleVar(value=0.0),
            'swing_end_deg': tk.DoubleVar(value=90.0),
        }

        # 磁矩模式
        self.magnet_mode = tk.StringVar(value="alternating")
        self.custom_magnet_angles = tk.StringVar(value="")

        self.simulation_running = False
        self.setup_gui()

    def setup_gui(self):
        # 左侧控制面板
        control_frame = ttk.LabelFrame(self.root, text="仿真参数", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # 结构参数
        ttk.Label(control_frame, text="机器人结构参数", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=2,
                                                                                         pady=(0, 10))

        param_defs = [
            ("链接数量:", 'n_links', 1, 20),
            ("链接长度(mm):", 'link_length_mm', 0.1, 10.0),
            ("宽度(mm):", 'width_mm', 0.1, 2.0),
            ("高度(mm):", 'height_mm', 0.1, 2.0),
            ("磁化强度(T):", 'magnetization_T', 0.1, 5.0),
            ("流体粘度(Pa·s):", 'fluid_viscosity', 1e-5, 1e-2),
            ("磁场幅度(mT):", 'B_amp_mT', 1.0, 100.0),
            ("磁场频率(Hz):", 'freq_Hz', 0.01, 10.0),
            ("仿真时间(s):", 'sim_time', 1.0, 100.0),
            ("时间步长(s):", 'dt', 0.001, 0.1),
            ("最大关节角(度):", 'max_angle_deg', 10.0, 180.0),
            ("关节刚度:", 'barrier_stiffness', 0.1, 10.0)
        ]

        for i, (label, param, min_val, max_val) in enumerate(param_defs):
            ttk.Label(control_frame, text=label).grid(row=i + 1, column=0, sticky=tk.W, pady=2)

            if 'int' in str(type(self.params[param])):
                entry = ttk.Spinbox(control_frame, from_=int(min_val), to=int(max_val),
                                    textvariable=self.params[param], width=15)
            else:
                entry = ttk.Spinbox(control_frame, from_=min_val, to=max_val,
                                    textvariable=self.params[param], width=15, increment=(max_val - min_val) / 100)
            entry.grid(row=i + 1, column=1, pady=2, padx=(5, 0))

        # 磁场模式控制部分
        ttk.Label(control_frame, text="磁场模式设置", font=('Arial', 10, 'bold')).grid(
            row=len(param_defs) + 2, column=0, columnspan=2, pady=(20, 10), sticky=tk.W
        )

        # 磁场模式选择
        row_idx = len(param_defs) + 3
        ttk.Label(control_frame, text="磁场模式:").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        mode_combo = ttk.Combobox(control_frame, textvariable=self.params['field_mode'],
                                  values=["full_rotation", "limited_swing"], state="readonly", width=15)
        mode_combo.grid(row=row_idx, column=1, pady=2, padx=(5, 0))

        # 摆动角度范围
        row_idx += 1
        ttk.Label(control_frame, text="摆动起始角(度):").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        start_spin = ttk.Spinbox(control_frame, from_=-180.0, to=180.0,
                                 textvariable=self.params['swing_start_deg'], width=15)
        start_spin.grid(row=row_idx, column=1, pady=2, padx=(5, 0))

        row_idx += 1
        ttk.Label(control_frame, text="摆动终止角(度):").grid(row=row_idx, column=0, sticky=tk.W, pady=2)
        end_spin = ttk.Spinbox(control_frame, from_=-180.0, to=180.0,
                               textvariable=self.params['swing_end_deg'], width=15)
        end_spin.grid(row=row_idx, column=1, pady=2, padx=(5, 0))

        # 根据选择的模式启用/禁用角度输入框
        def toggle_swing_fields(*args):
            state = 'normal' if self.params['field_mode'].get() == 'limited_swing' else 'disabled'
            start_spin.config(state=state)
            end_spin.config(state=state)

        self.params['field_mode'].trace('w', toggle_swing_fields)
        toggle_swing_fields()  # 初始化状态

        # 磁矩方向控制部分
        ttk.Label(control_frame, text="磁矩方向设置", font=('Arial', 10, 'bold')).grid(
            row=row_idx + 1, column=0, columnspan=2, pady=(20, 10), sticky=tk.W
        )

        # 磁矩模式选择
        magnet_mode_frame = ttk.Frame(control_frame)
        magnet_mode_frame.grid(row=row_idx + 2, column=0, columnspan=2, sticky=tk.W, pady=5)

        ttk.Radiobutton(magnet_mode_frame, text="交替磁化 (0°, 180°)",
                        variable=self.magnet_mode, value="alternating").pack(anchor=tk.W)
        ttk.Radiobutton(magnet_mode_frame, text="同向磁化 (全部0°)",
                        variable=self.magnet_mode, value="uniform").pack(anchor=tk.W)
        ttk.Radiobutton(magnet_mode_frame, text="正弦波分布",
                        variable=self.magnet_mode, value="sinusoidal").pack(anchor=tk.W)
        ttk.Radiobutton(magnet_mode_frame, text="线性梯度",
                        variable=self.magnet_mode, value="gradient").pack(anchor=tk.W)
        ttk.Radiobutton(magnet_mode_frame, text="自定义角度",
                        variable=self.magnet_mode, value="custom").pack(anchor=tk.W)

        # 自定义角度输入
        custom_frame = ttk.Frame(control_frame)
        custom_frame.grid(row=row_idx + 3, column=0, columnspan=2, sticky=tk.W, pady=5)

        ttk.Label(custom_frame, text="自定义角度(度,逗号分隔):").pack(side=tk.LEFT, padx=(0, 5))
        custom_entry = ttk.Entry(custom_frame, textvariable=self.custom_magnet_angles, width=25)
        custom_entry.pack(side=tk.LEFT)

        # 磁矩预览按钮
        preview_btn = ttk.Button(control_frame, text="预览磁矩方向", command=self.preview_magnetization)
        preview_btn.grid(row=row_idx + 4, column=0, columnspan=2, pady=(10, 5))

        # 磁矩预览图区域
        self.magnet_preview_frame = ttk.Frame(control_frame)
        self.magnet_preview_frame.grid(row=row_idx + 5, column=0, columnspan=2, pady=5)

        # 控制按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=row_idx + 6, column=0, columnspan=2, pady=(20, 0))

        self.start_btn = ttk.Button(button_frame, text="开始仿真", command=self.start_simulation)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="重置参数", command=self.reset_params).pack(side=tk.LEFT, padx=5)

        # 状态显示
        ttk.Separator(control_frame, orient='horizontal').grid(row=row_idx + 7, column=0, columnspan=2, pady=10,
                                                               sticky=tk.EW)

        self.status_label = ttk.Label(control_frame, text="就绪", foreground="blue")
        self.status_label.grid(row=row_idx + 8, column=0, columnspan=2, pady=5)

        # 结果显示
        ttk.Label(control_frame, text="仿真结果", font=('Arial', 10, 'bold')).grid(row=row_idx + 9, column=0,
                                                                                   columnspan=2, pady=(10, 0))

        self.result_text = tk.Text(control_frame, height=8, width=25)
        self.result_text.grid(row=row_idx + 10, column=0, columnspan=2, pady=5)

        # 中间磁矩预览大图区域
        preview_frame = ttk.LabelFrame(self.root, text="磁矩方向预览", padding=10)
        preview_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.magnet_fig = Figure(figsize=(4, 5), dpi=80)
        self.magnet_canvas = FigureCanvasTkAgg(self.magnet_fig, master=preview_frame)
        self.magnet_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 右侧图形显示区域
        plot_frame = ttk.LabelFrame(self.root, text="仿真结果图形", padding=10)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 初始预览
        self.preview_magnetization()

    def get_magnetization_config(self):
        """根据选择的模式生成磁矩配置"""
        n_links = self.params['n_links'].get()
        mode = self.magnet_mode.get()

        if mode == "alternating":  # 交替磁化 (0°, 180°交替)
            config = np.array([0 if i % 2 == 0 else np.pi for i in range(n_links)])

        elif mode == "uniform":  # 同向磁化 (全部0°)
            config = np.zeros(n_links)

        elif mode == "sinusoidal":  # 正弦波分布
            config = np.pi * np.sin(np.linspace(0, 2 * np.pi, n_links))

        elif mode == "gradient":  # 线性梯度
            config = np.linspace(0, np.pi, n_links)

        elif mode == "custom":  # 自定义角度
            custom_str = self.custom_magnet_angles.get()
            if custom_str:
                try:
                    angles_deg = [float(x.strip()) for x in custom_str.split(',')]
                    if len(angles_deg) != n_links:
                        raise ValueError(f"需要 {n_links} 个角度值，但输入了 {len(angles_deg)} 个")
                    config = np.deg2rad(angles_deg)
                except ValueError as e:
                    self.status_label.config(text=f"自定义角度错误: {str(e)}", foreground="red")
                    return None
            else:
                # 如果没有自定义输入，使用默认交替模式
                config = np.array([0 if i % 2 == 0 else np.pi for i in range(n_links)])
        else:
            config = np.array([0 if i % 2 == 0 else np.pi for i in range(n_links)])

        return config

    def preview_magnetization(self):
        """预览磁矩方向"""
        n_links = self.params['n_links'].get()
        config = self.get_magnetization_config()

        if config is None:
            return

        self.magnet_fig.clear()
        ax = self.magnet_fig.add_subplot(111)

        # 绘制链接和磁矩箭头
        for i in range(n_links):
            # 链接位置
            x = i * 0.5
            y = 0

            # 绘制链接
            ax.plot([x, x + 0.4], [y, y], 'k-', linewidth=3, alpha=0.7)

            # 磁矩角度
            angle_deg = np.rad2deg(config[i])

            # 绘制磁矩箭头
            dx = 0.3 * np.cos(config[i])
            dy = 0.3 * np.sin(config[i])

            ax.arrow(x + 0.2, y, dx, dy, head_width=0.08, head_length=0.08,
                     fc='red', ec='red', alpha=0.8)

            # 添加角度标签
            ax.text(x + 0.2, y + 0.25, f'{angle_deg:.1f}°',
                    ha='center', va='center', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

            # 添加链接编号
            ax.text(x + 0.2, y - 0.15, f'链接 {i}',
                    ha='center', va='center', fontsize=7, fontweight='bold')

        ax.set_xlim(-0.5, n_links * 0.5 + 0.5)
        ax.set_ylim(-0.5, 0.8)
        ax.set_aspect('equal')
        ax.set_title(f'磁矩方向预览 (模式: {self.magnet_mode.get()})')
        ax.set_xlabel('链接位置')
        ax.set_ylabel('磁矩方向')
        ax.grid(True, alpha=0.3)

        # 添加图例
        from matplotlib.patches import FancyArrowPatch
        arrow = FancyArrowPatch((0, 0.6), (0.3, 0.6),
                                arrowstyle='->', color='red', linewidth=2)
        ax.add_patch(arrow)
        ax.text(0.35, 0.6, '磁矩方向', va='center')

        self.magnet_canvas.draw()

    def reset_params(self):
        # 重置为默认值
        default_values = {
            'n_links': 6,
            'link_length_mm': 1.5,
            'width_mm': 0.4,
            'height_mm': 0.4,
            'magnetization_T': 1.2,
            'fluid_viscosity': 1e-3,
            'B_amp_mT': 15.0,
            'freq_Hz': 0.1,
            'sim_time': 10.0,
            'dt': 0.005,
            'max_angle_deg': 70.0,
            'barrier_stiffness': 1.0,
            'field_mode': "full_rotation",
            'swing_start_deg': 0.0,
            'swing_end_deg': 90.0,
        }

        for key, value in default_values.items():
            if isinstance(self.params[key], tk.IntVar):
                self.params[key].set(int(value))
            else:
                self.params[key].set(value)

        self.magnet_mode.set("alternating")
        self.custom_magnet_angles.set("")

        self.status_label.config(text="参数已重置", foreground="green")
        self.preview_magnetization()

    def start_simulation(self):
        if self.simulation_running:
            return

        self.simulation_running = True
        self.start_btn.config(state='disabled')
        self.status_label.config(text="仿真运行中...", foreground="orange")

        # 在新线程中运行仿真
        sim_thread = threading.Thread(target=self.run_simulation)
        sim_thread.daemon = True
        sim_thread.start()

    def run_simulation(self):
        try:
            # 获取参数
            n_links = self.params['n_links'].get()
            link_length_mm = self.params['link_length_mm'].get()
            width_mm = self.params['width_mm'].get()
            height_mm = self.params['height_mm'].get()
            magnetization_T = self.params['magnetization_T'].get()
            fluid_viscosity = self.params['fluid_viscosity'].get()
            B_amp_mT = self.params['B_amp_mT'].get()
            freq_Hz = self.params['freq_Hz'].get()
            sim_time = self.params['sim_time'].get()
            dt = self.params['dt'].get()

            # 获取磁矩配置
            mpc_config = self.get_magnetization_config()
            if mpc_config is None:
                self.root.after(0, self.simulation_error, "磁矩配置错误")
                return

            # 创建机器人实例
            robot = RobustMicroSnake(
                n_links=n_links,
                link_length_mm=link_length_mm,
                width_mm=width_mm,
                height_mm=height_mm,
                magnetization_T=magnetization_T,
                fluid_viscosity=fluid_viscosity
            )

            # 设置最大关节角
            robot.max_angle = np.deg2rad(self.params['max_angle_deg'].get())
            robot.barrier_stiffness = self.params['barrier_stiffness'].get()

            steps = int(sim_time / dt)

            state = np.zeros(robot.N_dof)
            state[3:] = np.random.uniform(-0.05, 0.05, size=robot.N_joints)

            omega = 2 * np.pi * freq_Hz

            traj_x, traj_y = [], []
            snapshots = []

            for k in range(steps):
                t = k * dt
                B_amp = self.params['B_amp_mT'].get()

                field_mode = self.params['field_mode'].get()

                if field_mode == 'full_rotation':
                    # 原始完整旋转模式
                    Bx = B_amp * np.cos(omega * t)
                    By = B_amp * np.sin(omega * t)
                elif field_mode == 'limited_swing':
                    # 角度受限摆动模式
                    start_rad = np.deg2rad(self.params['swing_start_deg'].get())
                    end_rad = np.deg2rad(self.params['swing_end_deg'].get())
                    # 确保 start_rad <= end_rad
                    if start_rad > end_rad:
                        start_rad, end_rad = end_rad, start_rad
                    swing_range = end_rad - start_rad
                    # 使用正弦函数在指定角度区间内往复摆动
                    # 公式解释：sin(omega*t) 在 [-1,1] 之间变化，映射到 [0,1]，再缩放到目标角度范围，最后加上起始角度偏移。
                    field_angle = start_rad + swing_range * (np.sin(omega * t) * 0.5 + 0.5)
                    Bx = B_amp * np.cos(field_angle)
                    By = B_amp * np.sin(field_angle)
                else:
                    # 默认回退到完整旋转
                    Bx = B_amp * np.cos(omega * t)
                    By = B_amp * np.sin(omega * t)

                state = robot.step_rk4(state, np.array([Bx, By]), mpc_config, dt)

                traj_x.append(state[0])
                traj_y.append(state[1])

                if k % 100 == 0:
                    _, joints_pos, _ = robot._get_kinematics(state)
                    snapshots.append(joints_pos)

            # 计算位移
            displacement = np.linalg.norm([traj_x[-1] - traj_x[0], traj_y[-1] - traj_y[0]])

            # 更新结果文本
            result_info = f"仿真完成！\n"
            result_info += f"总位移: {displacement:.2f} mm\n"
            result_info += f"机器人结构:\n"
            result_info += f"  链接数: {n_links}\n"
            result_info += f"  长度: {link_length_mm} mm\n"
            result_info += f"  截面: {width_mm}x{height_mm} mm\n"
            result_info += f"磁场参数:\n"
            result_info += f"  幅度: {B_amp_mT} mT\n"
            result_info += f"  频率: {freq_Hz} Hz\n"
            result_info += f"磁矩模式: {self.magnet_mode.get()}\n"
            result_info += f"磁场模式: {field_mode}\n"
            if field_mode == 'limited_swing':
                result_info += f"摆动角度范围: {self.params['swing_start_deg'].get()}° 到 {self.params['swing_end_deg'].get()}°\n"

            # 显示磁矩角度
            result_info += f"磁矩角度(度):\n"
            angles_deg = np.rad2deg(mpc_config)
            for i, angle in enumerate(angles_deg):
                result_info += f"  L{i}: {angle:.1f}°\n"

            self.root.after(0, self.update_results, result_info, traj_x, traj_y, snapshots, displacement, mpc_config)

        except Exception as e:
            self.root.after(0, self.simulation_error, str(e))

    def update_results(self, result_info, traj_x, traj_y, snapshots, displacement, mpc_config):
        # 更新文本结果
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, result_info)

        # 更新主图形
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        for i, shape in enumerate(snapshots):
            alpha = 0.2 + 0.8 * (i / len(snapshots))
            ax.plot(shape[:, 0], shape[:, 1], '.-', color='navy', alpha=alpha, lw=1)

        ax.plot(traj_x, traj_y, 'r-', lw=2, label='头部轨迹')
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(f"Micro Snake 仿真 - 位移: {displacement:.2f} mm")
        ax.legend()
        ax.grid(True)
        ax.axis('equal')

        # 添加磁矩信息
        magnet_mode = self.magnet_mode.get()
        field_mode = self.params['field_mode'].get()
        ax.text(0.02, 0.98, f"磁矩模式: {magnet_mode}\n磁场模式: {field_mode}",
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        self.canvas.draw()

        # 更新状态
        self.status_label.config(text="仿真完成", foreground="green")
        self.simulation_running = False
        self.start_btn.config(state='normal')

    def simulation_error(self, error_msg):
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, f"仿真错误:\n{error_msg}")
        self.status_label.config(text="仿真错误", foreground="red")
        self.simulation_running = False
        self.start_btn.config(state='normal')


def main():
    root = tk.Tk()
    app = MicroSnakeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
