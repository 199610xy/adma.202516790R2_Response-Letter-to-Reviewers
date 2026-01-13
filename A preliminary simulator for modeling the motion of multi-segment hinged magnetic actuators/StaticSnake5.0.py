import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import warnings
import itertools
import time

warnings.filterwarnings('ignore')

# 设置中文字体，防止乱码 (根据系统选择)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class MagneticRobotDesignerMatplotlib:
    """
    基于 Matplotlib Widgets 的交互式设计器
    适用于 PyCharm 直接运行

    新增功能：
    1. 每个连杆可以独立选择是否有磁矩
    2. 优化磁矩分布（哪些连杆应该有磁矩）
    3. 优化磁矩方向
    """

    def __init__(self):
        # 默认参数
        self.N = 5
        self.L = 1.0
        self.theta_max = np.deg2rad(60)
        self.B_angle = np.deg2rad(45)

        # deltas: 磁矩方向数组，delta_i 表示第i个磁铁相对于连杆的方向角度
        # 角度范围：-90°到90°
        self.deltas = np.zeros(self.N)

        # 新增：每个连杆是否有磁矩（True/False数组）
        self.has_magnet = [True] * self.N  # 默认全部有磁矩

        # 新增：存储所有磁场方向下的平衡姿态
        self.all_configurations = None

        # 计算缓存
        self.head_points = None
        self.sweep_length = 0  # 头部扫掠轨迹长度
        self.current_config = None

        # 绘图对象
        self.fig = None
        self.ax_robot = None
        self.ax_sweep = None
        self.sliders_delta = []
        self.switches_magnet = []  # 存储磁矩开关按钮
        self.text_info = None  # 信息文本框

    def solve_static_configuration(self, B_angle=None):
        """求解静态平衡，考虑哪些连杆有磁矩"""
        if B_angle is None:
            B_angle = self.B_angle
        B_angle = (B_angle + np.pi) % (2 * np.pi) - np.pi

        # 目标：最小化势能 (最大化磁矩在磁场方向上的投影)
        # 只有有磁矩的连杆对磁场有响应
        def objective(theta):
            total_energy = 0

            for i in range(self.N):
                if self.has_magnet[i]:  # 只有有磁矩的连杆才计算磁能
                    total_energy -= np.cos(theta[i] + self.deltas[i] - B_angle)

            return total_energy

        # 初始猜测：有磁矩的连杆尽量对齐磁场，无磁矩的连杆保持0度
        theta0 = np.zeros(self.N)
        for i in range(self.N):
            if self.has_magnet[i]:
                theta0[i] = B_angle - self.deltas[i]

        # 约束：相对角度限制（相邻关节的转角差不能超过theta_max）
        constraints = []
        for i in range(1, self.N):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, idx=i: self.theta_max - abs(x[idx] - x[idx - 1])
            })

        bounds = [(-np.pi, np.pi) for _ in range(self.N)]

        result = minimize(
            objective, theta0, bounds=bounds, constraints=constraints,
            method='SLSQP', options={'ftol': 1e-6, 'maxiter': 1000}
        )

        if result.success:
            self.current_config = result.x
        else:
            # 使用贪心算法作为备选
            self.current_config = self._greedy_solution(B_angle)

        return self.current_config

    def _greedy_solution(self, B_angle):
        """贪心算法求解，考虑磁矩存在性"""
        theta = np.zeros(self.N)

        # 第一个有磁矩的连杆
        first_magnet_idx = next((i for i in range(self.N) if self.has_magnet[i]), 0)
        if self.has_magnet[first_magnet_idx]:
            theta[first_magnet_idx] = B_angle - self.deltas[first_magnet_idx]

        for i in range(1, self.N):
            if self.has_magnet[i]:
                desired = B_angle - self.deltas[i]  # 磁矩期望的连杆方向
            else:
                desired = theta[i - 1]  # 无磁矩时尽量保持与前一个连杆相同的方向

            diff = desired - theta[i - 1]

            # 限制在关节限位内
            if abs(diff) > self.theta_max:
                diff = np.sign(diff) * self.theta_max

            theta[i] = theta[i - 1] + diff

        return theta

    def compute_head_sweep_length(self, num_samples=360):
        """计算头部扫掠轨迹长度"""
        points = []
        original_B = self.B_angle  # 保存当前磁场方向

        # 采样不同磁场方向下的头部位置
        for psi in np.linspace(0, 2 * np.pi, num_samples, endpoint=True):
            # 对于每个磁场方向psi，计算平衡姿态
            theta = self.solve_static_configuration(psi)
            # 计算头部位置（连杆末端位置）
            x = np.sum(self.L * np.cos(theta))
            y = np.sum(self.L * np.sin(theta))
            points.append([x, y])

        self.head_points = np.array(points)

        # 计算轨迹长度（连接所有点的多边形周长）
        if len(points) >= 2:
            total_length = 0
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                total_length += np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

            # 闭合轨迹：连接最后一个点和第一个点
            p1 = points[-1]
            p2 = points[0]
            total_length += np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

            self.sweep_length = total_length
        else:
            self.sweep_length = 0

        # 重置回当前磁场方向的解
        self.solve_static_configuration(original_B)
        return self.sweep_length

    def get_robot_shape(self):
        """获取关节点坐标"""
        if self.current_config is None:
            self.solve_static_configuration()

        theta = self.current_config
        positions = np.zeros((self.N + 1, 2))
        positions[0] = [0, 0]  # 基座

        for i in range(self.N):
            positions[i + 1] = [
                positions[i, 0] + self.L * np.cos(theta[i]),
                positions[i, 1] + self.L * np.sin(theta[i])
            ]
        return positions

    def get_head_position(self):
        """获取头部位置"""
        positions = self.get_robot_shape()
        return positions[-1]

    def compute_metrics(self):
        """计算头部运动指标"""
        if self.head_points is None or len(self.head_points) == 0:
            return 0, 0, 0, 0, 0

        # 计算所有头部点到基座的距离（半径）
        radii = np.sqrt(np.sum(self.head_points ** 2, axis=1))

        max_radius = np.max(radii)
        min_radius = np.min(radii)
        avg_radius = np.mean(radii)

        # 归一化变形率：衡量相对于自身尺度的伸缩能力
        if avg_radius > 1e-6:
            normalized_deformation = (max_radius - min_radius) / avg_radius
        else:
            normalized_deformation = 0

        return self.sweep_length, max_radius, min_radius, avg_radius, normalized_deformation

    def objective_function_for_optimization(self, deltas_array):
        """用于优化的目标函数（最小化），只优化有磁矩的连杆"""
        # 只更新有磁矩的连杆的方向
        for i in range(self.N):
            if self.has_magnet[i]:
                self.deltas[i] = deltas_array[i]

        self.compute_head_sweep_length(num_samples=90)  # 优化时使用较少采样
        _, _, _, _, score = self.compute_metrics()
        return -score  # 我们想要最大化变形率，但优化器是最小化，所以取负

    def objective_function_for_distribution(self, magnet_array):
        """用于优化磁矩分布的目标函数"""
        # magnet_array是0/1数组，表示每个连杆是否有磁矩
        # 同时也要优化磁矩方向

        # 设置磁矩分布
        self.has_magnet = [bool(m) for m in magnet_array[:self.N]]

        # 如果有磁矩的连杆，则使用对应的磁矩方向
        if len(magnet_array) > self.N:
            magnet_directions = magnet_array[self.N:]
            dir_idx = 0
            for i in range(self.N):
                if self.has_magnet[i]:
                    self.deltas[i] = magnet_directions[dir_idx]
                    dir_idx += 1
                else:
                    self.deltas[i] = 0

        self.compute_head_sweep_length(num_samples=90)
        _, _, _, _, score = self.compute_metrics()
        return -score  # 最小化负的变形率

    def normalize_delta_angle(self, deg_val):
        """将角度标准化到-90°到90°范围"""
        # 首先标准化到0-360°
        deg_val = deg_val % 360

        # 映射到-180°到180°
        if deg_val > 180:
            deg_val -= 360

        # 映射到-90°到90°
        if deg_val > 90:
            deg_val -= 180
        elif deg_val < -90:
            deg_val += 180

        return deg_val

    def optimize_deltas_continuous(self, event):
        """
        使用连续优化算法（差分进化）优化磁矩配置
        只优化有磁矩的连杆
        """
        print("=" * 50)
        print("开始连续优化磁矩配置...")
        print("优化算法：差分进化算法")
        print("优化目标：最大化归一化变形率 (R_max - R_min) / R_avg")
        print("磁矩角度范围：-90° 到 90°")
        print("=" * 50)

        start_time = time.time()

        # 保存当前配置以防优化失败
        original_deltas = self.deltas.copy()
        original_has_magnet = self.has_magnet.copy()

        # 只优化有磁矩的连杆
        magnet_indices = [i for i in range(self.N) if self.has_magnet[i]]

        if len(magnet_indices) == 0:
            print("没有磁矩需要优化！")
            return

        # 定义优化问题的边界（每个有磁矩的连杆的角度范围：-π/2到π/2弧度）
        bounds = [(-np.pi / 2, np.pi / 2) for _ in range(len(magnet_indices))]

        print(f"优化 {len(magnet_indices)} 个有磁矩的连杆...")
        print("正在运行差分进化算法...")

        try:
            # 定义临时目标函数
            def temp_objective(active_deltas):
                # 将优化变量放回完整的deltas数组
                temp_deltas = self.deltas.copy()
                for idx, magnet_idx in enumerate(magnet_indices):
                    temp_deltas[magnet_idx] = active_deltas[idx]

                self.deltas = temp_deltas
                self.compute_head_sweep_length(num_samples=90)
                _, _, _, _, score = self.compute_metrics()
                return -score

            # 使用差分进化算法进行全局优化
            result = differential_evolution(
                temp_objective,
                bounds=bounds,
                strategy='best1bin',
                maxiter=100,
                popsize=15,
                tol=0.01,
                disp=True,
                workers=1,
                updating='deferred'
            )

            if result.success:
                print(f"优化成功！耗时: {time.time() - start_time:.2f}秒")
                print(f"最优目标函数值: {-result.fun:.4f}")

                # 将优化结果转换为角度
                optimized_active_deltas = result.x

                # 将角度舍入到最近的5度倍数
                step_size = np.deg2rad(5)  # 5度对应的弧度
                optimized_active_deltas = np.round(optimized_active_deltas / step_size) * step_size

                # 确保在-90°到90°范围内
                for i in range(len(optimized_active_deltas)):
                    if optimized_active_deltas[i] < -np.pi / 2:
                        optimized_active_deltas[i] = -np.pi / 2
                    elif optimized_active_deltas[i] > np.pi / 2:
                        optimized_active_deltas[i] = np.pi / 2

                # 更新完整的deltas数组
                for idx, magnet_idx in enumerate(magnet_indices):
                    self.deltas[magnet_idx] = optimized_active_deltas[idx]

                # 更新滑块显示
                for i in range(min(self.N, len(self.sliders_delta))):
                    if self.has_magnet[i]:
                        deg_val = np.rad2deg(self.deltas[i])
                        # 确保在滑块范围内
                        if deg_val < -90:
                            deg_val = -90
                        elif deg_val > 90:
                            deg_val = 90
                        self.sliders_delta[i].set_val(deg_val)
                    else:
                        self.sliders_delta[i].set_val(0)

                print("优化后的磁矩配置（度）：")
                for i in range(self.N):
                    if self.has_magnet[i]:
                        deg_val = np.rad2deg(self.deltas[i])
                        print(f"  连杆{i + 1}: {deg_val:6.1f}° (有磁矩)")
                    else:
                        print(f"  连杆{i + 1}: 无磁矩")

                # 使用高精度重新计算并绘图
                self.compute_head_sweep_length(num_samples=360)
                self.update_plot()

            else:
                print("优化未收敛，恢复原配置")
                self.deltas = original_deltas
                self.has_magnet = original_has_magnet

        except Exception as e:
            print(f"优化过程中出现错误: {e}")
            print("恢复原配置")
            self.deltas = original_deltas
            self.has_magnet = original_has_magnet

    def optimize_deltas_discrete(self, event):
        """
        使用离散优化算法优化磁矩配置
        只优化有磁矩的连杆
        """
        print("=" * 50)
        print("开始离散优化磁矩配置...")
        print("优化算法：改进的局部搜索")
        print("优化目标：最大化归一化变形率 (R_max - R_min) / R_avg")
        print("磁矩角度范围：-90° 到 90°，步长：5度")
        print("=" * 50)

        start_time = time.time()

        # 保存当前配置
        original_deltas = self.deltas.copy()
        original_has_magnet = self.has_magnet.copy()

        # 只考虑有磁矩的连杆
        magnet_indices = [i for i in range(self.N) if self.has_magnet[i]]

        if len(magnet_indices) == 0:
            print("没有磁矩需要优化！")
            return

        best_deltas = original_deltas.copy()

        # 计算当前得分
        self.compute_head_sweep_length(num_samples=90)
        _, _, _, _, current_score = self.compute_metrics()
        best_score = current_score

        print(f"初始得分: {current_score:.4f}")

        # 定义角度步长（-90°到90°，步长5°）
        min_angle_deg = -90
        max_angle_deg = 90
        step_size_deg = 5
        num_steps = int((max_angle_deg - min_angle_deg) / step_size_deg) + 1  # 37步

        # 改进的局部搜索算法
        improved = True
        iteration = 0

        while improved and iteration < 10:  # 最多10轮迭代
            improved = False
            iteration += 1
            print(f"\n第{iteration}轮迭代...")

            for magnet_idx in magnet_indices:
                # 保存当前磁矩值
                original_delta_i = best_deltas[magnet_idx]
                best_delta_i = original_delta_i
                local_best_score = best_score

                # 测试当前磁矩的所有可能角度（-90°到90°，步长5°）
                for step in range(num_steps):
                    test_angle_deg = min_angle_deg + step * step_size_deg
                    test_angle = np.deg2rad(test_angle_deg)

                    # 设置测试角度
                    best_deltas[magnet_idx] = test_angle
                    self.deltas = best_deltas

                    # 计算得分
                    self.compute_head_sweep_length(num_samples=90)
                    _, _, _, _, test_score = self.compute_metrics()

                    # 如果得分更好，更新最佳角度
                    if test_score > local_best_score:
                        local_best_score = test_score
                        best_delta_i = test_angle
                        improved = True

                # 更新最佳磁矩值
                best_deltas[magnet_idx] = best_delta_i

                # 如果本地搜索改进了，更新全局最佳得分
                if local_best_score > best_score:
                    best_score = local_best_score

            print(f"  当前最佳得分: {best_score:.4f}")

        print(f"\n优化完成！耗时: {time.time() - start_time:.2f}秒")
        print(f"最终得分: {best_score:.4f} (提升: {(best_score - current_score) / current_score * 100:.1f}%)")

        # 应用优化结果
        self.deltas = best_deltas

        # 更新滑块显示
        for i in range(min(self.N, len(self.sliders_delta))):
            if self.has_magnet[i]:
                deg_val = np.rad2deg(best_deltas[i])
                # 确保在-90°到90°范围内显示
                if deg_val > 180:
                    deg_val -= 360
                if deg_val > 90:
                    deg_val -= 180
                elif deg_val < -90:
                    deg_val += 180
                self.sliders_delta[i].set_val(deg_val)
            else:
                self.sliders_delta[i].set_val(0)

        print("优化后的磁矩配置（度，已标准化到-90°到90°）：")
        for i in range(self.N):
            if self.has_magnet[i]:
                deg_val = np.rad2deg(self.deltas[i])
                # 标准化显示
                if deg_val > 180:
                    deg_val -= 360
                if deg_val > 90:
                    deg_val -= 180
                elif deg_val < -90:
                    deg_val += 180
                print(f"  连杆{i + 1}: {deg_val:6.1f}° (有磁矩)")
            else:
                print(f"  连杆{i + 1}: 无磁矩")

        # 使用高精度重新计算并绘图
        self.compute_head_sweep_length(num_samples=360)
        self.update_plot()

    def optimize_magnet_distribution(self, event):
        """
        优化磁矩分布：哪些连杆应该有磁矩
        同时优化磁矩方向
        """
        print("=" * 50)
        print("开始优化磁矩分布...")
        print("优化算法：差分进化 + 离散优化")
        print("优化目标：最大化归一化变形率 (R_max - R_min) / R_avg")
        print("=" * 50)

        start_time = time.time()

        # 保存当前配置
        original_deltas = self.deltas.copy()
        original_has_magnet = self.has_magnet.copy()

        # 方法1：使用启发式搜索
        print("使用启发式搜索...")

        # 首先尝试全部有磁矩
        best_has_magnet = [True] * self.N
        best_deltas = original_deltas.copy()

        # 优化全部有磁矩的情况
        self.has_magnet = best_has_magnet
        self.optimize_deltas_discrete(event)  # 使用离散优化方向
        self.compute_head_sweep_length(num_samples=90)
        _, _, _, _, best_score = self.compute_metrics()

        print(f"\n全部有磁矩的得分: {best_score:.4f}")

        # 尝试逐个关闭磁矩，看看是否改善
        improved = True
        iteration = 0

        while improved and iteration < 2:  # 最多2轮
            improved = False
            iteration += 1

            for i in range(self.N):
                if best_has_magnet[i]:
                    # 尝试关闭第i个磁矩
                    test_has_magnet = best_has_magnet.copy()
                    test_has_magnet[i] = False

                    # 如果有磁矩的连杆数大于0
                    if sum(test_has_magnet) > 0:
                        self.has_magnet = test_has_magnet
                        self.optimize_deltas_discrete(event)
                        self.compute_head_sweep_length(num_samples=90)
                        _, _, _, _, test_score = self.compute_metrics()

                        if test_score > best_score:
                            best_score = test_score
                            best_has_magnet = test_has_magnet
                            best_deltas = self.deltas.copy()
                            improved = True
                            print(f"关闭连杆 {i + 1} 的磁矩，得分提升到: {best_score:.4f}")

                    # 恢复最佳配置
                    self.has_magnet = best_has_magnet
                    self.deltas = best_deltas

        # 应用最佳配置
        self.has_magnet = best_has_magnet
        self.deltas = best_deltas

        # 更新磁矩开关显示
        for i, switch in enumerate(self.switches_magnet):
            if i < self.N:
                if self.has_magnet[i]:
                    switch.label.set_text('✓')
                    switch.ax.set_facecolor('lightgreen')
                else:
                    switch.label.set_text('✗')
                    switch.ax.set_facecolor('lightgray')

        # 更新滑块显示
        for i in range(min(self.N, len(self.sliders_delta))):
            if self.has_magnet[i]:
                deg_val = np.rad2deg(self.deltas[i])
                if deg_val < -90:
                    deg_val = -90
                elif deg_val > 90:
                    deg_val = 90
                self.sliders_delta[i].set_val(deg_val)
            else:
                self.sliders_delta[i].set_val(0)

        # 高精度重新计算并绘图
        self.compute_head_sweep_length(num_samples=360)
        self.update_plot()

        # 显示结果
        print("\n优化后的磁矩分布:")
        magnet_count = sum(self.has_magnet)
        for i in range(self.N):
            status = '有' if self.has_magnet[i] else '无'
            if self.has_magnet[i]:
                deg_val = np.rad2deg(self.deltas[i])
                deg_val = self.normalize_delta_angle(deg_val)
                print(f"  连杆{i + 1}: {status}磁矩, 方向: {deg_val:6.1f}°")
            else:
                print(f"  连杆{i + 1}: {status}磁矩")

        print(f"\n总计: {magnet_count}/{self.N} 个连杆有磁矩")
        print(f"最终归一化变形率: {best_score:.4f}")
        print(f"总耗时: {time.time() - start_time:.2f}秒")

    def toggle_magnet(self, event):
        """切换单个连杆的磁矩开关"""
        # 获取是哪个开关被点击
        for widget in event.inaxes.get_children():
            if hasattr(widget, 'link_index'):
                link_index = widget.link_index
                break
        else:
            return

        if link_index < self.N:
            # 切换磁矩状态
            self.has_magnet[link_index] = not self.has_magnet[link_index]

            # 更新开关显示
            if self.has_magnet[link_index]:
                event.inaxes.get_children()[0].label.set_text('✓')
                event.inaxes.set_facecolor('lightgreen')
            else:
                event.inaxes.get_children()[0].label.set_text('✗')
                event.inaxes.set_facecolor('lightgray')

            print(f"连杆 {link_index + 1} 磁矩: {'开启' if self.has_magnet[link_index] else '关闭'}")

            # 重新计算并更新图形
            self.update_plot()

        self.fig.canvas.draw_idle()

    def turn_all_magnets_on(self, event):
        """打开所有磁矩"""
        self.has_magnet = [True] * self.N

        # 更新所有开关显示
        for i, switch in enumerate(self.switches_magnet):
            if i < self.N:
                switch.label.set_text('✓')
                switch.ax.set_facecolor('lightgreen')

        print("所有磁矩已开启")
        self.update_plot()
        self.fig.canvas.draw_idle()

    def turn_all_magnets_off(self, event):
        """关闭所有磁矩"""
        self.has_magnet = [False] * self.N

        # 更新所有开关显示
        for i, switch in enumerate(self.switches_magnet):
            if i < self.N:
                switch.label.set_text('✗')
                switch.ax.set_facecolor('lightgray')

        print("所有磁矩已关闭")
        self.update_plot()
        self.fig.canvas.draw_idle()

    def update_deltas_from_sliders(self):
        """从滑块更新磁矩配置"""
        if len(self.deltas) != self.N:
            self.deltas = np.zeros(self.N)

        # 从滑块读取值并转换为弧度
        for i in range(min(self.N, len(self.sliders_delta))):
            deg_val = self.sliders_delta[i].val

            # 确保在-90°到90°范围内
            if deg_val < -90:
                deg_val = -90
                self.sliders_delta[i].set_val(deg_val)
            elif deg_val > 90:
                deg_val = 90
                self.sliders_delta[i].set_val(deg_val)

            # 转换为弧度
            rad_val = np.deg2rad(deg_val)
            self.deltas[i] = rad_val

    def update_plot(self, val=None):
        """更新绘图的回调函数"""
        new_N = int(self.slider_N.val)

        if new_N != self.N:
            self.N = new_N
            self.deltas = np.zeros(self.N)
            self.has_magnet = [True] * self.N
            self.update_deltas_from_sliders()

        self.theta_max = np.deg2rad(self.slider_theta.val)
        self.B_angle = np.deg2rad(self.slider_B.val)

        self.update_deltas_from_sliders()

        # 计算当前配置
        self.solve_static_configuration()
        self.compute_head_sweep_length()
        positions = self.get_robot_shape()

        # 清除重绘
        self.ax_robot.clear()
        self.ax_sweep.clear()

        # --- 左图：机器人形态 ---
        self.ax_robot.plot(positions[:, 0], positions[:, 1], 'b-o',
                           linewidth=3, markersize=8, markeredgecolor='black')

        # 绘制磁铁和磁矩方向
        for i in range(self.N):
            # 磁铁位置（在连杆的0.75L处）
            segment_vec = positions[i + 1] - positions[i]
            center = positions[i] + 0.75 * segment_vec

            if self.has_magnet[i]:
                # 有磁矩：绘制彩色箭头
                mag_angle = self.current_config[i] + self.deltas[i]
                dx = 0.3 * np.cos(mag_angle)
                dy = 0.3 * np.sin(mag_angle)

                # 将角度映射到颜色
                angle_deg = np.rad2deg(self.deltas[i])
                normalized_angle = (angle_deg + 90) / 180
                color = plt.cm.hsv(normalized_angle)

                self.ax_robot.arrow(center[0], center[1], dx, dy,
                                    head_width=0.15, head_length=0.2,
                                    fc=color, ec=color, alpha=0.8, zorder=5)

                # 在磁铁位置添加标记点
                self.ax_robot.plot(center[0], center[1], 's',
                                   markersize=10, markeredgecolor='black',
                                   markerfacecolor=color, alpha=0.7, zorder=4,
                                   label='有磁矩' if i == 0 else "")
            else:
                # 无磁矩：绘制灰色圆圈
                self.ax_robot.plot(center[0], center[1], 'o',
                                   markersize=12, markeredgecolor='gray',
                                   markerfacecolor='lightgray', alpha=0.5, zorder=3,
                                   label='无磁矩' if i == 0 else "")

        # 磁场方向
        field_len = self.N * self.L * 0.8
        self.ax_robot.arrow(0, 0,
                            field_len * np.cos(self.B_angle),
                            field_len * np.sin(self.B_angle),
                            width=0.05, head_width=0.15, head_length=0.2,
                            fc='orange', ec='orange', alpha=0.8,
                            label=f'磁场方向 ({np.rad2deg(self.B_angle):.0f}°)')

        limit = self.N * self.L * 1.2
        self.ax_robot.set_xlim(-limit, limit)
        self.ax_robot.set_ylim(-limit, limit)
        self.ax_robot.set_title(f"机器人形态 (N={self.N}, θ_max={np.rad2deg(self.theta_max):.0f}°)")
        self.ax_robot.grid(True, alpha=0.3)
        self.ax_robot.set_aspect('equal')
        self.ax_robot.legend(loc='upper right', fontsize=8)

        # --- 右图：头部扫掠轨迹 ---
        if self.head_points is not None and len(self.head_points) > 0:
            self.ax_sweep.plot(self.head_points[:, 0], self.head_points[:, 1],
                               'b-', linewidth=2, alpha=0.7, label='头部扫掠轨迹')

            self.ax_sweep.scatter(self.head_points[:, 0], self.head_points[:, 1],
                                  s=5, c='blue', alpha=0.5, label='采样点')

            current_head = positions[-1]
            self.ax_sweep.plot(current_head[0], current_head[1], 'r*',
                               markersize=15, label='当前头部位置')

            self.ax_sweep.plot(0, 0, 'ks', markersize=10, label='基座')

            # 显示指标
            length, max_r, min_r, avg_r, normalized_deformation = self.compute_metrics()

            info_text = f'头部扫掠长度: {length:.3f}\n'
            info_text += f'最大半径: {max_r:.3f}\n'
            info_text += f'最小半径: {min_r:.3f}\n'
            info_text += f'平均半径: {avg_r:.3f}\n'
            info_text += f'归一化变形率: {normalized_deformation:.3f}\n\n'

            # 显示当前磁矩配置
            info_text += '当前磁矩配置:\n'
            for i in range(min(self.N, 5)):
                if self.has_magnet[i]:
                    deg_val = np.rad2deg(self.deltas[i])
                    # 标准化显示到-90°到90°范围
                    deg_val = self.normalize_delta_angle(deg_val)
                    info_text += f'δ{i + 1}: {deg_val:6.1f}° ✓\n'
                else:
                    info_text += f'δ{i + 1}: 无磁矩 ✗\n'

            self.ax_sweep.text(0.02, 0.98, info_text,
                               transform=self.ax_sweep.transAxes,
                               verticalalignment='top',
                               fontsize=9,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        self.ax_sweep.set_xlim(-limit, limit)
        self.ax_sweep.set_ylim(-limit, limit)
        self.ax_sweep.set_title("头部扫掠轨迹")
        self.ax_sweep.grid(True, alpha=0.3)
        self.ax_sweep.set_aspect('equal')
        if self.head_points is not None and len(self.head_points) > 0:
            self.ax_sweep.legend(loc='lower left', fontsize=8)

        self.fig.canvas.draw_idle()

    def create_interface(self):
        """创建界面控件"""
        self.fig, (self.ax_robot, self.ax_sweep) = plt.subplots(1, 2, figsize=(14, 7))
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.35, top=0.9)

        axcolor = 'lightgoldenrodyellow'

        # 1. N 连杆数滑块
        ax_N = plt.axes([0.1, 0.25, 0.3, 0.03], facecolor=axcolor)
        self.slider_N = Slider(ax_N, '连杆数 N', 2, 8, valinit=self.N, valstep=1)

        # 2. 关节限位滑块
        ax_theta = plt.axes([0.1, 0.20, 0.3, 0.03], facecolor=axcolor)
        self.slider_theta = Slider(ax_theta, '关节限位(°)', 10, 120, valinit=60)

        # 3. 磁场方向滑块
        ax_B = plt.axes([0.1, 0.15, 0.3, 0.03], facecolor=axcolor)
        self.slider_B = Slider(ax_B, '磁场方向(°)', 0, 360, valinit=45)

        # 4. 磁矩配置滑块（范围-90°到90°，以5度为步长）
        self.sliders_delta = []
        for i in range(5):
            ax_d = plt.axes([0.55, 0.25 - i * 0.04, 0.3, 0.03], facecolor=axcolor)
            slider = Slider(ax_d, f'磁矩{i + 1}方向(°)', -90, 90, valinit=0, valstep=5)
            self.sliders_delta.append(slider)

        # 5. 磁矩开关（复选框样式）
        self.switches_magnet = []
        for i in range(5):  # 最多显示5个开关
            # 创建复选框位置
            ax_switch = plt.axes([0.88, 0.25 - i * 0.04, 0.04, 0.03])

            # 创建自定义的"开关"按钮
            switch = Button(ax_switch, '✓' if i < self.N and self.has_magnet[i] else '✗',
                            color='lightgreen' if i < self.N and self.has_magnet[i] else 'lightgray')

            # 为开关添加自定义属性
            switch.link_index = i  # 记录开关对应的连杆索引

            # 绑定点击事件
            switch.on_clicked(self.toggle_magnet)

            self.switches_magnet.append(switch)

        # 6. 连续优化按钮（使用差分进化）
        ax_opt_continuous = plt.axes([0.7, 0.05, 0.15, 0.04])
        self.btn_opt_continuous = Button(ax_opt_continuous, '连续优化磁矩', color='lightgreen', hovercolor='0.975')

        # 7. 离散优化按钮（使用局部搜索）
        ax_opt_discrete = plt.axes([0.45, 0.05, 0.2, 0.04])
        self.btn_opt_discrete = Button(ax_opt_discrete, '离散优化 (5°步长)', color='lightblue', hovercolor='0.975')

        # 8. 优化磁矩分布按钮
        ax_opt_dist = plt.axes([0.1, 0.05, 0.25, 0.04])
        self.btn_opt_distribution = Button(ax_opt_dist, '优化磁矩分布', color='gold', hovercolor='0.975')

        # 9. 全部开启/关闭按钮
        ax_all_on = plt.axes([0.7, 0.01, 0.06, 0.04])
        self.btn_all_on = Button(ax_all_on, '全开', color='lightgreen', hovercolor='0.975')

        ax_all_off = plt.axes([0.77, 0.01, 0.06, 0.04])
        self.btn_all_off = Button(ax_all_off, '全关', color='lightgray', hovercolor='0.975')

        # 10. 重置按钮
        ax_reset = plt.axes([0.25, 0.01, 0.15, 0.04])
        self.btn_reset = Button(ax_reset, '重置所有参数', color='lightcoral', hovercolor='0.975')

        # 绑定事件
        self.slider_N.on_changed(self.update_plot)
        self.slider_theta.on_changed(self.update_plot)
        self.slider_B.on_changed(self.update_plot)
        for s in self.sliders_delta:
            s.on_changed(self.update_plot)

        self.btn_opt_continuous.on_clicked(self.optimize_deltas_continuous)
        self.btn_opt_discrete.on_clicked(self.optimize_deltas_discrete)
        self.btn_opt_distribution.on_clicked(self.optimize_magnet_distribution)
        self.btn_all_on.on_clicked(self.turn_all_magnets_on)
        self.btn_all_off.on_clicked(self.turn_all_magnets_off)
        self.btn_reset.on_clicked(self.reset_parameters)

        self.update_plot()
        return self.fig

    def reset_parameters(self, event):
        """重置所有参数"""
        print("重置所有参数...")
        self.slider_N.set_val(5)
        self.slider_theta.set_val(60)
        self.slider_B.set_val(45)
        for slider in self.sliders_delta:
            slider.set_val(0)

        # 重置磁矩开关
        self.has_magnet = [True] * self.N
        for i, switch in enumerate(self.switches_magnet):
            if i < self.N:
                switch.label.set_text('✓')
                switch.ax.set_facecolor('lightgreen')

        self.update_plot()

    def show(self):
        """显示界面"""
        self.create_interface()
        plt.show()


# --- 运行主程序 ---
if __name__ == "__main__":
    print("磁控软体机器人设计器（支持磁矩开关）")
    print("=" * 70)
    print("功能说明:")
    print("  1. 每个连杆可以独立选择是否有磁矩（右侧✓/✗按钮）")
    print("  2. 优化磁矩方向（连续/离散优化）")
    print("  3. 优化磁矩分布（哪些连杆应该有磁矩）")
    print("  4. 全开/全关磁矩按钮")
    print()
    print("磁矩配置说明:")
    print("  1. delta_i 表示第i个磁铁相对于连杆的方向角")
    print("  2. 角度范围：-90° 到 90°")
    print("  3. 优化以5度为步长在-90°到90°范围内搜索")
    print("  4. 磁矩箭头颜色表示角度：")
    print("     红色=-90°, 绿色=0°, 蓝色=90°")
    print("  5. 方块表示有磁矩，圆圈表示无磁矩")
    print()
    print("优化方法:")
    print("  1. '连续优化磁矩': 使用差分进化算法，可找到全局较优解")
    print("  2. '离散优化 (5°步长)': 使用局部搜索，在-90°到90°范围内以5°步长搜索")
    print("  3. '优化磁矩分布': 优化哪些连杆应该有磁矩，同时优化磁矩方向")
    print()
    print("优化目标:")
    print("  最大化归一化变形率 J = (R_max - R_min) / R_avg")
    print("  该指标衡量机器人头部相对于自身尺度的伸缩能力")
    print("=" * 70)
    print()
    print("使用说明:")
    print("  1. 调整滑块改变参数")
    print("  2. 点击磁矩开关✓/✗按钮切换磁矩状态")
    print("  3. 点击优化按钮寻找最优配置")
    print("  4. 点击'全开'/'全关'按钮控制所有磁矩")
    print("  5. 点击'重置所有参数'恢复默认值")
    print()
    print("注意：磁矩角度范围已限制在-90°到90°之间")

    designer = MagneticRobotDesignerMatplotlib()
    designer.show()
