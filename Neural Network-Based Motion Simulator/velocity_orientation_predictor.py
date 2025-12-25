"""
complete_dynamics_model_optimized.py
磁驱执行器完整动力学模型 - 基于最新性能数据优化
"""

import numpy as np
import torch
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# 导入现有的模拟器（假设这些文件在相同目录）
try:
    from simplified_direction_predictor_202512161932_GREAT import DirectionPredictor
    from actuator_velocity_predictor_fixed_20251215_184845_GREAT import ActuatorPredictor

    print("✅ 成功导入现有模拟器")
except ImportError as e:
    print(f"⚠ 导入现有模拟器失败: {e}")
    print("请确保两个模拟器文件在相同目录")


class OptimizedNoiseModel:
    """
    基于最新性能数据优化的随机波动噪声模型
    """

    def __init__(self,
                 velocity_performance_data: Optional[pd.DataFrame] = None,
                 direction_performance_data: Optional[Dict] = None,
                 config: Optional[Dict] = None):
        """
        初始化优化的噪声模型
        Args:
            velocity_performance_data: 速度模型性能数据
            direction_performance_data: 方向模型性能数据
            config: 噪声模型配置
        """
        self.config = config or self._default_config()

        # 加载速度性能数据
        self.velocity_performance = velocity_performance_data
        self.direction_performance = direction_performance_data

        # 基于性能数据设置噪声参数
        self._set_parameters_from_performance()

        # 初始化噪声状态
        self.reset()

        print("=" * 50)
        print("优化噪声模型初始化")
        print("=" * 50)
        self._print_noise_summary()

    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'velocity_noise_type': 'colored_adaptive',  # 自适应有色噪声
            'direction_noise_type': 'colored_performance_based',  # 基于性能数据
            'use_velocity_uncertainty': True,  # 使用速度模型的不确定性
            'seed': 42,
        }

    def _set_parameters_from_performance(self):
        """基于性能数据设置噪声参数"""

        # 1. 速度噪声参数（基于训练结果CSV）
        if self.velocity_performance is not None:
            # 使用5折交叉验证的平均MAE作为速度噪声基准
            avg_mae = self.velocity_performance['mae'].mean()
            # 注意：CSV中的MAE是对数速度的误差，需要适当转换
            # 我们将其作为速度噪声的相对标准差
            velocity_mae = float(avg_mae)

            # 对于对数速度的MAE为0.253，对应实际速度的相对误差约为exp(0.253)-1 ≈ 28.8%
            # 但考虑到预测性能R²=0.95，实际噪声应更小
            velocity_std_factor = 0.15  # 保守估计：15%的速度标准差
        else:
            # 默认值（基于您的训练结果摘要）
            velocity_std_factor = 0.15  # 15%的速度标准差

        self.velocity_noise_params = {
            'std_factor': velocity_std_factor,  # 速度标准差因子（相对值）
            'base_std': 0.05,  # 基础标准差 (mm/s)
            'correlation_time': 5.0,  # 速度噪声相关时间（时间步）
            'use_model_uncertainty': self.config.get('use_velocity_uncertainty', True),
        }

        # 2. 方向噪声参数（基于performance.json）
        if self.direction_performance is not None:
            direction_mean_error = abs(self.direction_performance.get('mean_error', 3.81))
            direction_std_error = self.direction_performance.get('std_error', 2.38)
            direction_signed_error = self.direction_performance.get('mean_signed_error', -0.47)
        else:
            # 默认值（基于您的训练结果摘要）
            direction_mean_error = 3.81
            direction_std_error = 2.38
            direction_signed_error = -0.47

        self.direction_noise_params = {
            'std_deg': direction_std_error * 1.2,  # 使用标准差，稍微放大以覆盖更多变化
            'bias_deg': direction_signed_error,  # 平均偏差
            'correlation_time': 3.0,  # 方向噪声相关时间
            'alignment_threshold': 15.0,  # 对齐阈值
        }

        # 3. 轨迹噪声参数（基于综合性能）
        self.trajectory_noise_params = {
            'position_error_scale': 0.1,  # 位置误差缩放因子
            'correlation_time': 10.0,  # 位置噪声
        }

    def _print_noise_summary(self):
        """打印噪声模型摘要"""
        print("速度噪声参数:")
        print(f"  - 相对标准差因子: {self.velocity_noise_params['std_factor']:.3f}")
        print(f"  - 基础标准差: {self.velocity_noise_params['base_std']:.3f} mm/s")
        print(f"  - 相关时间: {self.velocity_noise_params['correlation_time']:.1f} 时间步")
        print(f"  - 使用模型不确定性: {self.velocity_noise_params['use_model_uncertainty']}")

        print("\n方向噪声参数:")
        print(f"  - 标准差: {self.direction_noise_params['std_deg']:.2f}°")
        print(f"  - 平均偏差: {self.direction_noise_params['bias_deg']:.2f}°")
        print(f"  - 相关时间: {self.direction_noise_params['correlation_time']:.1f} 时间步")
        print(f"  - 对齐阈值: {self.direction_noise_params['alignment_threshold']:.1f}°")

        print("\n轨迹噪声参数:")
        print(f"  - 位置误差缩放: {self.trajectory_noise_params['position_error_scale']:.3f}")
        print(f"  - 相关时间: {self.trajectory_noise_params['correlation_time']:.1f} 时间步")
        print("=" * 50)

    def reset(self, seed: Optional[int] = None):
        """
        重置噪声状态

        Args:
            seed: 随机种子
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.velocity_noise_state = 0.0
        self.direction_noise_state = 0.0
        self.position_noise_state = np.array([0.0, 0.0])

        # 存储历史噪声用于分析
        self.velocity_noise_history = []
        self.direction_noise_history = []
        self.position_noise_history = []

    def generate_velocity_noise(self,
                                predicted_velocity: float,
                                velocity_uncertainty: Optional[float] = None,
                                dt: float = 1.0) -> float:
        """
        生成速度随机波动（基于性能数据优化）

        Args:
            predicted_velocity: 预测速度 (mm/s)
            velocity_uncertainty: 速度模型的不确定性（如果可用）
            dt: 时间步长

        Returns:
            速度噪声值 (mm/s)
        """
        params = self.velocity_noise_params

        # 1. 确定噪声标准差
        if params['use_model_uncertainty'] and velocity_uncertainty is not None:
            # 使用模型预测的不确定性
            base_std = velocity_uncertainty
        else:
            # 基于预测速度和性能数据计算标准差
            relative_std = params['std_factor'] * predicted_velocity
            base_std = max(params['base_std'], relative_std)

        # 2. 白噪声成分
        white_noise = np.random.normal(0, base_std)

        # 3. 有色噪声成分（有记忆性）
        alpha = np.exp(-dt / params['correlation_time'])
        self.velocity_noise_state = (
                alpha * self.velocity_noise_state +
                np.sqrt(1 - alpha ** 2) * white_noise
        )

        # 4. 限制噪声幅度（不超过预测速度的50%）
        max_noise = 0.5 * predicted_velocity
        if abs(self.velocity_noise_state) > max_noise:
            self.velocity_noise_state = np.sign(self.velocity_noise_state) * max_noise

        total_noise = self.velocity_noise_state

        # 记录历史
        self.velocity_noise_history.append(total_noise)

        return total_noise

    def generate_direction_noise(self,
                                 predicted_direction: float,
                                 direction_confidence: Optional[float] = None,
                                 dt: float = 1.0) -> float:
        """
        生成方向随机波动（基于性能数据优化）

        Args:
            predicted_direction: 预测方向（度）
            direction_confidence: 方向置信度（如果可用）
            dt: 时间步长

        Returns:
            方向噪声值（度）
        """
        params = self.direction_noise_params

        # 1. 确定噪声标准差（考虑置信度）
        if direction_confidence is not None:
            # 置信度越高，噪声越小
            confidence_factor = 1.0 / (direction_confidence + 0.1)
            effective_std = params['std_deg'] * confidence_factor
        else:
            effective_std = params['std_deg']

        # 2. 白噪声成分
        white_noise = np.random.normal(0, effective_std)

        # 3. 有色噪声成分
        alpha = np.exp(-dt / params['correlation_time'])
        self.direction_noise_state = (
                alpha * self.direction_noise_state +
                np.sqrt(1 - alpha ** 2) * white_noise
        )

        # 4. 添加平均偏差（从性能数据中得到）
        total_noise_deg = self.direction_noise_state + params['bias_deg']

        # 5. 限制噪声幅度（不超过对齐阈值）
        max_noise = params['alignment_threshold']
        if abs(total_noise_deg) > max_noise:
            total_noise_deg = np.sign(total_noise_deg) * max_noise

        # 记录历史
        self.direction_noise_history.append(total_noise_deg)

        return total_noise_deg

    def generate_position_noise(self,
                                current_position: np.ndarray,
                                dt: float = 1.0) -> np.ndarray:
        """
        生成位置随机波动（2D）

        Args:
            current_position: 当前位置 [x, y]
            dt: 时间步长

        Returns:
            位置噪声值 [Δx, Δy]
        """
        params = self.trajectory_noise_params

        # 1. 基于速度噪声估计位置噪声幅度
        avg_velocity_noise = 0.0
        if self.velocity_noise_history:
            avg_velocity_noise = np.mean(np.abs(self.velocity_noise_history[-10:]))

        position_noise_magnitude = params['position_error_scale'] * avg_velocity_noise * dt

        # 2. 有色位置噪声
        white_noise = np.random.normal(0, position_noise_magnitude, size=2)

        alpha = np.exp(-dt / params['correlation_time'])
        self.position_noise_state = (
                alpha * self.position_noise_state +
                np.sqrt(1 - alpha ** 2) * white_noise
        )

        # 记录历史
        self.position_noise_history.append(self.position_noise_state.copy())

        return self.position_noise_state

    def get_noise_statistics(self) -> Dict:
        """获取噪声统计信息"""
        if not self.velocity_noise_history:
            return {}

        v_noise = np.array(self.velocity_noise_history)
        d_noise = np.array(self.direction_noise_history)

        stats = {
            'velocity_noise': {
                'mean': float(np.mean(v_noise)),
                'std': float(np.std(v_noise)),
                'min': float(np.min(v_noise)),
                'max': float(np.max(v_noise)),
                'mean_abs': float(np.mean(np.abs(v_noise))),
            },
            'direction_noise': {
                'mean_deg': float(np.mean(d_noise)),
                'std_deg': float(np.std(d_noise)),
                'min_deg': float(np.min(d_noise)),
                'max_deg': float(np.max(d_noise)),
                'mean_bias': float(np.mean(d_noise)),
            }
        }

        # 计算对齐率（噪声小于阈值）
        alignment_threshold = self.direction_noise_params.get('alignment_threshold', 15.0)
        aligned_mask = np.abs(d_noise) < alignment_threshold
        stats['direction_noise']['alignment_rate'] = float(np.mean(aligned_mask))

        return stats

    def calibrate_from_real_data(self, real_trajectory_data: Dict):
        """
        基于真实轨迹数据校准噪声参数

        Args:
            real_trajectory_data: 真实轨迹数据，包含速度和方向信息
        """
        print("基于真实轨迹数据校准噪声参数...")

        # 这里可以实现基于真实数据的参数校准逻辑
        # 暂时使用简单的启发式校准
        if 'velocity_error' in real_trajectory_data:
            real_velocity_errors = np.array(real_trajectory_data['velocity_error'])
            self.velocity_noise_params['std_factor'] = np.std(real_velocity_errors) / 2.0

        if 'direction_error' in real_trajectory_data:
            real_direction_errors = np.array(real_trajectory_data['direction_error'])
            self.direction_noise_params['std_deg'] = np.std(real_direction_errors)
            self.direction_noise_params['bias_deg'] = np.mean(real_direction_errors)

        print("✅ 噪声参数校准完成")


class CompleteDynamicsModelOptimized:
    """
    完整动力学模型：整合速度 + 方向 + 基于性能的噪声
    """

    def __init__(self,
                 velocity_model: ActuatorPredictor,
                 direction_model: DirectionPredictor,
                 noise_config: Optional[Dict] = None,
                 dt: float = 0.1):
        """
        初始化完整动力学模型

        Args:
            velocity_model: 速度预测器
            direction_model: 方向预测器
            noise_config: 噪声模型配置
            dt: 仿真时间步长（秒）
        """
        self.velocity_model = velocity_model
        self.direction_model = direction_model
        self.dt = dt

        # 加载性能数据
        velocity_perf = self._load_velocity_performance()
        direction_perf = self._load_direction_performance()

        # 初始化优化的噪声模型
        self.noise_model = OptimizedNoiseModel(
            velocity_performance_data=velocity_perf,
            direction_performance_data=direction_perf,
            config=noise_config
        )

        # 状态变量
        self.state = {
            'position': np.array([0.0, 0.0]),  # [x, y] 位置 (mm)
            'velocity': 0.0,  # 当前速度 (mm/s)
            'direction': 0.0,  # 当前运动方向 (度)
            'time': 0.0,  # 当前时间 (s)
        }

        # 历史记录
        self.history = {
            'time': [],
            'positions': [],
            'velocities': [],
            'directions': [],
            'predicted_velocities': [],
            'predicted_directions': [],
            'velocity_uncertainties': [],
            'direction_confidences': [],
            'velocity_noise': [],
            'direction_noise': [],
            'B0_mT': [],           # 添加磁场强度记录
            'f_Hz': [],            # 添加频率记录
            'viscosity_cP': [],    # 添加粘度记录
            'field_theta_deg': [], # 添加磁场方向记录
        }

        print("=" * 60)
        print("磁驱执行器完整动力学模型（优化版）")
        print("=" * 60)
        print(f"时间步长: {dt}秒")
        print(f"速度模型: R²={velocity_perf['r2'].mean():.4f}, MAE={velocity_perf['mae'].mean():.4f}")
        print(
            f"方向模型: 平均误差={direction_perf.get('mean_error', 3.81):.2f}°, 对齐率={direction_perf.get('alignment_rate', 1.0):.1%}")
        print("=" * 60)

    def _load_velocity_performance(self) -> pd.DataFrame:
        """加载速度模型性能数据"""
        try:
            velocity_perf = pd.read_csv('20251215_184845_training_results.csv')
            print(f"✅ 加载速度性能数据: {len(velocity_perf)}折交叉验证")
            return velocity_perf
        except FileNotFoundError:
            print("⚠ 速度性能文件未找到，使用默认性能估计")
            # 创建默认性能数据
            return pd.DataFrame({
                'r2': [0.9503],
                'mae': [0.2529]
            })

    def _load_direction_performance(self) -> Dict:
        """加载方向模型性能数据"""
        try:
            with open('performance.json', 'r', encoding='utf-8') as f:
                direction_perf = json.load(f)
            print(f"✅ 加载方向性能数据: 平均误差={direction_perf.get('mean_error', 3.81):.2f}°")
            return direction_perf
        except FileNotFoundError:
            print("⚠ 方向性能文件未找到，使用默认性能估计")
            return {
                'mean_error': 3.81,
                'std_error': 2.38,
                'mean_signed_error': -0.47,
                'alignment_rate': 1.0
            }

    def reset(self,
              initial_position: Optional[np.ndarray] = None,
              initial_velocity: float = 0.0,
              initial_direction: float = 0.0,
              seed: Optional[int] = None):
        """
        重置模型状态

        Args:
            initial_position: 初始位置 [x, y]
            initial_velocity: 初始速度
            initial_direction: 初始方向
            seed: 随机种子
        """
        self.state['position'] = initial_position if initial_position is not None else np.array([0.0, 0.0])
        self.state['velocity'] = initial_velocity
        self.state['direction'] = initial_direction % 360
        self.state['time'] = 0.0

        # 重置噪声模型
        self.noise_model.reset(seed)

        # 重置历史记录
        for key in self.history:
            self.history[key] = []

        # 记录初始状态
        self._record_state()

    def _record_state(self):
        """记录当前状态到历史"""
        self.history['time'].append(self.state['time'])
        self.history['positions'].append(self.state['position'].copy())
        self.history['velocities'].append(self.state['velocity'])
        self.history['directions'].append(self.state['direction'])

    def step(self,
             action: Dict,
             apply_noise: bool = True) -> Dict:
        """
        执行一步动力学仿真（使用性能优化模型）

        Args:
            action: 控制动作，包含以下键：
                - 'B0_mT': 磁场强度 (mT)
                - 'f_Hz': 频率 (Hz)
                - 'viscosity_cP': 粘度 (cP)
                - 'field_theta_deg': 磁场方向 (度)
            apply_noise: 是否应用噪声

        Returns:
            更新后的状态
        """
        # 提取动作参数
        B0_mT = action['B0_mT']
        f_Hz = action['f_Hz']
        viscosity_cP = action['viscosity_cP']
        field_theta_deg = action['field_theta_deg']

        # 1. 预测速度（基于物理参数）
        predicted_velocity, velocity_uncertainty = self.velocity_model.predict_single(
            viscosity_cP, B0_mT, f_Hz
        )

        if predicted_velocity is None:
            raise ValueError("速度预测失败")

        # 2. 预测方向（基于物理参数和磁场方向）
        predicted_direction, direction_confidence = self.direction_model.predict(
            B0_mT, f_Hz, viscosity_cP, field_theta_deg
        )

        # 3. 添加随机波动（如果启用）
        if apply_noise:
            # 使用优化的噪声生成方法
            velocity_noise = self.noise_model.generate_velocity_noise(
                predicted_velocity,
                velocity_uncertainty,
                self.dt
            )

            direction_noise_deg = self.noise_model.generate_direction_noise(
                predicted_direction,
                direction_confidence,
                self.dt
            )

            # 位置噪声
            position_noise = self.noise_model.generate_position_noise(
                self.state['position'],
                self.dt
            )

            actual_velocity = max(0.0, predicted_velocity + velocity_noise)
            actual_direction = predicted_direction + direction_noise_deg

            # 确保方向在0-360度范围内
            actual_direction = actual_direction % 360
        else:
            velocity_noise = 0.0
            direction_noise_deg = 0.0
            position_noise = np.array([0.0, 0.0])
            actual_velocity = predicted_velocity
            actual_direction = predicted_direction

        # 4. 更新状态
        # 位置更新: Δx = v * cos(θ) * Δt + 位置噪声, Δy = v * sin(θ) * Δt + 位置噪声
        theta_rad = np.radians(actual_direction)
        delta_x = actual_velocity * np.cos(theta_rad) * self.dt + position_noise[0]
        delta_y = actual_velocity * np.sin(theta_rad) * self.dt + position_noise[1]

        self.state['position'][0] += delta_x
        self.state['position'][1] += delta_y
        self.state['velocity'] = actual_velocity
        self.state['direction'] = actual_direction
        self.state['time'] += self.dt

        # 5. 记录预测值、不确定性和噪声
        self.history['predicted_velocities'].append(predicted_velocity)
        self.history['predicted_directions'].append(predicted_direction)
        self.history['velocity_uncertainties'].append(velocity_uncertainty)
        self.history['direction_confidences'].append(direction_confidence)
        self.history['velocity_noise'].append(velocity_noise)
        self.history['direction_noise'].append(direction_noise_deg)
        self.history['B0_mT'].append(B0_mT)
        self.history['f_Hz'].append(f_Hz)
        self.history['viscosity_cP'].append(viscosity_cP)
        self.history['field_theta_deg'].append(field_theta_deg)

        # 6. 记录状态
        self._record_state()

        # 返回当前状态和预测信息
        return {
            'state': self.state.copy(),
            'predictions': {
                'velocity': predicted_velocity,
                'velocity_uncertainty': velocity_uncertainty,
                'direction': predicted_direction,
                'direction_confidence': direction_confidence,
            },
            'noise': {
                'velocity': velocity_noise,
                'direction': direction_noise_deg,
                'position': position_noise.tolist(),
            }
        }

    def simulate_trajectory(self,
                            actions: List[Dict],
                            initial_position: Optional[np.ndarray] = None,
                            seed: Optional[int] = None,
                            apply_noise: bool = True) -> Dict:
        """
        仿真完整轨迹

        Args:
            actions: 动作序列
            initial_position: 初始位置
            seed: 随机种子
            apply_noise: 是否应用噪声

        Returns:
            仿真结果
        """
        # 重置模型
        self.reset(initial_position=initial_position, seed=seed)

        print(f"开始轨迹仿真，共{len(actions)}步...")

        # 执行每一步
        states = []
        for i, action in enumerate(actions):
            if i % 20 == 0:
                print(f"  进度: {i + 1}/{len(actions)}步")

            step_result = self.step(action, apply_noise=apply_noise)
            states.append(step_result)

        print("✅ 轨迹仿真完成")

        # 获取噪声统计
        noise_stats = self.noise_model.get_noise_statistics()

        # 计算轨迹统计
        positions = np.array(self.history['positions'])
        velocities = np.array(self.history['velocities'])
        predicted_velocities = np.array(self.history['predicted_velocities'])

        # 计算轨迹长度
        if len(positions) > 1:
            trajectory_length = np.sum(np.sqrt(np.sum(np.diff(positions, axis=0) ** 2, axis=1)))
        else:
            trajectory_length = 0.0

        # 计算速度预测误差统计
        if len(velocities) > 1 and len(predicted_velocities) > 0:
            # 注意：velocities 包括初始状态，predicted_velocities 不包括
            # 所以我们应该比较 velocities[1:] 和 predicted_velocities
            actual_velocities = velocities[1:]  # 排除初始状态
            min_len = min(len(actual_velocities), len(predicted_velocities))
            velocity_errors = actual_velocities[:min_len] - predicted_velocities[:min_len]
            velocity_mae = np.mean(np.abs(velocity_errors))
        else:
            velocity_mae = 0.0

        return {
            'states': states,
            'history': self.history.copy(),
            'noise_statistics': noise_stats,
            'trajectory_statistics': {
                'length': float(trajectory_length),
                'duration': float(self.state['time']),
                'avg_velocity': float(np.mean(velocities)) if len(velocities) > 0 else 0.0,
                'max_velocity': float(np.max(velocities)) if len(velocities) > 0 else 0.0,
                'min_velocity': float(np.min(velocities)) if len(velocities) > 0 else 0.0,
                'velocity_mae': float(velocity_mae),
            }
        }

    def analyze_performance(self) -> Dict:
        """
        分析模型性能
        """
        if not self.history['time']:
            return {"error": "没有仿真数据"}

        # 计算各种性能指标
        velocities = np.array(self.history['velocities'])
        predicted_velocities = np.array(self.history['predicted_velocities'])
        directions = np.array(self.history['directions'])
        predicted_directions = np.array(self.history['predicted_directions'])

        if len(velocities) > 1 and len(predicted_velocities) > 0:
            actual_velocities = velocities[1:]  # 排除初始状态
            min_len = min(len(actual_velocities), len(predicted_velocities))
            velocity_errors = actual_velocities[:min_len] - predicted_velocities[:min_len]
            velocity_mae = np.mean(np.abs(velocity_errors))
            velocity_rmse = np.sqrt(np.mean(velocity_errors ** 2))
            velocity_r2 = 1 - np.sum(velocity_errors ** 2) / np.sum(
                (actual_velocities[:min_len] - np.mean(actual_velocities[:min_len])) ** 2) if len(
                actual_velocities[:min_len]) > 1 else 0
        else:
            velocity_mae = 0.0
            velocity_rmse = 0.0
            velocity_r2 = 0.0

        # 方向性能（考虑角度周期性）
        direction_errors = []
        for actual, pred in zip(directions, predicted_directions):
            error = abs(actual - pred)
            if error > 180:
                error = 360 - error
            direction_errors.append(error)

        direction_errors = np.array(direction_errors)
        direction_mae = np.mean(direction_errors)
        direction_rmse = np.sqrt(np.mean(direction_errors ** 2))
        alignment_rate_15 = np.mean(direction_errors < 15)
        alignment_rate_5 = np.mean(direction_errors < 5)

        return {
            'velocity_performance': {
                'mae': float(velocity_mae),
                'rmse': float(velocity_rmse),
                'r2': float(velocity_r2),
                'mean_actual': float(np.mean(velocities)) if len(velocities) > 0 else 0.0,
                'mean_predicted': float(np.mean(predicted_velocities)) if len(predicted_velocities) > 0 else 0.0,
            },
            'direction_performance': {
                'mae_deg': float(direction_mae),
                'rmse_deg': float(direction_rmse),
                'alignment_rate_15deg': float(alignment_rate_15),
                'alignment_rate_5deg': float(alignment_rate_5),
                'max_error_deg': float(np.max(direction_errors)) if len(direction_errors) > 0 else 0.0,
            },
            'simulation_info': {
                'duration': float(self.state['time']),
                'num_steps': len(self.history['time']),
                'trajectory_length': float(np.sum(np.sqrt(np.sum(np.diff(
                    np.array(self.history['positions']), axis=0) ** 2, axis=1)))) if len(
                    self.history['positions']) > 1 else 0.0,
            }
        }


# ==================== 使用示例 ====================

def create_demo_actions(num_steps: int = 100) -> List[Dict]:
    """
    创建演示动作序列

    Args:
        num_steps: 步数

    Returns:
        动作序列
    """
    actions = []

    for i in range(num_steps):
        # 创建变化的控制参数
        # 磁场方向：从0到360度循环变化
        # field_theta = (i * 3.6) % 360  # 每步变化3.6度，100步完成一圈
        field_theta = 30

        # 磁场强度：在5-10mT之间正弦变化
        #B0 = 7.5 + 2.5 * np.sin(2 * np.pi * i / 40)
        B0 = 5

        # 频率：在1-5Hz之间正弦变化
        #f = 3.0 + 2.0 * np.cos(2 * np.pi * i / 60)
        f=1

        action = {
            'B0_mT': float(B0),
            'f_Hz': float(f),
            'viscosity_cP': 9.5,  # 固定粘度
            'field_theta_deg': float(field_theta),
        }
        actions.append(action)

    return actions


def main():
    """主函数：演示优化后的动力学模型"""

    print("=" * 70)
    print("磁驱执行器完整动力学模型（基于性能优化）")
    print("=" * 70)

    try:
        # 1. 加载现有的模拟器
        print("1. 加载速度模拟器...")
        velocity_predictor = ActuatorPredictor()

        print("2. 加载方向模拟器...")
        # 注意：需要根据您的实际文件路径调整
        direction_predictor = DirectionPredictor(
            model_weights_path="model_weights_202512161932.pth",
            model_config_path="model_config_202512161932.json"
        )

        # 2. 创建优化后的完整动力学模型
        print("3. 创建优化动力学模型...")
        dynamics_model = CompleteDynamicsModelOptimized(
            velocity_model=velocity_predictor,
            direction_model=direction_predictor,
            dt=0.05,  # 50ms时间步长（更精细）
            noise_config={
                'velocity_noise_type': 'colored_adaptive',
                'direction_noise_type': 'colored_performance_based',
                'use_velocity_uncertainty': True,
                'seed': 42,
            }
        )

        # 3. 创建演示动作序列
        print("4. 创建演示动作序列...")
        actions = create_demo_actions(num_steps=200)  # 200步，总仿真时间10秒

        # 4. 运行仿真
        print("5. 运行轨迹仿真...")
        result = dynamics_model.simulate_trajectory(
            actions=actions,
            initial_position=np.array([0.0, 0.0]),
            seed=42,
            apply_noise=True
        )

        # 5. 分析性能
        print("6. 分析模型性能...")
        performance = dynamics_model.analyze_performance()

        # 6. 显示结果
        print("\n" + "=" * 70)
        print("仿真结果总结")
        print("=" * 70)

        stats = result['trajectory_statistics']
        print(f"轨迹长度: {stats['length']:.2f} mm")
        print(f"仿真时长: {stats['duration']:.2f} s")
        print(f"平均速度: {stats['avg_velocity']:.2f} mm/s")
        print(f"速度预测MAE: {stats['velocity_mae']:.3f} mm/s")

        print(f"\n方向性能:")
        print(f"  平均误差: {performance['direction_performance']['mae_deg']:.2f}°")
        print(f"  对齐率(<15°): {performance['direction_performance']['alignment_rate_15deg']:.1%}")
        print(f"  对齐率(<5°): {performance['direction_performance']['alignment_rate_5deg']:.1%}")

        # 噪声统计
        noise_stats = result['noise_statistics']
        if noise_stats:
            print(f"\n噪声统计:")
            print(f"  速度噪声均值: {noise_stats['velocity_noise']['mean']:.3f} mm/s")
            print(f"  速度噪声标准差: {noise_stats['velocity_noise']['std']:.3f} mm/s")
            print(f"  方向噪声均值: {noise_stats['direction_noise']['mean_deg']:.2f}°")
            print(f"  方向噪声标准差: {noise_stats['direction_noise']['std_deg']:.2f}°")
            if 'alignment_rate' in noise_stats['direction_noise']:
                print(f"  方向对齐率: {noise_stats['direction_noise']['alignment_rate']:.1%}")

        # 7. 导出轨迹数据...
        print("\n7. 导出轨迹数据...")

        # 创建可序列化的数据副本
        serializable_history = {}
        for key, value in dynamics_model.history.items():
            if isinstance(value, np.ndarray):
                # 将 NumPy 数组转换为列表
                serializable_history[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                # 列表中包含 NumPy 数组
                serializable_history[key] = [v.tolist() if isinstance(v, np.ndarray) else v for v in value]
            else:
                # 其他类型直接复制
                serializable_history[key] = value

        export_data = {
            'metadata': {
                'dt': dynamics_model.dt,
                'simulation_time': float(dynamics_model.state['time']),  # 确保是 Python 浮点数
                'performance_summary': performance,
                'noise_statistics': noise_stats,
            },
            'trajectory': serializable_history,
        }

        # 确保所有数据都是可序列化的 Python 类型
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            else:
                return obj

        export_data = convert_to_serializable(export_data)

        with open('optimized_trajectory.json', 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        print(f"✅ 轨迹数据已导出到 optimized_trajectory.json")

        print("\n" + "=" * 70)
        print("✅ 优化动力学模型演示完成！")
        print("=" * 70)

        return dynamics_model, result, performance

    except Exception as e:
        print(f"\n❌ 错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    # 运行演示
    dynamics_model, result, performance = main()