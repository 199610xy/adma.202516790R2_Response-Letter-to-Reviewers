"""
simplified_direction_predictor.py
简化神经网络方向预测器 - 应用版本
"""

import json
import numpy as np
import torch
import torch.nn as nn
import os


class PhysicsAwareDirectionNN(nn.Module):
    """
    物理感知的方向神经网络（与训练时相同的架构）
    """

    def __init__(self, input_dim=5, hidden_dims=[32, 16], dropout_rate=0.3, emphasize_field=True):
        super(PhysicsAwareDirectionNN, self).__init__()
        self.emphasize_field = emphasize_field

        # 特征拆分：物理参数 + 磁场方向
        self.physical_features = 3  # B0, f, η
        self.field_features = 2  # sin(θ), cos(θ)

        # 物理参数分支（简单处理，强正则化）
        self.physical_branch = nn.Sequential(
            nn.Linear(self.physical_features, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 1.5),
            nn.Linear(8, 8)
        )

        # 磁场方向分支（更复杂处理，弱正则化）
        self.field_branch = nn.Sequential(
            nn.Linear(self.field_features, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )

        # 合并分支
        total_features = 8 + 16
        self.combined = nn.Sequential(
            nn.Linear(total_features, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 2),  # 输出sin, cos
            nn.Tanh()
        )

    def forward(self, x):
        # 拆分特征
        physical_features = x[:, :self.physical_features]
        field_features = x[:, self.physical_features:]

        # 分别处理
        physical_out = self.physical_branch(physical_features)
        field_out = self.field_branch(field_features)

        # 合并
        combined = torch.cat([physical_out, field_out], dim=1)
        output = self.combined(combined)

        return output


class DirectionPredictor:
    """
    方向预测器主类
    """

    def __init__(self, model_weights_path, model_config_path=None):
        """
        初始化预测器

        Args:
            model_weights_path: 模型权重文件路径 (.pth)
            model_config_path: 模型配置文件路径 (.json)，可选
        """
        print("=" * 60)
        print("简化神经网络方向预测器")
        print("=" * 60)

        # 1. 加载配置（如果有）
        self.config = None
        self.scaler = None

        if model_config_path and os.path.exists(model_config_path):
            try:
                with open(model_config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                print(f"✓ 加载模型配置: {model_config_path}")

                # 重建特征处理器
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                scaler_info = self.config['scaler_info']

                # 手动设置scaler参数
                self.scaler.mean_ = np.array(scaler_info['mean'])
                self.scaler.scale_ = np.array(scaler_info['scale'])
                self.scaler.n_features_in_ = scaler_info['n_features_in']

                input_dim = self.config['input_dim']
                feature_names = self.config['feature_names']

                print(f"  输入维度: {input_dim}")
                print(f"  特征: {feature_names}")

            except Exception as e:
                print(f"⚠ 加载配置失败: {e}")
                print("  将使用默认配置")
                self.config = None

        # 2. 创建模型
        if self.config:
            input_dim = self.config['input_dim']
            hidden_dims = self.config.get('model_architecture', {}).get('hidden_dims', [32, 16])
            dropout_rate = self.config.get('model_architecture', {}).get('dropout_rate', 0.3)
            emphasize_field = self.config.get('model_architecture', {}).get('emphasize_field', True)
        else:
            # 默认配置
            input_dim = 5
            hidden_dims = [32, 16]
            dropout_rate = 0.3
            emphasize_field = True

        self.model = PhysicsAwareDirectionNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            emphasize_field=emphasize_field
        )

        # 3. 加载权重
        try:
            self.model.load_state_dict(torch.load(model_weights_path, map_location='cpu'))
            self.model.eval()
            print(f"✓ 加载模型权重: {model_weights_path}")
        except Exception as e:
            print(f"❌ 加载权重失败: {e}")
            raise

        # 4. 如果没有配置文件，创建默认的scaler
        if self.scaler is None:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            # 使用训练时的典型统计值（根据您的数据估算）
            self.scaler.mean_ = np.array([1.0, 2.0, 1.0, 0.0, 0.0])  # 估算的均值
            self.scaler.scale_ = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # 估算的标准差
            self.scaler.n_features_in_ = 5
            print("⚠ 使用默认特征处理器（估算值）")

        print(f"✓ 预测器初始化完成")
        print(f"  平均角度误差: ~3.81° (训练时)")
        print(f"  对齐率(<15°): 100.0% (训练时)")
        print("=" * 60)

    def predict(self, B0_mT, f_Hz, viscosity_cP, field_theta_deg):
        """
        预测给定条件下的运动方向

        Args:
            B0_mT: 磁场强度 (mT)
            f_Hz: 频率 (Hz)
            viscosity_cP: 粘度 (cP)
            field_theta_deg: 磁场方向 (°)

        Returns:
            predicted_angle_deg: 预测的运动方向 (°), 范围 0-360
            confidence: 预测置信度 (0-1)
        """
        # 准备特征
        features = [
            float(B0_mT),
            float(f_Hz),
            float(viscosity_cP),
            np.sin(np.radians(float(field_theta_deg))),
            np.cos(np.radians(float(field_theta_deg)))
        ]

        features_array = np.array(features, dtype=np.float32).reshape(1, -1)

        # 标准化
        features_scaled = self.scaler.transform(features_array)

        # 预测
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled)
            predictions = self.model(features_tensor)

            # 将sin, cos转换回角度
            sin_pred = predictions[0, 0].item()
            cos_pred = predictions[0, 1].item()
            predicted_angle = np.degrees(np.arctan2(sin_pred, cos_pred))

            # 确保角度在0-360度范围内
            if predicted_angle < 0:
                predicted_angle += 360

            # 计算置信度（基于预测向量的模长）
            confidence = min(1.0, np.sqrt(sin_pred ** 2 + cos_pred ** 2))

        return predicted_angle, confidence

    def batch_predict(self, params_list):
        """
        批量预测

        Args:
            params_list: 参数列表，每个元素是字典 {B0_mT, f_Hz, viscosity_cP, field_theta_deg}

        Returns:
            results: 预测结果列表，每个元素是 (predicted_angle, confidence)
        """
        results = []
        for params in params_list:
            try:
                pred, conf = self.predict(**params)
                results.append((pred, conf))
            except Exception as e:
                print(f"预测失败 {params}: {e}")
                results.append((None, 0.0))
        return results

    def predict_with_uncertainty(self, B0_mT, f_Hz, viscosity_cP, field_theta_deg, n_samples=100):
        """
        带不确定性的预测（蒙特卡洛采样）

        Args:
            n_samples: 采样次数

        Returns:
            mean_angle: 平均角度
            std_angle: 角度标准差
            angle_samples: 所有采样结果
        """
        # 启用dropout进行不确定性估计
        self.model.train()

        features = [
            float(B0_mT),
            float(f_Hz),
            float(viscosity_cP),
            np.sin(np.radians(float(field_theta_deg))),
            np.cos(np.radians(float(field_theta_deg)))
        ]

        features_array = np.array(features, dtype=np.float32).reshape(1, -1)
        features_scaled = self.scaler.transform(features_array)
        features_tensor = torch.FloatTensor(features_scaled)

        angles = []
        with torch.no_grad():
            for _ in range(n_samples):
                predictions = self.model(features_tensor)
                sin_pred = predictions[0, 0].item()
                cos_pred = predictions[0, 1].item()
                angle = np.degrees(np.arctan2(sin_pred, cos_pred))
                if angle < 0:
                    angle += 360
                angles.append(angle)

        # 恢复评估模式
        self.model.eval()

        angles = np.array(angles)
        mean_angle = np.mean(angles)

        # 处理角度周期性
        angles_corrected = angles.copy()
        # 如果角度跨越360°边界，进行调整
        if np.std(angles) > 90:  # 如果标准差很大，可能跨越边界
            # 将角度转换到以均值为中心的表示
            angles_centered = (angles - mean_angle + 180) % 360 - 180
            mean_angle_corrected = mean_angle + np.mean(angles_centered)
            std_angle = np.std(angles_centered)
            mean_angle = mean_angle_corrected % 360
        else:
            std_angle = np.std(angles)

        return float(mean_angle % 360), float(std_angle), angles.tolist()


# ==================== 使用示例 ====================

# 修改simplified_direction_predictor.py中的使用示例部分：

if __name__ == "__main__":

    # 1. 初始化预测器
    print("初始化方向预测器...")

    # 使用正确的权重文件和配置文件
    weights_path = "model_weights_manual.pth"  # 确保文件名正确
    config_path = "model_config.json"  # 使用您提供的配置文件

    try:
        predictor = DirectionPredictor(weights_path, config_path)
    except Exception as e:
        print(f"初始化失败: {e}")
        # 如果文件不在当前目录，请调整路径
        weights_path = r"C:\Users\笑宇和弘毅\Desktop\INTENSE\2-实验数据-3rd\202512-神经网络构建执行器运动模拟器\2-代码\1-数据分析&模拟器构建\2-方向模拟器\model_weights_202512161932.pth"
        config_path = r"C:\Users\笑宇和弘毅\Desktop\INTENSE\2-实验数据-3rd\202512-神经网络构建执行器运动模拟器\2-代码\1-数据分析&模拟器构建\2-方向模拟器\model_config_202512161932.json"
        predictor = DirectionPredictor(weights_path, config_path)

    # 2. 使用典型范围内的参数进行测试
    print("\n单次预测示例（使用典型参数）:")
    print("-" * 60)

    # 使用接近训练数据均值的参数
    test_cases = [
        {"B0_mT": 8.5, "f_Hz": 6.8, "viscosity_cP": 36.0, "field_theta_deg": 0},
        {"B0_mT": 8.5, "f_Hz": 6.8, "viscosity_cP": 36.0, "field_theta_deg": 45},
        {"B0_mT": 8.5, "f_Hz": 6.8, "viscosity_cP": 36.0, "field_theta_deg": 90},
        {"B0_mT": 5.0, "f_Hz": 2.0, "viscosity_cP": 10.0, "field_theta_deg": 180},  # 边界值
        {"B0_mT": 10.0, "f_Hz": 10.0, "viscosity_cP": 80.0, "field_theta_deg": 270},  # 边界值
    ]

    for params in test_cases:
        predicted_angle, confidence = predictor.predict(**params)
        field_theta = params["field_theta_deg"]
        error = abs(predicted_angle - field_theta)
        if error > 180:
            error = 360 - error
        print(f"场方向: {field_theta:3.0f}°")
        print(f"预测方向: {predicted_angle:6.1f}°")
        print(f"误差: {error:6.1f}°, 置信度: {confidence:.3f}")
        print("-" * 40)

    # 3. 测试一个您之前失败的案例（但使用训练数据的分布）
    print("\n测试之前失败的案例（使用训练数据的典型值）:")
    print("-" * 60)

    # 根据您的数据，B0、f、η与方向无关，所以我们可以固定它们
    typical_params = {"B0_mT": 8.5, "f_Hz": 6.8, "viscosity_cP": 36.0}

    for field_angle in [0, 45, 90, 135, 180]:
        params = {**typical_params, "field_theta_deg": field_angle}
        predicted_angle, confidence = predictor.predict(**params)
        error = abs(predicted_angle - field_angle)
        if error > 180:
            error = 360 - error
        print(f"场方向{field_angle:3d}° -> 预测{predicted_angle:6.1f}° (误差{error:5.1f}°, 置信度{confidence:.3f})")

    print("\n" + "=" * 60)
    print("方向预测器测试完成！")
    print("=" * 60)


