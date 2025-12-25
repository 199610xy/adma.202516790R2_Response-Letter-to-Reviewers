# actuator_predictor_20251215_184845_fixed.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import joblib
import warnings

warnings.filterwarnings('ignore')

# ==================== 自定义层定义 ====================
class PositiveConstraint(layers.Layer):
    """自定义层：确保输出为正"""

    def __init__(self, **kwargs):
        super(PositiveConstraint, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.abs(inputs)

    def get_config(self):
        return super(PositiveConstraint, self).get_config()

class ActuatorPredictor:
    """
    执行器速度预测器 - 修复版
    使用自定义层确保正确加载
    """

    def __init__(self, timestamp="20251215_184845"):
        self.models = []
        self.scalers_x = []
        self.scalers_y = []
        self.feature_order = ['Viscosity(cp)', 'Ba(mT)', 'Frequency (Hz)', 'log_viscosity', 'Ba_Freq_product', 'Viscosity_Freq_ratio', 'Ba_Viscosity_ratio', 'sqrt_frequency', 'reynolds_like']
        self.timestamp = timestamp

        print(f"正在加载预测器 (版本: {timestamp})...")

        # 加载模型和标准化器
        success_count = 0
        for fold in range(1, 6):
            model_path = f'{timestamp}_model_fold_{fold}.keras'
            scaler_x_path = f'{timestamp}_scaler_x_fold_{fold}.pkl'
            scaler_y_path = f'{timestamp}_scaler_y_fold_{fold}.pkl'

            try:
                # 定义custom_objects
                custom_objects = {
                    'PositiveConstraint': PositiveConstraint,
                    'AdamW': optimizers.AdamW
                }

                # 加载模型
                model = keras.models.load_model(model_path, custom_objects=custom_objects)

                # 加载标准化器
                scaler_x = joblib.load(scaler_x_path)
                scaler_y = joblib.load(scaler_y_path)

                self.models.append(model)
                self.scalers_x.append(scaler_x)
                self.scalers_y.append(scaler_y)

                print(f"✅ 成功加载第{fold}折模型")
                success_count += 1

            except FileNotFoundError:
                print(f"❌ 文件未找到: {model_path}")
            except Exception as e:
                print(f"❌ 加载第{fold}折模型失败: {type(e).__name__}: {str(e)[:100]}...")

        if success_count == 0:
            print("❌ 错误：没有模型加载成功！")
            print("请确保以下文件存在：")
            for fold in range(1, 6):
                print(f"  - {timestamp}_model_fold_{fold}.keras")
                print(f"  - {timestamp}_scaler_x_fold_{fold}.pkl")
                print(f"  - {timestamp}_scaler_y_fold_{fold}.pkl")
        else:
            print(f"✅ 模型加载完成，共{success_count}个模型可用")

    def _create_features(self, viscosity, ba, frequency):
        """创建特征向量"""
        return {
            'Viscosity(cp)': viscosity,
            'Ba(mT)': ba,
            'Frequency (Hz)': frequency,
            'log_viscosity': np.log1p(viscosity),
            'Ba_Freq_product': ba * frequency,
            'Viscosity_Freq_ratio': frequency / (viscosity + 1e-8),
            'Ba_Viscosity_ratio': ba / (viscosity + 1e-8),
            'sqrt_frequency': np.sqrt(frequency),
            'reynolds_like': (ba * frequency) / (viscosity + 1e-8)
        }

    def predict_single(self, viscosity, ba, frequency, verbose=False):
        """
        单个预测

        参数:
        ----------
        viscosity : float
            粘度 (cp)
        ba : float
            磁场强度 (mT)
        frequency : float
            频率 (Hz)
        verbose : bool
            是否显示详细信息

        返回:
        ----------
        tuple (预测速度, 不确定性)
        """
        if len(self.models) == 0:
            if verbose:
                print("错误：没有可用的模型！")
            return None, None

        # 创建特征
        features = self._create_features(viscosity, ba, frequency)
        X = np.array([[features[col] for col in self.feature_order]])

        # 集成预测
        predictions = []
        for i, (model, scaler_x, scaler_y) in enumerate(zip(self.models, self.scalers_x, self.scalers_y), 1):
            try:
                X_scaled = scaler_x.transform(X)
                y_pred_scaled = model.predict(X_scaled, verbose=0).flatten()
                y_pred_log = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                y_pred = np.expm1(y_pred_log)[0]
                predictions.append(y_pred)

                if verbose:
                    print(f"  模型{i}预测: {y_pred:.3f} mm/s")

            except Exception as e:
                if verbose:
                    print(f"  模型{i}预测失败: {type(e).__name__}")
                continue

        # 计算统计量
        if predictions:
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)

            if verbose:
                print(f"\n平均预测: {mean_pred:.3f} mm/s")
                print(f"标准差: {std_pred:.3f} mm/s")
                print(f"预测范围: {mean_pred-std_pred:.3f} - {mean_pred+std_pred:.3f} mm/s")

            return mean_pred, std_pred
        else:
            if verbose:
                print("错误：所有模型预测失败！")
            return None, None

    def predict_batch(self, params_list, verbose=False):
        """
        批量预测

        参数:
        ----------
        params_list : list of tuples
            [(viscosity, ba, frequency), ...]
        verbose : bool
            是否显示详细信息

        返回:
        ----------
        list of dicts
            每个参数组合的预测结果
        """
        results = []
        for viscosity, ba, frequency in params_list:
            if verbose:
                print(f"\n预测: 粘度{viscosity}cp, 磁场{ba}mT, 频率{frequency}Hz")

            pred_mean, pred_std = self.predict_single(viscosity, ba, frequency, verbose=verbose)

            if pred_mean is not None:
                results.append({
                    'viscosity': viscosity,
                    'ba': ba,
                    'frequency': frequency,
                    'predicted_velocity': pred_mean,
                    'uncertainty': pred_std,
                    'confidence_interval': [
                        pred_mean - 1.96 * pred_std,
                        pred_mean + 1.96 * pred_std
                    ]
                })

                if verbose:
                    print(f"  结果: {pred_mean:.3f} ± {pred_std:.3f} mm/s")

        return results

    def save_predictions(self, results, filename):
        """保存预测结果到CSV文件"""
        import pandas as pd

        if not results:
            print("没有结果可保存")
            return False

        try:
            df = pd.DataFrame(results)
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"预测结果已保存到 {filename}")
            return True
        except Exception as e:
            print(f"保存失败: {e}")
            return False

# ==================== 使用示例 ====================
if __name__ == "__main__":
    print("=" * 60)
    print("磁驱执行器速度预测系统 - 修复版")
    print("=" * 60)

    try:
        # 创建预测器
        predictor = ActuatorPredictor()

        # 检查是否有模型加载成功
        if len(predictor.models) > 0:
            print("\n✅ 预测器准备就绪！")
            print(f"可用模型数: {len(predictor.models)}")

            # 示例1：单个预测（详细模式）
            print("\n📊 示例1：详细预测")
            speed, uncertainty = predictor.predict_single(35.0, 10.0, 5.0, verbose=True)

            if speed is not None:
                print(f"\n📈 最终结果:")
                print(f"  预测速度: {speed:.3f} ± {uncertainty:.3f} mm/s")
                print(f"  95%置信区间: [{speed-1.96*uncertainty:.3f}, {speed+1.96*uncertainty:.3f}] mm/s")

            # 示例2：批量预测
            print("\n📊 示例2：批量预测")
            test_params = [
                (35.0, 10.0, 5.0),
                (50.0, 8.0, 10.0),
                (100.0, 12.0, 15.0)
            ]

            batch_results = predictor.predict_batch(test_params, verbose=False)

            if batch_results:
                print("\n批量预测结果:")
                for result in batch_results:
                    print(f"  粘度{result['viscosity']}cp, 磁场{result['ba']}mT, 频率{result['frequency']}Hz: {result['predicted_velocity']:.3f} ± {result['uncertainty']:.3f} mm/s")

                # 保存结果
                predictor.save_predictions(batch_results, 'predictions.csv')

            # 示例3：快速预测
            print("\n📊 示例3：快速预测")
            speed, error = predictor.predict_single(35.0, 10.0, 5.0, verbose=False)
            if speed:
                print(f"快速预测: {speed:.3f} ± {error:.3f} mm/s")

        else:
            print("\n❌ 没有模型可用，请检查文件路径")

    except Exception as e:
        print(f"\n❌ 系统错误: {type(e).__name__}: {e}")
        print("\n故障排除:")
        print("1. 确保所有模型文件在当前目录")
        print("2. 检查TensorFlow/Keras版本")
        print("3. 确保有足够的磁盘空间和内存")

    print("\n" + "=" * 60)
    print("预测完成")
    print("=" * 60)
