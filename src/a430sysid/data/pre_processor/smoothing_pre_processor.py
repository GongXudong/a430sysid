import numpy as np
import pandas as pd
from scipy import signal

from a430sysid.data.pre_processor.pre_processor_base import PreProcessorBase


def robust_savgol_filter(x, window_length, polyorder, max_iter=5, threshold=2.0):
    """
    实现鲁棒的Savitzky-Golay滤波（迭代重加权最小二乘）

    参数:
    x: 输入信号
    window_length: 窗口长度
    polyorder: 多项式阶数
    max_iter: 最大迭代次数
    threshold: 异常值检测阈值

    返回:
    平滑后的信号
    """
    y = x.copy()
    weights = np.ones_like(x)

    for iteration in range(max_iter):
        # 使用当前权重进行加权Savitzky-Golay滤波
        try:
            # 标准Savitzky-Golay滤波（实际上scipy的savgol_filter不支持权重）
            # 这里我们实现一个简化的鲁棒版本：先平滑，然后检测异常值并调整权重
            y_smooth = signal.savgol_filter(y, window_length, polyorder, mode="interp")

            # 计算残差
            residuals = np.abs(y - y_smooth)
            mad = np.median(residuals)  # 中位数绝对偏差

            # 更新权重：残差大的点权重小
            new_weights = 1.0 / (1.0 + (residuals / (threshold * mad)) ** 2)

            # 检查收敛
            if np.max(np.abs(new_weights - weights)) < 0.01:
                break

            weights = new_weights
            y = y_smooth * (1 - weights) + x * weights  # 加权组合

        except Exception as e:
            print(f"鲁棒平滑迭代 {iteration} 失败: {e}")
            break

    return y_smooth


class SmoothingPreProcessor(PreProcessorBase):
    def __init__(
        self,
        column_names_to_smooth: list[str],
        new_column_names: list[str],
        window_size: int = 11,
        poly_order: int = 3,
        robust_smoothing: bool = True,
    ):
        """使用Savitzky-Golay滤波器进行数据平滑

        Args:
            column_names_to_smooth: 准备施加平滑操作的列
            new_column_names: 平滑之后新的列名
            window_size: 滤波窗口大小
            poly_order: 多项式阶数
            robust_smoothing: 是否使用鲁棒平滑
        """
        self.column_names_to_smooth = column_names_to_smooth
        self.new_column_names = new_column_names
        self.window_size = window_size
        self.poly_order = poly_order
        self.robust_smoothing = robust_smoothing

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        使用Savitzky-Golay滤波器进行数据平滑，支持鲁棒平滑

        参数:
        data: 输入数据 (n_samples, n_features)
        window_size: 滤波窗口大小
        poly_order: 多项式阶数
        robust_smoothing: 是否使用鲁棒平滑

        返回:
        平滑后的数据
        """
        n_samples = len(df)

        # 调整窗口大小确保为奇数且不超过数据长度
        window_size = min(self.window_size, n_samples)
        if window_size % 2 == 0 and window_size > 1:
            window_size -= 1
        if window_size < self.poly_order + 1:
            window_size = (
                self.poly_order + 2 if self.poly_order + 2 <= n_samples else n_samples
            )
            if window_size % 2 == 0 and window_size > 1:
                window_size -= 1

        # 检查数据长度是否足够进行平滑
        if n_samples < window_size:
            print(
                f"警告: 数据长度({n_samples})小于窗口大小({window_size})，无法进行平滑，返回原始数据"
            )
            return df

        for column, new_column in zip(
            self.column_names_to_smooth, self.new_column_names
        ):
            try:
                # 提取列数据并转换为numpy数组
                column_data = df[column].values.astype(float)

                # 检查是否存在缺失值
                if pd.isna(column_data).any():
                    print(f"警告: 列 '{column}' 包含缺失值，将在平滑前进行插值")
                    # 使用线性插值填充缺失值
                    column_series = pd.Series(column_data)
                    column_data = column_series.interpolate(
                        method="linear", limit_direction="both"
                    ).values

                if self.robust_smoothing and n_samples > window_size:
                    # 使用鲁棒平滑（迭代重加权最小二乘）
                    smoothed_values = robust_savgol_filter(
                        column_data,
                        window_length=window_size,
                        polyorder=self.poly_order,
                    )
                else:
                    smoothed_values = signal.savgol_filter(
                        column_data,
                        window_length=window_size,
                        polyorder=self.poly_order,
                        mode="interp",
                    )

                df[new_column] = smoothed_values

            except Exception as e:
                print(f"列 '{column}' 平滑失败: {e}，返回原始数据")
                df[new_column] = df[column]  # 失败时返回原始数据

        return df

    def __str__(self):
        return f"SmoothingPreProcessor: smooth {self.new_column_names} from {self.column_names_to_smooth} with window_size = {self.window_size}, poly_order = {self.poly_order}, robust_smoothing = {self.robust_smoothing}."
