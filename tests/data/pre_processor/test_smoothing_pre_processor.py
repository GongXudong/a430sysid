from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from scipy import signal

from a430sysid.data.pre_processor.smoothing_pre_processor import (
    SmoothingPreProcessor,
    robust_savgol_filter,
)

# 设置全局字体
plt.rcParams["font.family"] = ["SimHei", "FangSong_GB2312", "KaiTi_GB2312"]
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent


@pytest.mark.parametrize(
    "traj_file, show_figure",
    [
        (
            PROJECT_ROOT_DIR
            / "data/custom_a430_gym/1_filtered/20230215/short_u_8_20230215_143434.csv",
            False,
        ),
        (
            PROJECT_ROOT_DIR
            / "data/custom_a430_gym/1_filtered/20230215/short_u_8_20230215_145902.csv",
            False,
        ),
        (
            PROJECT_ROOT_DIR
            / "data/custom_a430_gym/1_filtered/20230215/short_u_8_20230215_150042.csv",
            False,
        ),
        (
            PROJECT_ROOT_DIR
            / "data/custom_a430_gym/1_filtered/20230215/short_u_8_20230215_150158.csv",
            False,
        ),
    ],
)
def test_smoothing_pre_processor(traj_file: Path, show_figure: bool):
    """
    增强版的平滑函数，包含平滑前后的统计信息

    参数:
    data: 输入数据 (pandas.DataFrame)
    window_size: 滤波窗口大小
    poly_order: 多项式阶数
    robust_smoothing: 是否使用鲁棒平滑

    返回:
    smoothed_data: 平滑后的数据 (pandas.DataFrame)
    statistics: 平滑统计信息 (dict)
    """

    traj_df = pd.read_csv(traj_file)

    smoother = SmoothingPreProcessor(
        column_names_to_smooth=["x", "y", "z", "phi", "theta", "psi"],
        new_column_names=[
            "x_smoothed",
            "y_smoothed",
            "z_smoothed",
            "phi_smoothed",
            "theta_smoothed",
            "psi_smoothed",
        ],
        window_size=15,
        poly_order=3,
        robust_smoothing=False,
    )

    print("=== 数据平滑处理 ===")
    print(f"输入数据形状: {traj_df.shape}")
    print(f"窗口大小: {smoother.window_size}, 多项式阶数: {smoother.poly_order}")
    print(f"鲁棒平滑: {smoother.robust_smoothing}")

    # 平滑前的统计
    original_stats = traj_df.describe()

    # 执行平滑
    smoothed_traj_df = smoother.process(traj_df)

    # 平滑后的统计
    smoothed_stats = smoothed_traj_df.describe()

    # 计算变化统计
    changes = {}
    for column, new_column in zip(
        smoother.column_names_to_smooth, smoother.new_column_names
    ):
        original_std = smoothed_traj_df[column].std()
        smoothed_std = smoothed_traj_df[new_column].std()
        change_ratio = (smoothed_std - original_std) / original_std * 100

        changes[column] = {
            "original_std": original_std,
            "smoothed_std": smoothed_std,
            "std_change_%": change_ratio,
            "max_change": (smoothed_traj_df[column] - smoothed_traj_df[new_column])
            .abs()
            .max(),
        }

    statistics = {
        "original_stats": original_stats,
        "smoothed_stats": smoothed_stats,
        "changes": changes,
    }

    # 打印统计信息
    print("\n=== 平滑效果统计 ===")
    for column, stats in changes.items():
        print(f"{column}:")
        print(f"  原始标准差: {stats['original_std']:.6f}")
        print(f"  平滑后标准差: {stats['smoothed_std']:.6f}")
        print(f"  标准差变化: {stats['std_change_%']:.2f}%")
        print(f"  最大变化量: {stats['max_change']:.6f}")

    if show_figure:
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))

        for i, (orig_col, new_col) in enumerate(
            zip(smoother.column_names_to_smooth, smoother.new_column_names)
        ):
            # 准备数据用于seaborn
            comparison_data = pd.DataFrame(
                {
                    "index": np.tile(smoothed_traj_df["Systime"].values, 2),
                    "value": np.concatenate(
                        [
                            smoothed_traj_df[orig_col].values,
                            smoothed_traj_df[new_col].values,
                        ]
                    ),
                    "type": ["original"] * len(smoothed_traj_df)
                    + ["smoothed"] * len(smoothed_traj_df),
                    "variable": [f"{orig_col}"] * (2 * len(smoothed_traj_df)),
                }
            )

            ax = axes[i // 2, i % 2]
            sns.lineplot(
                data=comparison_data,
                x="index",
                y="value",
                hue="type",
                style="type",
                ax=ax,
                markers=False,
                dashes=False,
            )
            ax.set_ylabel(orig_col)

        plt.tight_layout()
        plt.show()


def demonstrate_robust_smoothing():
    """演示鲁棒平滑与标准平滑的区别"""

    # 创建包含异常值的测试数据
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 200)
    clean_signal = np.sin(t) + 0.5 * np.cos(2 * t)

    # 添加高斯噪声
    noisy_signal = clean_signal + 0.1 * np.random.randn(len(t))

    # 添加一些异常值
    outlier_indices = [50, 80, 120, 160]
    noisy_signal[outlier_indices] += 2.0  # 大异常值

    # 应用不同平滑方法
    standard_smoothed = signal.savgol_filter(
        noisy_signal, window_length=21, polyorder=3
    )
    robust_smoothed = robust_savgol_filter(noisy_signal, window_length=21, polyorder=3)

    # 绘图比较
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t, clean_signal, "g-", label="真实信号", linewidth=2)
    plt.plot(
        t, noisy_signal, "ro", markersize=3, alpha=0.6, label="含噪声和异常值的数据"
    )
    plt.plot(t, standard_smoothed, "b-", label="标准平滑", linewidth=2)
    plt.plot(t, robust_smoothed, "m-", label="鲁棒平滑", linewidth=2)
    plt.legend()
    plt.title("标准平滑 vs 鲁棒平滑")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    plt.plot(
        t,
        np.abs(standard_smoothed - clean_signal),
        "b-",
        label="标准平滑误差",
        linewidth=2,
    )
    plt.plot(
        t,
        np.abs(robust_smoothed - clean_signal),
        "m-",
        label="鲁棒平滑误差",
        linewidth=2,
    )
    plt.legend()
    plt.title("平滑误差对比")
    plt.xlabel("时间")
    plt.ylabel("绝对误差")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 计算误差统计
    standard_error = np.mean((standard_smoothed - clean_signal) ** 2)
    robust_error = np.mean((robust_smoothed - clean_signal) ** 2)

    print(f"标准平滑的均方误差: {standard_error:.6f}")
    print(f"鲁棒平滑的均方误差: {robust_error:.6f}")
    print(f"鲁棒平滑改进: {(1 - robust_error/standard_error) * 100:.2f}%")


if __name__ == "__main__":
    test_smoothing_pre_processor(
        traj_file=PROJECT_ROOT_DIR
        / "data/custom_a430_gym/1_filtered/20230215/short_u_8_20230215_143434.csv",
        show_figure=True,
    )

    # demonstrate_robust_smoothing()
