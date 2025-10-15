from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from a430sysid.data.pre_processor.interpolation_pre_processor import (
    InterpolationPreProcessor,
)

# 设置全局字体
plt.rcParams["font.family"] = ["SimHei", "FangSong_GB2312", "KaiTi_GB2312"]
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent


def compare_original_vs_interpolated(
    original_df: pd.DataFrame,
    interpolated_df: pd.DataFrame,
    columns_to_plot: list[str] = None,
):
    """比较原始数据和插值后的数据"""
    if columns_to_plot is None:
        columns_to_plot = original_df.columns

    n_cols = len(columns_to_plot)
    fig, axes = plt.subplots((n_cols + 1) // 2, 2, figsize=(12, 9))

    if n_cols == 1:
        axes = [axes]

    for i, column in enumerate(columns_to_plot):
        # 创建索引用于绘图
        indices = np.arange(len(original_df))

        # 标记缺失值的位置
        missing_mask = original_df[column].isnull().to_numpy()
        missing_indices = indices[missing_mask]

        axes[i // 2, i % 2].plot(
            indices,
            original_df[column],
            "o-",
            alpha=0.7,
            label="原始数据",
            markersize=3,
        )
        axes[i // 2, i % 2].plot(
            indices,
            interpolated_df[column],
            ".-",
            alpha=0.9,
            label="插值后数据",
            markersize=2,
        )

        # 高亮显示被插值的点
        if len(missing_indices) > 0:
            axes[i // 2, i % 2].scatter(
                missing_indices,
                interpolated_df[column].iloc[missing_indices],
                color="red",
                s=50,
                zorder=5,
                label="插值点",
            )

        axes[i // 2, i % 2].set_title(f"{column} - 原始数据 vs 插值后数据")
        axes[i // 2, i % 2].set_ylabel(column)
        axes[i // 2, i % 2].legend()
        axes[i // 2, i % 2].grid(True, alpha=0.3)

    # axes[-1].set_xlabel('数据点索引')
    plt.tight_layout()
    plt.show()


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
def test_interpolation_pre_processor(traj_file: Path, show_figure: bool):

    interp = InterpolationPreProcessor(
        method="cubic",
        # method="linear",
        limit_direction="both",
        interpolation_columns=[
            "x",
            "y",
            "z",
            "vt",
            "alpha",
            "beta",
            "phi",
            "theta",
            "psi",
            "p",
            "q",
            "r",
            "u",
            "v",
            "w",
        ],
    )

    try:
        df = pd.read_csv(traj_file)
        print("数据读取成功!")
        print(f"数据形状: {df.shape}")
        print("\n数据前5行:")
        print(df.head())
        print("\n数据基本信息:")
        print(df.info())
    except FileNotFoundError:
        print(f"错误: 找不到文件 {traj_file}")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    # 检测缺失值
    missing_stats = interp.detect_missing_values(df)

    if missing_stats.sum() == 0:
        print("\n数据中没有缺失值，无需处理。")
        return

    # 选择插值方法
    methods = ["linear", "cubic", "spline"]
    print(f"\n可用的插值方法: {methods}")

    # 根据数据特性选择合适的方法
    if len(df) < 10:
        method = "linear"  # 数据点少时使用线性插值
    else:
        method = "cubic"  # 数据点多时使用三次样条插值

    print(f"\n选择插值方法: {method}")

    # 进行插值
    df_interpolated = interp.process(df)

    # 显示处理前后的统计信息
    print("\n=== 处理前后对比 ===")
    for column in df.columns:
        original_missing = df[column].isnull().sum()
        interpolated_missing = df_interpolated[column].isnull().sum()
        print(f"{column}: 缺失值 {original_missing} → {interpolated_missing}")

    # 可视化比较
    if show_figure:
        print("\n生成可视化对比图...")
        compare_original_vs_interpolated(
            df, df_interpolated, columns_to_plot=["x", "y", "z", "phi", "theta", "psi"]
        )

    # 保存处理后的数据
    # output_file = 'flight_trajectory_interpolated.csv'
    # df_interpolated.to_csv(output_file, index=False)
    # print(f"\n处理后的数据已保存为: {output_file}")

    # 显示处理后的数据统计
    print("\n=== 处理后数据统计 ===")
    print(df_interpolated.describe())


if __name__ == "__main__":
    test_interpolation_pre_processor(
        PROJECT_ROOT_DIR
        / "data/custom_a430_gym/1_filtered/20230215/short_u_8_20230215_150158.csv",
        True,
    )
