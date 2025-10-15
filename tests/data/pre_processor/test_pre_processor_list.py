from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns

from a430sysid.data.pre_processor.calc_p_q_r_pre_processor import CalcPQRPreProcessor
from a430sysid.data.pre_processor.differential_pre_processor import (
    DifferentialPreProcessor,
)
from a430sysid.data.pre_processor.interpolation_pre_processor import (
    InterpolationPreProcessor,
)
from a430sysid.data.pre_processor.pre_processor_base import PreProcessorList
from a430sysid.data.pre_processor.smoothing_pre_processor import SmoothingPreProcessor
from a430sysid.data.pre_processor.unwrap_euler_angle_pre_processor import (
    UnwrapEulerAnglePreProcessor,
)

# 设置全局字体
plt.rcParams["font.family"] = ["SimHei", "FangSong_GB2312", "KaiTi_GB2312"]
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent


@pytest.mark.parametrize(
    "traj_file, show_figure",
    [
        (
            PROJECT_ROOT_DIR
            / "tests/data/data_used_in_tests/custom_a430py/1_filtered/short_straight_4_20230215_142714.csv",
            False,
        ),
        (
            PROJECT_ROOT_DIR
            / "tests/data/data_used_in_tests/custom_a430py/1_filtered/short_straight_4_20230215_143034.csv",
            False,
        ),
        (
            PROJECT_ROOT_DIR
            / "tests/data/data_used_in_tests/custom_a430py/1_filtered/short_u_8_20230215_143434.csv",
            False,
        ),
        (
            PROJECT_ROOT_DIR
            / "tests/data/data_used_in_tests/custom_a430py/1_filtered/short_u_8_20230215_145902.csv",
            False,
        ),
    ],
)
def test_pre_processor_list(traj_file: Path, show_figure: bool):

    traj_df = pd.read_csv(traj_file)

    pre_processor_list = PreProcessorList()

    pre_processor_list.add(
        UnwrapEulerAnglePreProcessor(
            euler_angle_column_names=["phi", "psi"],
            unwrapped_euler_angle_column_names=["phi_unwrapped", "psi_unwrapped"],
            threshold=180.0,
        )
    )

    pre_processor_list.add(
        InterpolationPreProcessor(
            method="cubic",
            limit_direction="both",
            interpolation_columns=[
                "x",
                "y",
                "z",
                "vt",
                "alpha",
                "beta",
                "phi_unwrapped",
                "theta",
                "psi_unwrapped",
                "p",
                "q",
                "r",
                "u",
                "v",
                "w",
            ],
        )
    )

    pre_processor_list.add(
        SmoothingPreProcessor(
            column_names_to_smooth=[
                "x",
                "y",
                "z",
                "phi_unwrapped",
                "theta",
                "psi_unwrapped",
            ],
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
    )

    pre_processor_list.add(
        DifferentialPreProcessor(
            column_names_to_diff=[
                "x_smoothed",
                "y_smoothed",
                "z_smoothed",
                "phi_smoothed",
                "theta_smoothed",
                "psi_smoothed",
            ],
            new_column_names=[
                "x_diff",
                "y_diff",
                "z_diff",
                "phi_diff",
                "theta_diff",
                "psi_diff",
            ],
            dt=0.01,
            method="savgol",
        )
    )

    pre_processor_list.add(
        CalcPQRPreProcessor(
            phi_rate_column_name="phi_diff",
            theta_rate_column_name="theta_diff",
            psi_rate_column_name="psi_diff",
            phi_column_name="phi_smoothed",
            theta_column_name="theta_smoothed",
            new_p_column_name="p_calc",
            new_q_column_name="q_calc",
            new_r_column_name="r_calc",
        )
    )

    traj_df = pre_processor_list.process(traj_df)

    if show_figure:
        # 对比 unwrap+插值 前后的值
        # plot(traj_df, ["phi", "psi"], ["phi_unwrapped", "psi_unwrapped"])

        # 对比 平滑 前后的值
        # plot(traj_df, ["x", "y", "z", "phi", "theta", "psi"], ["x_smoothed", "y_smoothed", "z_smoothed", "phi_smoothed", "theta_smoothed", "psi_smoothed"])

        # 对比 微分 的值
        plot(traj_df, ["u", "v", "w"], ["x_diff", "y_diff", "z_diff"])
        plot(traj_df, ["p", "q", "r"], ["p_calc", "q_calc", "r_calc"])


def plot(
    traj_df: pd.DataFrame,
    columns: list[str] = ["p", "q", "r"],
    columns_for_compare: list[str] = ["p_calc", "q_calc", "r_calc"],
):
    fig, axes = plt.subplots(len(columns), 1, figsize=(12, 9))

    for index, (column, new_column) in enumerate(zip(columns, columns_for_compare)):
        comparison_data = pd.DataFrame(
            {
                "index": np.tile(traj_df["Systime"].values, 2),
                "value": np.concatenate(
                    [traj_df[column].values, traj_df[new_column].values]
                ),
                "type": [column] * len(traj_df) + [new_column] * len(traj_df),
            }
        )
        ax = axes[index]
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
        ax.set_ylabel(column)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_pre_processor_list(
        traj_file=PROJECT_ROOT_DIR
        / "tests/data/data_used_in_tests/custom_a430py/1_filtered/short_u_8_20230215_145902.csv",
        show_figure=True,
    )
