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

# 设置全局字体
plt.rcParams["font.family"] = ["SimHei", "FangSong_GB2312", "KaiTi_GB2312"]
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent


@pytest.mark.parametrize(
    "traj_file, show_figure",
    [
        (
            PROJECT_ROOT_DIR
            / "tests/data/data_used_in_tests/custom_a430py/short_loop_6_20230227_120302.csv",
            False,
        ),
        (
            PROJECT_ROOT_DIR
            / "tests/data/data_used_in_tests/custom_a430py/short_straight_4_20230215_133852.csv",
            False,
        ),
        (
            PROJECT_ROOT_DIR
            / "tests/data/data_used_in_tests/custom_a430py/short_straight_4_20230306_102637.csv",
            False,
        ),
    ],
)
def test_calc_p_q_r_pre_processor(traj_file: Path, show_figure: bool):

    traj_df = pd.read_csv(traj_file)

    differential_pre_processor = DifferentialPreProcessor(
        column_names_to_diff=["x", "y", "z", "phi", "theta", "psi"],
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
        # method="gradient",
    )

    calc_p_q_r_pre_processor = CalcPQRPreProcessor(
        phi_rate_column_name="phi_diff",
        theta_rate_column_name="theta_diff",
        psi_rate_column_name="psi_diff",
        phi_column_name="phi",
        theta_column_name="theta",
        new_p_column_name="p_calc",
        new_q_column_name="q_calc",
        new_r_column_name="r_calc",
    )

    traj_df = differential_pre_processor.process(traj_df)
    traj_df = calc_p_q_r_pre_processor.process(traj_df)

    if show_figure:
        plot(traj_df[:350])


def plot(traj_df: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    for index, (column, new_column) in enumerate(
        zip(["p", "q", "r"], ["p_calc", "q_calc", "r_calc"])
    ):
        comparison_data = pd.DataFrame(
            {
                "index": np.tile(traj_df["Systime"].values, 2),
                "value": np.concatenate(
                    [traj_df[column].values, traj_df[new_column].values]
                ),
                "type": ["original"] * len(traj_df) + ["diff"] * len(traj_df),
                "variable": [f"{column}"] * (2 * len(traj_df)),
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
    test_calc_p_q_r_pre_processor(
        traj_file=PROJECT_ROOT_DIR
        / "tests/data/data_used_in_tests/custom_a430py/1_filtered/short_u_8_20230215_143434.csv",
        show_figure=True,
    )
