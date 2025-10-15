from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns

from a430sysid.data.pre_processor.differential_pre_processor import (
    DifferentialPreProcessor,
)

# 设置全局字体
plt.rcParams["font.family"] = ["SimHei", "FangSong_GB2312", "KaiTi_GB2312"]
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent


@pytest.mark.parametrize(
    "traj_file, show_figure, diff_method",
    [
        (
            PROJECT_ROOT_DIR
            / "data/custom_a430_gym/1_filtered/20230215/short_u_8_20230215_143434.csv",
            False,
            "gradient",
        ),
        (
            PROJECT_ROOT_DIR
            / "data/custom_a430_gym/1_filtered/20230215/short_u_8_20230215_145902.csv",
            False,
            "gradient",
        ),
        (
            PROJECT_ROOT_DIR
            / "data/custom_a430_gym/1_filtered/20230215/short_u_8_20230215_150042.csv",
            False,
            "gradient",
        ),
        (
            PROJECT_ROOT_DIR
            / "data/custom_a430_gym/1_filtered/20230215/short_u_8_20230215_150158.csv",
            False,
            "gradient",
        ),
        (
            PROJECT_ROOT_DIR
            / "data/custom_a430_gym/1_filtered/20230215/short_u_8_20230215_143434.csv",
            False,
            "savgol",
        ),
    ],
)
def test_differential_pre_processor(
    traj_file: Path, show_figure: bool, diff_method: str
):

    traj_df = pd.read_csv(traj_file)

    columns = ["x", "y", "z", "phi", "theta", "psi"]
    new_columns = ["x_diff", "y_diff", "z_diff", "phi_diff", "theta_diff", "psi_diff"]

    differential_pre_processor = DifferentialPreProcessor(
        column_names_to_diff=columns,
        new_column_names=new_columns,
        dt=0.01,
        method=diff_method,
    )

    traj_df = differential_pre_processor.process(traj_df)

    assert set(new_columns) < set(traj_df.columns)

    if show_figure:
        print(traj_df[:10])
        plot(traj_df)


def plot(traj_df: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    for index, (column, new_column) in enumerate(
        zip(["u", "v", "w"], ["x_diff", "y_diff", "z_diff"])
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
    test_differential_pre_processor(
        traj_file=PROJECT_ROOT_DIR
        / "data/custom_a430_gym/1_filtered/20230215/short_u_8_20230215_143434.csv",
        show_figure=True,
        diff_method="savgol",
    )
