from pathlib import Path

import pandas as pd
import pytest

from a430sysid.data.pre_processor.unwrap_euler_angle_pre_processor import (
    UnwrapEulerAnglePreProcessor,
)

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent


@pytest.mark.parametrize(
    "traj_file",
    [
        (
            PROJECT_ROOT_DIR
            / "tests/data/data_used_in_tests/custom_a430py/short_loop_6_20230227_120302.csv"
        ),
        (
            PROJECT_ROOT_DIR
            / "tests/data/data_used_in_tests/custom_a430py/short_straight_4_20230215_133852.csv"
        ),
        (
            PROJECT_ROOT_DIR
            / "tests/data/data_used_in_tests/custom_a430py/short_straight_4_20230306_102637.csv"
        ),
    ],
)
def test_unwrap_euler_angle_pre_processor(traj_file: Path):

    traj_df = pd.read_csv(traj_file)

    processor = UnwrapEulerAnglePreProcessor(
        euler_angle_column_names=["psi"],
        unwrapped_euler_angle_column_names=["psi_unwrapped"],
        threshold=180.0,
        use_rad=False,
    )

    traj_df = processor.process(traj_df)

    for index, row in traj_df.iterrows():
        print(row["psi"], row["psi_unwrapped"])

        if index > 0:
            # assert -100.0 < row["psi"] - traj_df["psi"][index-1] < 100.0
            assert (
                -100.0
                < row["psi_unwrapped"] - traj_df["psi_unwrapped"][index - 1]
                < 100.0
            )


if __name__ == "__main__":
    test_unwrap_euler_angle_pre_processor(
        traj_file=PROJECT_ROOT_DIR
        / "tests/data/data_used_in_tests/custom_a430py/short_loop_6_20230227_120302.csv",
    )
