from pathlib import Path

import pandas as pd
import pytest

from a430sysid.data.row_filter.differential_correlation_filter import (
    DifferentialCorrelationFilter,
)

PROJECT_ROOT_PATH = Path(__file__).parent.parent.parent.parent


@pytest.mark.parametrize(
    "column_to_check, column_to_reference",
    [
        ("x", "u"),
        ("y", "v"),
        ("z", "w"),
    ],
)
def test_differential_correlation_filter_1(
    column_to_check: str, column_to_reference: str
):
    traj_df_path = (
        PROJECT_ROOT_PATH
        / "tests/data/data_used_in_tests/custom_a430py/short_straight_4_20230215_133852.csv"
    )
    traj_df = pd.read_csv(traj_df_path)

    observation_keys = [
        "phi",
        "theta",
        "psi",
        "p",
        "q",
        "r",
        "x",
        "y",
        "z",
        "u",
        "v",
        "w",
    ]
    action_keys = ["da", "de", "dt"]

    dc_filter = DifferentialCorrelationFilter(
        observation_keys=observation_keys,
        action_keys=action_keys,
        column_to_check=column_to_check,
        column_to_reference=column_to_reference,
        dt=0.01,
        error_threshold=2.0,
    )

    obs_df = traj_df.iloc[:-1][observation_keys]
    act_df = traj_df.iloc[:-1][action_keys]
    next_obs_df = traj_df[1:][observation_keys]

    filtered_obs_df, filtered_act_df, filtered_next_obs_df = dc_filter.filter(
        obs_df=obs_df,
        act_df=act_df,
        next_obs_df=next_obs_df,
    )

    print(len(dc_filter.filter_row_index))
    print(dc_filter.filter_row_index)

    for index, v in enumerate(dc_filter.filter_row_index):
        if index + 1 < len(dc_filter.filter_row_index):
            if v + 1 != dc_filter.filter_row_index[index + 1]:
                print(v)


if __name__ == "__main__":
    test_differential_correlation_filter_1()
