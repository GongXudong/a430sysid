from pathlib import Path

import pandas as pd

from a430sysid.data.row_filter.default_value_filter import DefaultValueFilter
from a430sysid.data.row_filter.differential_correlation_filter import (
    DifferentialCorrelationFilter,
)
from a430sysid.data.row_filter.filter_base import FilterList
from a430sysid.data.row_filter.in_range_filter import InRangeFilter

PROJECT_ROOT_PATH = Path(__file__).parent.parent.parent.parent


def test_filter_list_1():
    traj_df_path = PROJECT_ROOT_PATH / "tests/data/short_straight_4_20230215_133852.csv"
    # traj_df_path = PROJECT_ROOT_PATH / "tests/data/short_loop_6_20230227_120302.csv"
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

    filter_list = FilterList()

    z_filter = InRangeFilter(
        observation_keys=observation_keys,
        action_keys=action_keys,
        columns_to_check=[
            "z",
        ],
        min_values=[
            -30.0,
        ],
        max_values=[
            0.0,
        ],
    )
    filter_list.add(z_filter)

    dv_filter = DefaultValueFilter(
        observation_keys=observation_keys,
        action_keys=action_keys,
        same_value_column_max_num=9,
    )
    filter_list.add(dv_filter)

    for x, x_ref in [
        ["x", "u"],
        ["y", "v"],
        ["z", "w"],
        # ["phi", "p"], ["theta", "q"], ["psi", "r"]
    ]:
        dc_filter = DifferentialCorrelationFilter(
            observation_keys=observation_keys,
            action_keys=action_keys,
            column_to_check=x,
            column_to_reference=x_ref,
            dt=0.01,
            error_threshold=2.0,
        )
        filter_list.add(dc_filter)

    obs_df = traj_df.iloc[:-1][observation_keys]
    act_df = traj_df.iloc[:-1][action_keys]
    next_obs_df = traj_df.iloc[1:][observation_keys]

    filtered_obs_df, filtered_act_df, filtered_next_obs_df = filter_list.filter(
        obs_df=obs_df,
        act_df=act_df,
        next_obs_df=next_obs_df,
    )

    print(
        f"Filtered {len(filter_list.filter_row_index)} rows:\n{filter_list.filter_row_index}"
    )

    print(len(filtered_obs_df), len(filtered_act_df), len(filtered_next_obs_df))

    assert set(filtered_obs_df.index) | set(filter_list.filter_row_index) == set(
        range(len(obs_df))
    )


if __name__ == "__main__":
    test_filter_list_1()
