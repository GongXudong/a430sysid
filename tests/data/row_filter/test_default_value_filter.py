from pathlib import Path

import pandas as pd

from a430sysid.data.row_filter.default_value_filter import DefaultValueFilter

PROJECT_ROOT_PATH = Path(__file__).parent.parent.parent.parent


def test_default_value_filter_1():
    traj_df_path = PROJECT_ROOT_PATH / "tests/data/short_straight_4_20230215_133852.csv"
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

    dv_filter = DefaultValueFilter(
        observation_keys=observation_keys,
        action_keys=action_keys,
        same_value_column_max_num=10,
    )

    obs_df = traj_df.iloc[:-1][observation_keys]
    act_df = traj_df.iloc[:-1][action_keys]
    next_obs_df = traj_df[1:][observation_keys]

    filtered_obs_df, filtered_act_df, filtered_next_obs_df = dv_filter.filter(
        obs_df=obs_df,
        act_df=act_df,
        next_obs_df=next_obs_df,
    )

    print(len(dv_filter.filter_row_index))
    print(dv_filter.filter_row_index)

    for index, v in enumerate(dv_filter.filter_row_index):
        if index + 1 < len(dv_filter.filter_row_index):
            if v + 1 != dv_filter.filter_row_index[index + 1]:
                print(v)


if __name__ == "__main__":
    test_default_value_filter_1()
