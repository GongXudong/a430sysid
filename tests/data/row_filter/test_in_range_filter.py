from pathlib import Path

import pandas as pd

from a430sysid.data.row_filter.in_range_filter import InRangeFilter

PROJECT_ROOT_PATH = Path(__file__).parent.parent.parent.parent

df_for_test = pd.DataFrame(
    data=[
        [1, 2, 3],
        [2, 4, 6],
        [3, 6, 9],
        [4, 8, 12],
    ],
    columns=["a", "b", "c"],
)

# transition
# (1, 2) + (3) -> (2, 4)
# (2, 4) + (6) -> (3, 6)
# (3, 6) + (9) -> (4, 8)

observation_keys_for_test = ["a", "b"]
action_keys_for_test = ["c"]


def test_in_range_filter_mock_1():
    z_filter = InRangeFilter(
        observation_keys=observation_keys_for_test,
        action_keys=action_keys_for_test,
        columns_to_check=["b", "c"],
        min_values=[2, 6],
        max_values=[6, 12],
    )

    obs_df = df_for_test.iloc[:-1][observation_keys_for_test]
    act_df = df_for_test.iloc[:-1][action_keys_for_test]
    next_obs_df = df_for_test[1:][observation_keys_for_test]

    filtered_obs_df, filtered_act_df, filtered_next_obs_df = z_filter.filter(
        obs_df=obs_df, act_df=act_df, next_obs_df=next_obs_df
    )

    assert filtered_obs_df.reset_index(drop=True).equals(
        pd.DataFrame(data=[[2, 4]], columns=["a", "b"])
    )
    assert filtered_act_df.reset_index(drop=True).equals(
        pd.DataFrame(data=[[6]], columns=["c"])
    )
    assert filtered_next_obs_df.reset_index(drop=True).equals(
        pd.DataFrame(data=[[3, 6]], columns=["a", "b"])
    )


def test_in_range_filter_mock_2():
    z_filter = InRangeFilter(
        observation_keys=observation_keys_for_test,
        action_keys=action_keys_for_test,
        columns_to_check=["b", "c"],
        min_values=[1, 7],
        max_values=[5, 13],
    )

    obs_df = df_for_test.iloc[:-1][observation_keys_for_test]
    act_df = df_for_test.iloc[:-1][action_keys_for_test]
    next_obs_df = df_for_test[1:][observation_keys_for_test]

    filtered_obs_df, filtered_act_df, filtered_next_obs_df = z_filter.filter(
        obs_df=obs_df, act_df=act_df, next_obs_df=next_obs_df
    )

    assert (
        len(filtered_obs_df) == len(filtered_act_df) == len(filtered_next_obs_df) == 0
    )


def test_in_range_filter_z():
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

    obs_df = traj_df.iloc[:-1][observation_keys]
    act_df = traj_df.iloc[:-1][action_keys]
    next_obs_df = traj_df[1:][observation_keys]

    filtered_obs_df, filtered_act_df, filtered_next_obs_df = z_filter.filter(
        obs_df=obs_df,
        act_df=act_df,
        next_obs_df=next_obs_df,
    )

    assert len(filtered_obs_df) == len(filtered_act_df) == len(filtered_next_obs_df)

    print(len(z_filter.filter_row_index))
    print(z_filter.filter_row_index)

    for index, v in enumerate(z_filter.filter_row_index):
        if index + 1 < len(z_filter.filter_row_index):
            if v + 1 != z_filter.filter_row_index[index + 1]:
                print(v)


if __name__ == "__main__":
    test_in_range_filter_z()
