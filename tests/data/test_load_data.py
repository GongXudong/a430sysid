from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from a430sysid.utils.load_data import (
    load_data_for_action_list,
    load_data_for_action_list_recursively_from_csv_files,
    load_data_from_trajectory,
)

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent


def test_load_data_for_action_list():
    obs_list_real, act_list_real, next_obs_list_real = load_data_for_action_list(
        data_dir=PROJECT_ROOT_DIR / "tests/data/data_used_in_tests/custom_cartpole",
        action_list_len=5,
    )
    print(
        f"obs_list_real shape: {obs_list_real.shape}, act_list_real shape: {act_list_real.shape}, next_obs_list_real shape: {next_obs_list_real.shape}"
    )
    assert len(obs_list_real.shape) == 2
    assert len(next_obs_list_real.shape) == 3
    assert (
        obs_list_real.shape[0] == act_list_real.shape[0] == next_obs_list_real.shape[0]
    )
    assert act_list_real.shape[1] == next_obs_list_real.shape[1] == 5

    print(obs_list_real[0])
    print(act_list_real[0])
    print(next_obs_list_real[0])


@pytest.mark.parametrize(
    "traj_path",
    [
        PROJECT_ROOT_DIR
        / "tests/data/data_used_in_tests/custom_a430py/short_straight_4_20230306_102637.csv",
        PROJECT_ROOT_DIR
        / "tests/data/data_used_in_tests/custom_a430py/short_loop_6_20230227_120302.csv",
    ],
)
def test_load_data_from_trajectory_1(traj_path: Path):

    traj_df = pd.read_csv(traj_path)

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

    obs_list, act_list, next_obs_list = load_data_from_trajectory(
        data_path=traj_path,
        observation_keys=observation_keys,
        action_keys=action_keys,
    )

    assert obs_list.shape[0] == traj_df.shape[0] - 1, ""
    assert act_list.shape[0] == traj_df.shape[0] - 1, ""
    assert next_obs_list.shape[0] == traj_df.shape[0] - 1, ""

    for obs, next_obs in zip(obs_list[1:], next_obs_list[:-1]):
        assert np.allclose(obs, next_obs), ""


@pytest.mark.parametrize(
    "root_dir, action_list_len",
    [
        (
            PROJECT_ROOT_DIR
            / "tests/data/data_used_in_tests/custom_a430py/2_processed",
            1,
        ),
        (
            PROJECT_ROOT_DIR
            / "tests/data/data_used_in_tests/custom_a430py/2_processed",
            2,
        ),
    ],
)
def test_load_data_for_action_list_recursively_from_csv_files(
    root_dir: Path,
    action_list_len: int,
):
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

    obs_list, act_list, next_obs_list = (
        load_data_for_action_list_recursively_from_csv_files(
            root_dir=root_dir,
            observation_keys=observation_keys,
            action_keys=action_keys,
            action_list_len=action_list_len,
        )
    )

    print(f"{obs_list.shape}, {act_list.shape}, {next_obs_list.shape}")


if __name__ == "__main__":
    test_load_data_for_action_list_recursively_from_csv_files(
        root_dir=PROJECT_ROOT_DIR
        / "tests/data/data_used_in_tests/custom_a430py/2_processed",
        action_list_len=1,
    )
