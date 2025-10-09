from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest

import a430sysid.envs
from a430sysid.envs.custom_a430_gym import CustomA430Gym

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
gym.register_envs(a430sysid.envs)


@pytest.mark.parametrize(
    "config_dict_for_test",
    [
        ({"m": 100.0, "cbar": 5.0}),
        ({"m": 10.0, "cbar": 3.0, "CLq": 1.0, "Cybe": 2.0}),
        (
            {
                "m": 1.0,
                "cbar": 2.0,
                "CLq": 12.0,
                "Cybe": 33.0,
                "Cmal": 33.1,
                "Cnda": 0.222,
            }
        ),
    ],
)
def test_init_1(config_dict_for_test):
    print("In test init 1: ")

    env = CustomA430Gym(**config_dict_for_test)

    EPS = 1e-6

    config_read_from_sim = {
        **env.simulator.get_plane_const(),
        **env.simulator.get_aero_coeffs(),
    }
    print(config_read_from_sim)

    for ky in config_dict_for_test.keys():
        print(
            f"check config {ky}, {config_dict_for_test[ky]}, {env.get_config()[ky]}, {config_read_from_sim[ky]}"
        )
        assert (
            config_dict_for_test[ky] == env.get_config()[ky] == config_read_from_sim[ky]
        )
        assert env.get_config()[ky] == getattr(env, ky)


@pytest.mark.parametrize(
    "demo_traj_path",
    [
        (PROJECT_ROOT_DIR / "tests/envs/a430_trace/trace.csv"),
        (PROJECT_ROOT_DIR / "tests/envs/a430_trace/trace2.csv"),
        (PROJECT_ROOT_DIR / "tests/envs/a430_trace/trace3.csv"),
    ],
)
def test_step_1(demo_traj_path: Path):
    print("In test step 1: ")
    # use default config
    config_dict_for_test = {}
    env = CustomA430Gym(**config_dict_for_test)
    env_2 = CustomA430Gym(**config_dict_for_test)

    demo_traj = pd.read_csv(demo_traj_path)

    action_dict = {
        "fStickLat": 0.0,
        "fStickLon": -1.998228,
        "fRudder": 0.0,
        "fThrottle": 0.689030,
    }

    EPS = 1e-4

    for index, row in demo_traj.iterrows():
        if index + 1 < demo_traj.shape[0]:
            print(f"step = {index}")

            next_obs = env.calc_next_obs(
                state=env.get_observation(row),
                action=env.get_action(action_dict),
                helper_env=env_2,
            )

            next_obs_dict = {ky: v for ky, v in zip(env.observation_keys, next_obs)}

            for ky in env.observation_keys:
                print(ky, next_obs_dict[ky])
                if ky in ["fnpos", "fepos"]:
                    assert np.allclose(
                        demo_traj.iloc[index][ky] + next_obs_dict[ky],
                        demo_traj.iloc[index + 1][ky],
                        atol=EPS,
                    ), f"{ky} not equal: next obs calculated: {demo_traj.iloc[index][ky] + next_obs_dict[ky]}, next obs from demo: {demo_traj.iloc[index+1][ky]}"
                else:
                    assert np.allclose(
                        next_obs_dict[ky],
                        demo_traj.iloc[index + 1][ky],
                        atol=EPS,
                    ), f"{ky} not equal: next obs calculated: {next_obs_dict[ky]}, next obs from demo: {demo_traj.iloc[index+1][ky]}"


if __name__ == "__main__":
    test_init_1({"m": 10.0, "cbar": 3.0, "CLq": 1.0, "Cybe": 2.0})
    # test_step_1()
