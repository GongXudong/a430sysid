import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf

# Add the parent directory to the system path
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))

import a430sysid.envs
from a430sysid.envs.custom_pendulum import CustomPendulumEnv

gym.register_envs(a430sysid.envs)


def test_init_1():
    print("test init 1:")

    g = 10.0
    m = 1.0
    l = 1.0

    env = gym.make("CustomPendulum-v0", g=g, m=m, l=l)

    assert env.unwrapped.g == g, f"g not equal: {env.unwrapped.g}, {g}"
    assert env.unwrapped.m == m, f"m not equal: {env.unwrapped.m}, {m}"
    assert env.unwrapped.l == l, f"l not equal: {env.unwrapped.l}, {l}"


def test_init_2():
    print("test init 2:")

    conf_dir = (
        PROJECT_ROOT_DIR
        / "tests"
        / "envs"
        / "env_configs"
        / "custom_pendulum_config.yaml"
    )
    conf = OmegaConf.load(conf_dir)

    env = gym.make(id=conf["env"]["id"], **conf["env"]["config"])

    assert (
        env.unwrapped.g == conf.env.config.g
    ), f"g not equal: {env.unwrapped.g}, {conf.env.config.g}"
    assert (
        env.unwrapped.m == conf.env.config.m
    ), f"m not equal: {env.unwrapped.m}, {conf.env.config.m}"
    assert (
        env.unwrapped.l == conf.env.config.l
    ), f"l not equal: {env.unwrapped.l}, {conf.env.config.l}"


def test_reset_1():
    print("test reset:")

    g_1 = 10.0
    m_1 = 1.0
    l_1 = 1.0

    env_1 = gym.make("CustomPendulum-v0", g=g_1, m=m_1, l=l_1)
    obs_1, info_1 = env_1.reset()


def test_get_env_from_config_1():
    print("test get_env_from_config 1:")

    custom_config = {"g": 10.0, "m": 1.0, "l": 1.0}

    env = CustomPendulumEnv.get_env_from_config(config=custom_config)

    assert (
        env.unwrapped.g == custom_config["g"]
    ), f"g not equal: {env.unwrapped.g}, {custom_config['g']}"
    assert (
        env.unwrapped.m == custom_config["m"]
    ), f"m not equal: {env.unwrapped.m}, {custom_config['m']}"
    assert (
        env.unwrapped.l == custom_config["l"]
    ), f"l not equal: {env.unwrapped.l}, {custom_config['l']}"


def test_get_env_from_config_2():
    print("test get_env_from_config 2:")

    custom_config = {"g": 10.0, "m": 1.0, "l": 1.0}

    env = CustomPendulumEnv.get_env_from_config(config=custom_config)

    assert (
        env.unwrapped.g == custom_config["g"]
    ), f"g not equal: {env.unwrapped.g}, {custom_config['g']}"
    assert (
        env.unwrapped.m == custom_config["m"]
    ), f"m not equal: {env.unwrapped.m}, {custom_config['m']}"
    assert (
        env.unwrapped.l == custom_config["l"]
    ), f"l not equal: {env.unwrapped.l}, {custom_config['l']}"

    custom_config["g"] = 9.8
    custom_config["m"] = 0.5
    custom_config["l"] = 0.8

    env2 = CustomPendulumEnv.get_env_from_config(config=custom_config)

    assert (
        env2.unwrapped.g == custom_config["g"]
    ), f"g not equal: {env2.unwrapped.g}, {custom_config['g']}"
    assert (
        env2.unwrapped.m == custom_config["m"]
    ), f"m not equal: {env2.unwrapped.m}, {custom_config['m']}"
    assert (
        env2.unwrapped.l == custom_config["l"]
    ), f"l not equal: {env2.unwrapped.l}, {custom_config['l']}"

    assert (
        env.unwrapped.g != env2.unwrapped.g
    ), f"g should not be equal: {env.unwrapped.g}, {env2.unwrapped.g}"
    assert (
        env.unwrapped.m != env2.unwrapped.m
    ), f"m should not be equal: {env.unwrapped.m}, {env2.unwrapped.m}"
    assert (
        env.unwrapped.l != env2.unwrapped.l
    ), f"l should not be equal: {env.unwrapped.l}, {env2.unwrapped.l}"


def test_calc_next_obs_1():
    print("test calc_next_obs 1:")

    conf_dir = (
        PROJECT_ROOT_DIR
        / "tests"
        / "envs"
        / "env_configs"
        / "custom_pendulum_config.yaml"
    )
    conf = OmegaConf.load(conf_dir)

    env = gym.make(id=conf["env"]["id"], **conf["env"]["config"])
    helper_env = gym.make(id=conf["env"]["id"], **conf["env"]["config"])

    obs, info = env.reset()

    for _ in range(10):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)

        next_obs_2 = CustomPendulumEnv.calc_next_obs(obs, action, helper_env)

        assert np.allclose(
            next_obs, next_obs_2
        ), f"next_obs not equal: {next_obs}, {next_obs_2}"

        obs = next_obs
