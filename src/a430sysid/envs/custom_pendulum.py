import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.envs.classic_control.pendulum import PendulumEnv

from a430sysid.envs.utils.env_config_mixins import DynamicsMixin, EnvConfigMixin


class CustomPendulumEnv(PendulumEnv, EnvConfigMixin, DynamicsMixin):
    def __init__(self, render_mode=None, g=10.0, m=1.0, l=1.0):
        super().__init__(render_mode, g)

        self.set_config(
            {
                "g": g,
                "m": m,
                "l": l,
            }
        )

    def get_config(self) -> dict:
        return {
            "g": self.g,
            "m": self.m,
            "l": self.l,
        }

    def set_config(self, config: dict):
        self.g = config.get("g", 10.0)
        self.m = config.get("m", 1.0)
        self.l = config.get("l", 1.0)

    @classmethod
    def get_env_from_config(cls, *args, config, **kwargs):
        return cls(*args, **config, **kwargs)

    @staticmethod
    def get_default_config() -> dict:
        return {"g": 10.0, "m": 1.0, "l": 1.0}

    @staticmethod
    def calc_next_obs(state: ObsType, action: ActType, helper_env: gym.Env) -> ObsType:
        assert isinstance(
            helper_env.unwrapped, PendulumEnv
        ), "helper_env must be an instance of PendulumEnv"

        helper_env.reset()

        # Pendulum环境中，state属性存储的是[角度、角速度]，而step函数返回的obs是[角度的cos值、角度的sin值、角速度]
        helper_env.unwrapped.state = [np.arctan2(state[1], state[0]), state[2]]

        next_obs, reward, terminated, truncated, info = helper_env.step(action)
        return next_obs
