from copy import deepcopy

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.envs.classic_control import CartPoleEnv

from a430sysid.envs.utils.env_config_mixins import DynamicsMixin, EnvConfigMixin


class CustomCartPoleEnv(CartPoleEnv, EnvConfigMixin, DynamicsMixin):
    def __init__(
        self,
        *args,
        gravity: float = 9.8,
        masscart: float = 1.0,
        masspole: float = 0.1,
        length: float = 0.5,
        force_mag: float = 10.0,
        tau: float = 0.02,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.set_config(
            {
                "gravity": gravity,
                "masscart": masscart,
                "masspole": masspole,
                "length": length,
                "force_mag": force_mag,
                "tau": tau,
            }
        )

    def get_config(self) -> dict:
        return {
            "gravity": self.gravity,
            "masscart": self.masscart,
            "masspole": self.masspole,
            "length": self.length,
            "force_mag": self.force_mag,
            "tau": self.tau,
        }

    def set_config(self, config: dict):
        self.gravity = config.get("gravity", 9.8)
        self.masscart = config.get("masscart", 1.0)
        self.masspole = config.get("masspole", 0.1)
        self.length = config.get("length", 0.5)
        self.force_mag = config.get("force_mag", 10.0)
        self.tau = config.get("tau", 0.02)
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length

    @classmethod
    def get_env_from_config(cls, *args, config, **kwargs):
        return cls(*args, **config, **kwargs)

    @staticmethod
    def get_default_config() -> dict:
        return {
            "gravity": 9.8,
            "masscart": 1.0,
            "masspole": 0.1,
            "length": 0.5,
            "force_mag": 10.0,
            "tau": 0.02,
        }

    @staticmethod
    def calc_next_obs(state: ObsType, action: ActType, helper_env: gym.Env) -> ObsType:
        assert isinstance(
            helper_env.unwrapped, CustomCartPoleEnv
        ), "helper_env must be an instance of CustomCartPoleEnv"

        helper_env.reset()
        helper_env.unwrapped.state = deepcopy(state)
        next_obs, reward, terminated, truncated, info = helper_env.step(action)
        return next_obs
