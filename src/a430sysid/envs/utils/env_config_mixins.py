from typing import Any, Protocol, SupportsFloat, runtime_checkable

import gymnasium as gym
from gymnasium.core import ActType, ObsType


class EnvConfigMixin:
    """
    Mixin class to provide configuration management for environments.
    """

    def get_config(self) -> dict:
        """
        Returns the current configuration of the environment.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def set_config(self, config: dict):
        """
        Sets the configuration of the environment.

        Args:
            config (dict): Configuration dictionary to set.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @classmethod
    def get_env_from_config(cls, *args, config: dict, **kwargs) -> gym.Env:
        """
        Creates an environment instance from the provided configuration.

        Args:
            config (dict): Configuration dictionary to create the environment.

        Returns:
            gym.Env: An instance of the environment configured with the provided settings.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def get_default_config() -> dict:
        """
        Returns the default configuration for the environment.

        Returns:
            dict: Default configuration dictionary.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @classmethod
    def merge_config_with_default_config(cls, custom_config: dict) -> dict:
        """
        Merges a custom configuration with the default configuration.

        Args:
            custom_config (dict): Custom configuration to merge.

        Returns:
            dict: Merged configuration dictionary.
        """
        default_config = cls.get_default_config()
        default_config.update(custom_config)
        return default_config


class DynamicsMixin:
    """
    Mixin class to provide dynamics calculation for environments.
    """

    @staticmethod
    def calc_next_obs(state: ObsType, action: ActType, helper_env: gym.Env) -> ObsType:
        """
        Calculates the next observation based on the current state and action.

        Args:
            state: Current state of the environment.
            action: Action taken in the environment.
            helper_env: An instance of the environment to use for dynamics calculation.

        Returns:
            The next observation after taking the action.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def calc_next_obs_list(
        state: ObsType, action_list: list[ActType], helper_env: gym.Env
    ) -> list[ObsType]:
        """
        Calculates the next observations for a list of actions based on the current state.

        Args:
            state: Current state of the environment.
            action_list (list): List of actions to take in the environment.
            helper_env: An instance of the environment to use for dynamics calculation.

        Returns:
            List of next observations after taking each action.
        """
        next_obs_list = []
        for action in action_list:
            if len(next_obs_list) == 0:
                cur_obs = state
            else:
                cur_obs = next_obs_list[-1]
            next_obs = helper_env.unwrapped.calc_next_obs(
                state=cur_obs, action=action, helper_env=helper_env
            )
            next_obs_list.append(next_obs)

        return next_obs_list


@runtime_checkable
class ConfigurableEnv(Protocol):
    """
    Protocol for environments that can be configured.
    """

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]: ...

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]: ...

    def get_config(self) -> dict: ...

    def set_config(self, config: dict): ...

    @classmethod
    def get_env_from_config(cls, *args, config: dict, **kwargs) -> gym.Env: ...

    @staticmethod
    def get_default_config() -> dict: ...

    @classmethod
    def merge_config_with_default_config(cls, custom_config: dict) -> dict: ...

    @staticmethod
    def calc_next_obs(
        state: ObsType, action: ActType, helper_env: gym.Env
    ) -> ObsType: ...
