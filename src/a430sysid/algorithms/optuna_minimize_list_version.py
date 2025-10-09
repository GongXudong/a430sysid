from copy import deepcopy

import numpy as np
import optuna

from a430sysid.envs.utils.env_config_mixins import DynamicsMixin


class SystemIdentificationWithOptunaListVersion:
    """Utilizing Bayesian Optimization for system identification in custom environments."""

    def __init__(
        self, current_params: dict, params_config: dict, helper_env_class: DynamicsMixin
    ):
        """_summary_

        Args:
            params_config (dict): system parameters configurations, form: {
                'param1': {
                    'range': [1, 10],
                    'search_num': 10,  # 将range多少等分
                },
                'param2': {}
            }
        """

        self.current_params = current_params  # 动力学全部的参数
        self.params_config = params_config  # 需要优化的参数的配置
        self.helper_env_class = helper_env_class

    def calc_loss(
        self,
        current_params: dict,
        obs_real: np.ndarray,
        act_list_real: np.ndarray,
        next_obs_list_real: np.ndarray,
        loss_aggrev_method: str = "last",
        loss_aggrev_gamma: float = 0.9,
    ) -> float:
        """_summary_

        Args:
            current_params (dict): _description_
            obs_real (np.ndarray): shape (N, obs_dim)
            act_list_real (np.ndarray): shape (N, k, act_dim)
            next_obs_list_real (np.ndarray): shape (N, k, obs_dim)

        Returns:
            float: _description_
        """
        assert (
            act_list_real.shape[1] == next_obs_list_real.shape[1]
        ), "The length of action list and next observation list must be the same."

        helper_env = self.helper_env_class(**current_params)
        next_obs_list_sim = np.array(
            [
                self.helper_env_class.calc_next_obs_list(
                    state=obs, action_list=act_list, helper_env=helper_env
                )
                for obs, act_list in zip(obs_real, act_list_real)
            ]
        )

        # print(f"next_obs_list: {next_obs_list_sim.shape} \n {next_obs_list_sim}")

        tmp = np.mean(
            (next_obs_list_sim - next_obs_list_real) ** 2, axis=-1
        )  # shape (N, k)

        # print(f"mean: {tmp[:, -1].shape}, {tmp[:, -1]}")

        if loss_aggrev_method == "last":
            # print("check loss: ", np.mean(tmp[:, -1]))
            # flag_arr = np.isnan(tmp[:, -1])
            # print(f"nan count: {(flag_arr == True).sum()}")
            # print(obs_real[flag_arr])
            # print(f"nan num of osb: {(np.isnan(obs_real) == True).sum()}")
            # print(nan_index := np.argwhere(np.isnan(tmp[:, -1])))
            # print(obs_real[nan_index])
            # exit()

            return np.mean(tmp[:, -1])
        elif loss_aggrev_method == "mean":
            return np.mean(tmp)
        elif loss_aggrev_method == "exp_average":
            weights = np.array([loss_aggrev_gamma**i for i in range(tmp.shape[1])])
            weights = weights / np.sum(weights)
            return np.mean(np.sum(tmp * weights, axis=-1))
        else:
            raise ValueError(f"Unknown loss_aggrev_method: {loss_aggrev_method}")

    def optimize(
        self,
        obs_real: np.ndarray,
        act_list_real: np.ndarray,
        next_obs_list_real: np.ndarray,
        loss_aggrev_method: str = "last",
        loss_aggrev_gamma: float = 0.9,
        n_trials: int = 1000,
        n_jobs: int = -1,
        seed: int = 42,
        show_progress_bar: bool = False,
    ):
        def objective(trial):
            params = {
                k: trial.suggest_float(k, v["range"][0], v["range"][1])
                for k, v in self.params_config.items()
            }
            tmp_params = deepcopy(self.current_params)
            tmp_params.update(params)

            return self.calc_loss(
                current_params=tmp_params,
                obs_real=obs_real,
                act_list_real=act_list_real,
                next_obs_list_real=next_obs_list_real,
                loss_aggrev_method=loss_aggrev_method,
                loss_aggrev_gamma=loss_aggrev_gamma,
            )

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar,
        )

        return study
