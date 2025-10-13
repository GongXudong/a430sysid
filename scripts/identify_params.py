import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from a430sysid.algorithms.optuna_minimize_list_version import (
    SystemIdentificationWithOptunaListVersion,
)
from a430sysid.utils.consts import HELPER_ENV_CLASS_DICT
from a430sysid.utils.load_data import (
    load_data_for_action_list_recursively_from_csv_files,
)

PROJECT_ROOT_DIR = Path(__file__).parent.parent
if str(PROJECT_ROOT_DIR.absolute()) not in sys.path:
    sys.path.append(str(PROJECT_ROOT_DIR.absolute()))


# A logger for this file
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs/sys_id", config_name="config")
def identify_params(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))

    # Load data
    (
        obs_real,
        act_real,
        next_obs_real,
    ) = load_data_for_action_list_recursively_from_csv_files(
        root_dir=PROJECT_ROOT_DIR / cfg.optimize.exciting_trajectory.data_path,
        observation_keys=[
            "phi_wrapped",
            "theta_smoothed",
            "psi_wrapped",
            "p_calc",
            "q_calc",
            "r_calc",
            "x_smoothed",
            "y_smoothed",
            "z_smoothed",
            "vt_calc",
            "alpha_calc",
            "beta_calc",
        ],
        action_keys=["da", "de", "dt"],
        insert_action_dr=True,
        insert_action_dr_index=2,
        deg2rad_columns=[
            "phi_wrapped",
            "theta_smoothed",
            "psi_wrapped",
            "p_calc",
            "q_calc",
            "r_calc",
            "alpha_calc",
            "beta_calc",
        ],
        action_list_len=cfg.optimize.optimize_config.action_list_len,
    )

    log.info(
        f"Load data success, obs_real shape: {obs_real.shape}, act_real shape: {act_real.shape}, next_obs_real shape: {next_obs_real.shape}"
    )

    # Initialize the system identification algorithm
    sys_id_algo = SystemIdentificationWithOptunaListVersion(
        current_params=cfg.optimize.env_params.current,
        params_config=cfg.optimize.env_params.to_optimize,
        helper_env_class=HELPER_ENV_CLASS_DICT[cfg.env.id],
    )
    log.info("Init SystemIdentificationWithOptunaListVersion success")

    initial_params_loss = sys_id_algo.calc_loss(
        current_params=cfg.optimize.env_params.current,
        obs_real=obs_real,
        act_list_real=act_real,
        next_obs_list_real=next_obs_real,
        loss_aggrev_method=cfg.optimize.optimize_config.loss_aggrev_method,
        loss_aggrev_gamma=cfg.optimize.optimize_config.loss_aggrev_gamma,
    )
    log.info(f"Initial params loss: {initial_params_loss}.")

    # 执行优化
    study = sys_id_algo.optimize(
        obs_real=obs_real,
        act_list_real=act_real,
        next_obs_list_real=next_obs_real,
        loss_aggrev_method=cfg.optimize.optimize_config.loss_aggrev_method,
        loss_aggrev_gamma=cfg.optimize.optimize_config.loss_aggrev_gamma,
        n_trials=cfg.optimize.optimize_config.n_trials,
        n_jobs=cfg.optimize.optimize_config.n_jobs,
        seed=cfg.optimize.optimize_config.seed,
    )

    log.info(
        f"Best parameters found: {study.best_params}, best value: {study.best_value}"
    )

    # 保存结果
    save_path = (
        PROJECT_ROOT_DIR
        / "logs"
        / "sys_id_by_optuna"
        / f"{cfg.optimize.log.exp_name}.csv"
    )
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Save all trials to {save_path}")
    study.trials_dataframe().to_csv(save_path, index=False)


# python scripts/identify_params.py optimize.optimize_config.seed=31 optimize.optimize_config.n_trials=100
# uv run scripts/identify_params.py optimize.optimize_config.seed=31 optimize.optimize_config.n_trials=100
if __name__ == "__main__":
    identify_params()
