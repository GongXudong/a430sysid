from pathlib import Path

import numpy as np
import pandas as pd

from a430sysid.utils.path_utils import find_csv_files

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent


def load_data(
    data_dir: Path = PROJECT_ROOT_DIR / "data/custom_cartpole",
    obs_real_file_name: str = "obs_real.npy",
    act_real_file_name: str = "act_real.npy",
    next_obs_real_file_name: str = "next_obs_real.npy",
) -> list[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs_real = np.load(data_dir / obs_real_file_name, allow_pickle=True)
    act_real = np.load(data_dir / act_real_file_name, allow_pickle=True)
    next_obs_real = np.load(data_dir / next_obs_real_file_name, allow_pickle=True)
    dones = np.load(data_dir / "dones.npy", allow_pickle=True)
    return obs_real, act_real, next_obs_real, dones


def load_data_for_action_list(
    data_dir: Path = PROJECT_ROOT_DIR / "data/custom_cartpole", action_list_len: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载数据，并将动作和下一个状态转换为动作列表和下一个状态列表的形式

    Args:
        data_dir (Path, optional): 数据存放路径. Defaults to PROJECT_ROOT_DIR / "data/custom_cartpole".
        action_list_len (int, optional): 动作列表的长度. Defaults to 5.

    Returns:
        _type_: obs_list_real: shape (N, obs_dim), act_list_real: shape (N, k, act_dim), next_obs_list_real: shape (N, k, obs_dim)
    """
    obs_real, act_real, next_obs_real, dones = load_data(data_dir)

    N = obs_real.shape[0]
    obs_list_real = []
    act_list_real = []
    next_obs_list_real = []

    for i in range(N - action_list_len + 1):
        if np.any(dones[i : i + action_list_len - 1]):  # 如果这段时间内有done，则跳过
            continue
        obs_list_real.append(obs_real[i])
        act_list_real.append(act_real[i : i + action_list_len])
        next_obs_list_real.append(next_obs_real[i : i + action_list_len])

    obs_list_real = np.array(obs_list_real)
    act_list_real = np.array(act_list_real)
    next_obs_list_real = np.array(next_obs_list_real)

    return obs_list_real, act_list_real, next_obs_list_real


def load_data_for_action_list_recursively_from_csv_files(
    root_dir: Path,
    observation_keys: list = [],
    action_keys: list = [],
    insert_action_dr: bool = True,
    insert_action_dr_index: int = 2,
    deg2rad_columns: list[str] = [],
    action_list_len: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 1.从root_dir目录下递归的读取所有csv文件，拼接成obs_list, act_list, next_obs_list, done_list数组
    obs_real, act_real, next_obs_real, dones = None, None, None, None

    for csv_file in find_csv_files(root_dir):
        tmp_obs_list, tmp_act_list, tmp_next_obs_list = load_data_from_trajectory(
            csv_file, observation_keys, action_keys
        )

        # 将deg2rad_columns列的值由degree转变为rad
        for tmp_index, obs_key in enumerate(observation_keys):
            if obs_key in deg2rad_columns:
                tmp_obs_list[:, tmp_index] = np.deg2rad(tmp_obs_list[:, tmp_index])
                tmp_next_obs_list[:, tmp_index] = np.deg2rad(
                    tmp_next_obs_list[:, tmp_index]
                )

        # 插入dr (Rudder)列。数据文件中没有dr，但是env的action中包含Rudder！
        if insert_action_dr:
            dr = np.array([0.0] * tmp_act_list.shape[0])
            tmp_act_list = np.insert(tmp_act_list, insert_action_dr_index, dr, axis=1)

        if obs_real is None:
            obs_real, act_real, next_obs_real = (
                tmp_obs_list,
                tmp_act_list,
                tmp_next_obs_list,
            )
            dones = np.array([False] * (len(tmp_obs_list) - 1) + [True])
        else:
            obs_real = np.concatenate((obs_real, tmp_obs_list), axis=0)
            act_real = np.concatenate((act_real, tmp_act_list), axis=0)
            next_obs_real = np.concatenate((next_obs_real, tmp_next_obs_list), axis=0)
            dones = np.concatenate(
                (dones, np.array([False] * (len(tmp_obs_list) - 1) + [True])), axis=0
            )

    # 2.将动作和下一个状态转换为动作列表和下一个状态列表的形式
    N = obs_real.shape[0]
    obs_list_real = []
    act_list_real = []
    next_obs_list_real = []

    for i in range(N - action_list_len + 1):
        if np.any(dones[i : i + action_list_len - 1]):  # 如果这段时间内有done，则跳过
            continue
        obs_list_real.append(obs_real[i])
        act_list_real.append(act_real[i : i + action_list_len])
        next_obs_list_real.append(next_obs_real[i : i + action_list_len])

    obs_list_real = np.array(obs_list_real)
    act_list_real = np.array(act_list_real)
    next_obs_list_real = np.array(next_obs_list_real)

    return obs_list_real, act_list_real, next_obs_list_real


def load_data_from_trajectory(
    data_path: Path = PROJECT_ROOT_DIR / "data/custom_cartpole",
    observation_keys: list = [],
    action_keys: list = [],
) -> list[np.ndarray, np.ndarray, np.ndarray]:
    traj_df = pd.read_csv(data_path)

    assert set(observation_keys) <= set(
        traj_df.columns
    ), f"{observation_keys} must all be in columns of {data_path}!"
    assert set(action_keys) <= set(
        traj_df.columns
    ), f"{action_keys} must all be in columns of {data_path}!"

    obs_list = traj_df[:-1][observation_keys]
    act_list = traj_df[:-1][action_keys]
    next_obs_list = traj_df[1:][observation_keys]

    return obs_list.to_numpy(), act_list.to_numpy(), next_obs_list.to_numpy()
