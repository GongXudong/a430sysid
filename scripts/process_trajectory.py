"""
1. 遍历数据文件夹中的所有csv
2. 定义pre_processors并处理所有csv
3. 保存新的csv
4. 画图并保存
"""

import os
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from a430sysid.data.pre_processor.calc_p_q_r_pre_processor import CalcPQRPreProcessor
from a430sysid.data.pre_processor.calc_u_v_w_pre_processor import CalcUVWPreProcessor
from a430sysid.data.pre_processor.differential_pre_processor import (
    DifferentialPreProcessor,
)
from a430sysid.data.pre_processor.interpolation_pre_processor import (
    InterpolationPreProcessor,
)
from a430sysid.data.pre_processor.pre_processor_base import PreProcessorList
from a430sysid.data.pre_processor.smoothing_pre_processor import SmoothingPreProcessor
from a430sysid.data.pre_processor.unwrap_euler_angle_pre_processor import (
    UnwrapEulerAnglePreProcessor,
)
from a430sysid.utils.path_utils import find_csv_files

PROJECT_ROOT_DIR = Path(__file__).parent.parent
PRE_PROCESSOR_DICT = {
    "unwrap_euler_angle_pre_processor": UnwrapEulerAnglePreProcessor,
    "interpolation_pre_processor": InterpolationPreProcessor,
    "smoothing_pre_processor": SmoothingPreProcessor,
    "differential_pre_processor": DifferentialPreProcessor,
    "calc_p_q_r_pre_processor": CalcPQRPreProcessor,
    "calc_u_v_w_pre_processor": CalcUVWPreProcessor,
}


def get_new_save_path(file_path: Path, to_replaced_str: str, new_str: str) -> Path:
    parts = list(file_path.parts)
    try:
        index = parts.index(to_replaced_str)
        # 替换目录名
        parts[index] = new_str

        # 构建新路径
        new_path = Path(*parts)

        # print(f"原始路径: {file_path}")
        # print(f"新 路 径: {new_path}")

        return new_path
    except ValueError:
        print(f"路径中未找到{to_replaced_str}目录")


@hydra.main(
    version_base=None,
    config_path="../configs/data/",
    config_name="pre_processing_config",
)
def process_all_trajectories(cfg: DictConfig):
    # 0.检查config
    # print(OmegaConf.to_yaml(cfg))
    # print(cfg.pre_processors[1])

    # 1.初始化pre_processors
    pre_processor_list = PreProcessorList()
    for pre_precessor in cfg.pre_processors:
        tmp = PRE_PROCESSOR_DICT[pre_precessor["pre_processor_name"]](
            **pre_precessor["args"]
        )
        pre_processor_list.add(tmp)

    # print(str(pre_processor_list))

    # 2.遍历所有csv文件
    # print('\n'.join([str(ff) for ff in find_csv_files("/home/gxd/code/a430sysid/data/custom_a430_gym/1_filtered")]))

    for csv_file in find_csv_files(PROJECT_ROOT_DIR / cfg.data_dir):
        print(f"Begin processing file: {str(csv_file)}...")
        traj_df = pd.read_csv(csv_file)
        traj_df = pre_processor_list.process(traj_df)

        new_file_path = get_new_save_path(
            file_path=csv_file,
            to_replaced_str=cfg.to_replaced_path_str,
            new_str=cfg.new_path_str,
        )

        if not new_file_path.parent.exists():
            os.makedirs(new_file_path.parent)

        traj_df.to_csv(new_file_path)
        print(f"Save file to: {new_file_path}")


if __name__ == "__main__":
    process_all_trajectories()
