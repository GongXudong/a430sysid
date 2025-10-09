from pathlib import Path
from typing import List, Union


def find_csv_files(root_dir: Union[str, Path]) -> List[Path]:
    """
    遍历指定根目录及其所有子目录，查找所有CSV文件

    参数:
        root_dir: 要开始搜索的根目录路径

    返回:
        所有找到的CSV文件的Path对象列表，按路径排序
    """
    # 将输入的路径字符串转换为Path对象
    if isinstance(root_dir, Path):
        root_path = root_dir
    elif isinstance(root_dir, str):
        root_path = Path(root_dir)
    else:
        raise ValueError(f"The type of root_dir must be str or Path!")

    # 检查路径是否存在且是一个目录
    if not root_path.exists():
        raise FileNotFoundError(f"目录不存在: {root_dir}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"不是一个目录: {root_dir}")

    # 递归查找所有.csv文件，使用glob模式**匹配所有子目录
    csv_files = list(root_path.glob("**/*.csv"))

    # 按路径排序，使结果更有条理
    csv_files.sort()

    return csv_files
