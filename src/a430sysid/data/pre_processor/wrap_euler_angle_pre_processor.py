import pandas as pd

from a430sysid.data.pre_processor.pre_processor_base import PreProcessorBase
from a430sysid.utils.euler_angle_utils import wrap_euler_angle


class WrapEulerAnglePreProcessor(PreProcessorBase):
    """将欧拉角phi、psi的范围限制在(-180, 180]。"""

    def __init__(
        self,
        euler_angle_column_names: list[str],
        wrapped_euler_angle_column_names: list[str],
        use_rad: bool = True,
    ):
        self.euler_angle_column_names = euler_angle_column_names
        self.wrapped_euler_angle_column_names = wrapped_euler_angle_column_names
        self.use_rad = use_rad

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        assert set(self.euler_angle_column_names) <= set(df.columns)
        # 判断列表中所有字符串是否都不在列名中
        assert all(
            new_column_name not in df.columns
            for new_column_name in self.wrapped_euler_angle_column_names
        )

        for column_name, new_column_name in zip(
            self.euler_angle_column_names, self.wrapped_euler_angle_column_names
        ):
            df[new_column_name] = wrap_euler_angle(
                euler_angle_raw=df[column_name].to_numpy(),
                use_rad=self.use_rad,
            )

        return df

    def __str__(self):
        return f"WrapEulerAnglePreProcessor: calculate {self.wrapped_euler_angle_column_names} from {self.euler_angle_column_names}."
