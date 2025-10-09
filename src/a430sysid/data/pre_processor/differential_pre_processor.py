import numpy as np
import pandas as pd
from scipy import signal

from a430sysid.data.pre_processor.pre_processor_base import PreProcessorBase


class DifferentialPreProcessor(PreProcessorBase):
    def __init__(
        self,
        column_names_to_diff: list[str],
        new_column_names: list[str],
        dt: float = 0.01,
        method: str = "gradient",
    ):
        self.column_names_to_diff = column_names_to_diff
        self.new_column_names = new_column_names
        self.dt = dt
        self.method = method

        assert self.method in [
            "gradient",
            "savgol",
        ], "method must be one of [gradient, savgol]."

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        t_arrays = np.arange(len(df)) * self.dt

        for column, new_column in zip(self.column_names_to_diff, self.new_column_names):
            if self.method == "gradient":
                df[new_column] = np.gradient(df[column], t_arrays)
            elif self.method == "savgol":
                # 使用Savitzky-Golay滤波器计算导数
                window_size = min(15, len(df))
                if window_size % 2 == 0:
                    window_size -= 1
                if window_size < 5:
                    window_size = 5

                try:
                    df[new_column] = signal.savgol_filter(
                        df[column],
                        window_length=window_size,
                        polyorder=3,
                        deriv=1,
                        delta=self.dt,
                    )
                except:
                    df[new_column] = np.gradient(df[column], t_arrays)

        return df

    def __str__(self):
        return f"DifferentialPreProcessor: calculate {self.new_column_names} from {self.column_names_to_diff} with dt = {self.dt}, method = {self.method}."
