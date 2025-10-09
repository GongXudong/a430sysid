import pandas as pd

from a430sysid.data.row_filter.filter_base import FilterBase


class InRangeFilter(FilterBase):
    """某一列的数值是否在预设范围内，超出范围则认为该transition无效"""

    def __init__(
        self,
        observation_keys: list[str],
        action_keys: list[str],
        columns_to_check: list[str],
        min_values: list[float],
        max_values: list[float],
    ):
        super().__init__(observation_keys, action_keys)

        self.columns_to_check = columns_to_check
        self.min_values = min_values
        self.max_values = max_values

        assert set(self.columns_to_check) <= set(self.observation_keys) | set(
            self.action_keys
        ), f"columns_to_check must all be in observation_keys and action_keys!"
        for t_column, t_min, t_max in zip(
            self.columns_to_check, self.min_values, self.max_values
        ):
            assert t_min <= t_max, f"{t_column}: min value must be max value!"

    def judge(
        self, obs: pd.Series, act: pd.Series, next_obs: pd.Series, **kwargs
    ) -> bool:
        flags = []
        for tmp_column, tmp_min_value, tmp_max_value in zip(
            self.columns_to_check, self.min_values, self.max_values
        ):
            if tmp_column in self.observation_keys:
                if tmp_min_value <= obs[tmp_column] <= tmp_max_value:
                    flags.append(True)
                else:
                    flags.append(False)

                if tmp_min_value <= next_obs[tmp_column] <= tmp_max_value:
                    flags.append(True)
                else:
                    flags.append(False)
            elif tmp_column in self.action_keys:
                if tmp_min_value <= act[tmp_column] <= tmp_max_value:
                    flags.append(True)
                else:
                    flags.append(False)
            else:
                raise KeyError(
                    f"{tmp_column} neither in observation_keys {self.observation_keys} nor in action_keys {self.action_keys}!"
                )

        return all(flags)
