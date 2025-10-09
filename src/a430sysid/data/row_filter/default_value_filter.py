import pandas as pd

from a430sysid.data.row_filter.filter_base import FilterBase


class DefaultValueFilter(FilterBase):
    """如果obs与next_obs有超过某个数量的列数值相等，则认为该transition无效"""

    def __init__(
        self,
        observation_keys: list[str],
        action_keys: list[str],
        same_value_column_max_num: int,
    ):
        super().__init__(observation_keys, action_keys)

        # 最大的值相等的列数
        self.same_value_column_max_num = same_value_column_max_num

    def judge(self, obs: pd.Series, act: pd.Series, next_obs: pd.Series, **kwargs):
        equal_mask = obs == next_obs

        if equal_mask.sum() >= self.same_value_column_max_num:
            return False
        else:
            return True
