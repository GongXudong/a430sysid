from abc import ABC, abstractmethod

import pandas as pd


class FilterBase(ABC):
    def __init__(self, observation_keys: list[str], action_keys: list[str]):
        self.observation_keys = observation_keys
        self.action_keys = action_keys

        # 记录被剔除的transition的index，用于debug
        self.filter_row_index: list[int] = []

    @abstractmethod
    def judge(
        self, obs: pd.Series, act: pd.Series, next_obs: pd.Series, **kwargs
    ) -> bool:
        """判断(obs, act) -> next_obs是否合理

        Args:
            obs (pd.Series): 当前观测
            act (pd.Series): 当前动作
            next_obs (pd.Series): 下一步观测

        Returns:
            bool: 合理则返回True；否则，返回False。
        """

    def filter(
        self, obs_df: pd.DataFrame, act_df: pd.DataFrame, next_obs_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        assert (
            len(obs_df) == len(act_df) == len(next_obs_df)
        ), "Length of obs_df, act_df, and next_obs_df must be same!"

        self.filter_row_index = []
        row_flags: list[bool] = []

        for (
            (obs_index, obs_row),
            (act_index, act_row),
            (next_obs_index, next_obs_row),
        ) in zip(obs_df.iterrows(), act_df.iterrows(), next_obs_df.iterrows()):
            this_row_flag = self.judge(
                obs=obs_row,
                act=act_row,
                next_obs=next_obs_row,
            )
            row_flags.append(this_row_flag)

            if not this_row_flag:
                self.filter_row_index.append(obs_index)

        return obs_df[row_flags], act_df[row_flags], next_obs_df[row_flags]


class FilterList(FilterBase):
    def __init__(self):
        self.filter_list: list[FilterBase] = []

    def judge(self, obs, act, next_obs, **kwargs):
        pass

    def add(self, filter: FilterBase):
        self.filter_list.append(filter)

    def filter(
        self, obs_df: pd.DataFrame, act_df: pd.DataFrame, next_obs_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        filtered_row_index: set[int] = set([])

        tmp_obs_df = obs_df
        tmp_act_df = act_df
        tmp_next_obs_df = next_obs_df

        for tmp_filter in self.filter_list:
            tmp_obs_df, tmp_act_df, tmp_next_obs_df = tmp_filter.filter(
                obs_df=tmp_obs_df,
                act_df=tmp_act_df,
                next_obs_df=tmp_next_obs_df,
            )

            # 集合并
            filtered_row_index |= set(tmp_filter.filter_row_index)

        self.filter_row_index = list(filtered_row_index)

        return tmp_obs_df, tmp_act_df, tmp_next_obs_df
