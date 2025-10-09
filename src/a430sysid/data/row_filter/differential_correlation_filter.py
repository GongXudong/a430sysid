import pandas as pd

from a430sysid.data.row_filter.filter_base import FilterBase


class DifferentialCorrelationFilter(FilterBase):
    """如果某一列的微分与参考列相差过大，则认为该transition无效"""

    def __init__(
        self,
        observation_keys: list[str],
        action_keys: list[str],
        column_to_check: str,
        column_to_reference: str,
        dt: float,
        error_threshold: float,
    ):
        super().__init__(observation_keys, action_keys)

        self.column_to_check = column_to_check
        self.column_to_reference = column_to_reference
        self.dt = dt
        self.error_threshold = error_threshold

        assert (
            self.column_to_check in self.observation_keys
        ), f"{self.column_to_check} must be in observation_keys: {self.observation_keys}"
        assert (
            self.column_to_reference in self.observation_keys
        ), f"{self.column_to_reference} must be in observation_keys: {self.observation_keys}"

    def judge(
        self, obs: pd.Series, act: pd.Series, next_obs: pd.Series, **kwargs
    ) -> bool:
        dx_dt = (next_obs[self.column_to_check] - obs[self.column_to_check]) / self.dt

        dx_dt_ref = obs[self.column_to_reference]

        if abs(dx_dt - dx_dt_ref) <= abs(self.error_threshold):
            return True
        else:
            return False
