import numpy as np
import pandas as pd

from a430sysid.data.pre_processor.pre_processor_base import PreProcessorBase
from a430sysid.utils.euler_angle_utils import euler_rates_to_body_rates


class CalcPQRPreProcessor(PreProcessorBase):
    def __init__(
        self,
        phi_rate_column_name: str,
        theta_rate_column_name: str,
        psi_rate_column_name: str,
        phi_column_name: str,
        theta_column_name: str,
        new_p_column_name: str,
        new_q_column_name: str,
        new_r_column_name: str,
    ):
        self.phi_rate_column_name = phi_rate_column_name
        self.theta_rate_column_name = theta_rate_column_name
        self.psi_rate_column_name = psi_rate_column_name
        self.phi_column_name = phi_column_name
        self.theta_column_name = theta_column_name
        self.new_p_column_name = new_p_column_name
        self.new_q_column_name = new_q_column_name
        self.new_r_column_name = new_r_column_name

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """根据欧拉角变化率计算p,q,r

        Args:
            df (pd.DataFrame): 其中的欧拉角、欧拉角变化率都是基于degree

        Returns:
            pd.DataFrame: _description_
        """
        new_p, new_q, new_r = euler_rates_to_body_rates(
            phi=np.deg2rad(df[self.phi_column_name].to_numpy()),
            theta=np.deg2rad(df[self.theta_column_name].to_numpy()),
            phi_dot=np.deg2rad(df[self.phi_rate_column_name].to_numpy()),
            theta_dot=np.deg2rad(df[self.theta_rate_column_name].to_numpy()),
            psi_dot=np.deg2rad(df[self.psi_rate_column_name].to_numpy()),
        )
        df[self.new_p_column_name] = np.rad2deg(new_p)
        df[self.new_q_column_name] = np.rad2deg(new_q)
        df[self.new_r_column_name] = np.rad2deg(new_r)

        return df

    def __str__(self):
        return f"CalcPQRPreProcessor: calculate {self.new_p_column_name}, {self.new_q_column_name}, {self.new_r_column_name} from {self.phi_rate_column_name}, {self.theta_rate_column_name}, {self.psi_rate_column_name}, {self.phi_column_name}, {self.theta_column_name}."
