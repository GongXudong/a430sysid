import numpy as np
import pandas as pd

from a430sysid.data.pre_processor.pre_processor_base import PreProcessorBase
from a430sysid.utils.euler_angle_utils import NED_velocities_to_BCS_velocities


class CalcUVWPreProcessor(PreProcessorBase):
    def __init__(
        self,
        x_dot_column_name: str,
        y_dot_column_name: str,
        z_dot_column_name: str,
        phi_column_name: str,
        theta_column_name: str,
        psi_column_name: str,
        new_u_column_name: str,
        new_v_column_name: str,
        new_w_column_name: str,
        new_vt_column_name: str,
        new_alpha_column_name: str,
        new_beta_column_name: str,
    ):
        self.x_dot_column_name = x_dot_column_name
        self.y_dot_column_name = y_dot_column_name
        self.z_dot_column_name = z_dot_column_name
        self.phi_column_name = phi_column_name
        self.theta_column_name = theta_column_name
        self.psi_column_name = psi_column_name
        self.new_u_column_name = new_u_column_name
        self.new_v_column_name = new_v_column_name
        self.new_w_column_name = new_w_column_name
        self.new_vt_column_name = new_vt_column_name
        self.new_alpha_column_name = new_alpha_column_name
        self.new_beta_column_name = new_beta_column_name

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """根据x,y,z的变化率与欧拉角计算u, v, w和vt, alpha, beta

        Args:
            df (pd.DataFrame): 其中的欧拉角的单位是degree

        Returns:
            pd.DataFrame: _description_
        """
        # calc u, v, w
        new_u, new_v, new_w = NED_velocities_to_BCS_velocities(
            x_dot=df[self.x_dot_column_name].to_numpy(),
            y_dot=df[self.y_dot_column_name].to_numpy(),
            z_dot=df[self.z_dot_column_name].to_numpy(),
            phi=np.deg2rad(df[self.phi_column_name].to_numpy()),
            theta=np.deg2rad(df[self.theta_column_name].to_numpy()),
            psi=np.deg2rad(df[self.psi_column_name].to_numpy()),
        )

        df[self.new_u_column_name] = new_u
        df[self.new_v_column_name] = new_v
        df[self.new_w_column_name] = new_w

        # calc vt, alpha, beta
        df[self.new_vt_column_name] = np.sqrt(new_u**2 + new_v**2 + new_w**2)
        df[self.new_alpha_column_name] = np.rad2deg(np.arctan2(new_w, new_u))  # degree
        df[self.new_beta_column_name] = np.rad2deg(
            np.arcsin(new_v / df[self.new_vt_column_name])
        )  # degree

        return df

    def __str__(self):
        return f"CalcUVWPreProcessor: calculate {self.new_u_column_name}, {self.new_v_column_name}, {self.new_w_column_name} from {self.x_dot_column_name}, {self.y_dot_column_name}, {self.z_dot_column_name}, {self.phi_column_name}, {self.theta_column_name}, {self.psi_column_name}."
