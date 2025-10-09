from a430py.env.a430_gym import A430Gym
from a430py.simulator.a430_sim import A430Simulator

from a430sysid.envs.utils.env_config_mixins import DynamicsMixin, EnvConfigMixin


class CustomA430Gym(A430Gym, EnvConfigMixin, DynamicsMixin):
    def __init__(
        self,
        # plane_const, 8个
        S: float = 0.040809,
        cbar: float = 0.09781,
        B: float = 0.43,
        m: float = 0.1,
        Jx: float = 0.00400,
        Jy: float = 0.00732,
        Jz: float = 0.01093,
        Jxz: float = 0.00014,
        # aero_coeffs, 27个
        CL0: float = 0.2,
        CLq: float = 6.898814,
        CLal: float = 4.235972,
        CLde: float = 0.011006,
        CD0: float = 0.04735,
        CDk: float = 1.0,
        CDde: float = -0.0013,
        CDda: float = 0.000149,
        Cy0: float = 0.0,
        Cybe: float = -0.356799,
        Cyp: float = -0.230683,
        Cyr: float = 0.378474,
        Cyda: float = -0.004417,
        Cl0: float = 0.0,
        Clbe: float = -0.01363,
        Clp: float = -0.340622,
        Clr: float = 0.015922,
        Clda: float = -0.006115,
        Cm0: float = 0.0,
        Cmal: float = -0.459587,
        Cmq: float = -6.644907,
        Cmde: float = -0.021453,
        Cn0: float = 0.0,
        Cnbe: float = 0.158268,
        Cnp: float = 0.110384,
        Cnr: float = -0.185416,
        Cnda: float = 0.002108,
        initial_lon: float = 120.0,
        initial_lat: float = 30.0,
        initial_alt: float = 10.0,
        initial_tas: float = 8.0,
        initial_yaw: float = 90.0,
        max_steps: int = 100,
    ):
        self.config_keys = self.__class__.get_default_config().keys()

        current_args = locals()
        self.env_config: dict = {ky: current_args[ky] for ky in self.config_keys}

        self.set_config(self.env_config)

        super().__init__(
            initial_lon,
            initial_lat,
            initial_alt,
            initial_tas,
            initial_yaw,
            max_steps,
            self.get_config(),
        )

    def get_config(self) -> dict:
        return {ky: getattr(self, ky) for ky in self.config_keys}

    def set_config(self, config):
        for ky in self.config_keys:
            setattr(self, ky, config[ky])

        #  # plane_const, 8个
        # self.S: float = config.get("S", 0.040809)
        # self.cbar: float = config.get("cbar", 0.09781)
        # self.B: float = config.get("B", 0.43)
        # self.m: float = config.get("m", 0.1)
        # self.Jx: float = config.get("Jx", 0.00400)
        # self.Jy: float = config.get("Jy", 0.00732)
        # self.Jz: float = config.get("Jz", 0.01093)
        # self.Jxz: float = config.get("Jxz", 0.00014)
        # # aero_coeffs, 27个
        # self.CL0: float = config.get("CL0", 0.2)
        # self.CLq: float = config.get("CLq", 6.898814)
        # self.CLal: float = config.get("CLal", 4.235972)
        # self.CLde: float = config.get("CLde", 0.011006)
        # self.CD0: float = config.get("CD0", 0.04735)
        # self.CDk: float = config.get("CDk", 1.0)
        # self.CDde: float = config.get("CDde", -0.0013)
        # self.CDda: float = config.get("CDda", 0.000149)
        # self.Cy0: float = config.get("Cy0", 0.0)
        # self.Cybe: float = config.get("Cybe", -0.356799)
        # self.Cyp: float = config.get("Cyp", -0.230683)
        # self.Cyr: float = config.get("Cyr", 0.378474)
        # self.Cyda: float = config.get("Cyda", -0.004417)
        # self.Cl0: float = config.get("Cl0", 0.0)
        # self.Clbe: float = config.get("Clbe", -0.01363)
        # self.Clp: float = config.get("Clp", -0.340622)
        # self.Clr: float = config.get("Clr", 0.015922)
        # self.Clda: float = config.get("Clda", -0.006115)
        # self.Cm0: float = config.get("Cm0", 0.0)
        # self.Cmal: float = config.get("Cmal", -0.459587)
        # self.Cmq: float = config.get("Cmq", -6.644907)
        # self.Cmde: float = config.get("Cmde", -0.021453)
        # self.Cn0: float = config.get("Cn0", 0.0)
        # self.Cnbe: float = config.get("Cnbe", 0.158268)
        # self.Cnp: float = config.get("Cnp", 0.110384)
        # self.Cnr: float = config.get("Cnr", -0.185416)
        # self.Cnda: float = config.get("Cnda", 0.002108)

    @classmethod
    def get_env_from_config(cls, *args, config, **kwargs):
        cls(*args, **config, **kwargs)

    @staticmethod
    def get_default_config() -> dict:
        return A430Simulator.get_default_config()

    @staticmethod
    def calc_next_obs(state, action, helper_env):
        assert isinstance(
            helper_env.unwrapped, CustomA430Gym
        ), "helper_env must be an instance of CustomA430Gym"

        helper_env.unwrapped.simulator.reset()

        tmp_state_dict = {
            ky: v for ky, v in zip(helper_env.unwrapped.observation_keys, state)
        }
        tmp_action_dict = {
            ky: v for ky, v in zip(helper_env.unwrapped.action_keys, action)
        }

        next_obs_dict = helper_env.unwrapped.simulator.step_from_customized_observation(
            obs_vt=tmp_state_dict["fTAS"],
            obs_alpha=tmp_state_dict["fAlpha"],
            obs_beta=tmp_state_dict["fBeta"],
            obs_phi=tmp_state_dict["fRoll"],
            obs_theta=tmp_state_dict["fPitch"],
            obs_psi=tmp_state_dict["fYaw"],
            obs_p=tmp_state_dict["fP"],
            obs_q=tmp_state_dict["fQ"],
            obs_r=tmp_state_dict["fR"],
            obs_h=tmp_state_dict["fAlt"],
            act_fStickLat=tmp_action_dict["fStickLat"],
            act_fStickLon=tmp_action_dict["fStickLon"],
            act_fThrottle=tmp_action_dict["fThrottle"],
            act_fRudder=tmp_action_dict["fRudder"],
            update_times=2,  # TODO: c++端debug完成后需要改成1！！！
        )

        return helper_env.unwrapped.get_observation(next_obs_dict)
