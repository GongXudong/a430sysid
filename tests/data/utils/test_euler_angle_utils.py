from pathlib import Path

import numpy as np
import pandas as pd

from a430sysid.utils.euler_angle_utils import (
    NED_velocities_to_BCS_velocities,
    unwrap_euler_angle,
    wrap_euler_angle,
)

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent


def test_wrap_euler_angle():

    # 生成1000个[-1000, 1000]的实数
    angles_degree = (np.random.rand(1000) - 0.5) * 2 * 1000

    wrapped_euler_angles_deg = wrap_euler_angle(angles_degree)

    assert np.all(
        (wrapped_euler_angles_deg <= 180.0) & (wrapped_euler_angles_deg > -180.0)
    )
    assert np.allclose(
        np.sin(np.deg2rad(angles_degree)),
        np.sin(np.deg2rad(wrapped_euler_angles_deg)),
        atol=1e-6,
    )
    assert np.allclose(
        np.cos(np.deg2rad(angles_degree)),
        np.cos(np.deg2rad(wrapped_euler_angles_deg)),
        atol=1e-6,
    )

    angles_rad = np.deg2rad((np.random.rand(1000) - 0.5) * 2 * 1000)

    wrapped_euler_angles_rad = wrap_euler_angle(angles_rad, use_rad=True)

    assert np.all(
        (wrapped_euler_angles_rad <= np.pi) & (wrapped_euler_angles_rad > -np.pi)
    )
    assert np.allclose(np.sin(angles_rad), np.sin(wrapped_euler_angles_rad))
    assert np.allclose(np.cos(angles_rad), np.cos(wrapped_euler_angles_rad))


def test_unwrap_euler_angle():
    df = pd.read_csv(PROJECT_ROOT_DIR / "tests/data/short_loop_6_20230227_120302.csv")

    psi_raw = df.iloc[:]["psi"]
    # print(psi_raw.to_numpy())

    psi_unwrapped = unwrap_euler_angle(psi_raw.to_numpy())

    for i, j in zip(psi_raw, psi_unwrapped):
        print(i, j)


def test_euler_rates_to_body_rates():
    pass


def test_NED_velocities_to_BCS_velocities():
    # df = pd.read_csv(PROJECT_ROOT_DIR / "tests/data/short_loop_6_20230227_120302.csv")
    df = pd.read_csv(
        PROJECT_ROOT_DIR / "tests/data/short_straight_4_20230215_133852.csv"
    )

    t_arrays = np.arange(len(df)) * 0.01
    x_dot = np.gradient(df["x"], t_arrays)
    y_dot = np.gradient(df["y"], t_arrays)
    z_dot = np.gradient(df["z"], t_arrays)
    u, v, w = NED_velocities_to_BCS_velocities(
        x_dot=x_dot,
        y_dot=y_dot,
        z_dot=z_dot,
        phi=np.deg2rad(df["phi"].to_numpy()),
        theta=np.deg2rad(df["theta"].to_numpy()),
        psi=np.deg2rad(df["psi"].to_numpy()),
    )

    print(f"{'u'.center(22)}        {'v'.center(22)}        {'w'.center(22)}")
    for (
        u_calc,
        v_calc,
        w_calc,
        x_dot_calc,
        y_dot_calc,
        z_dot_calc,
        u_ori,
        v_ori,
        w_ori,
    ) in zip(
        u,
        v,
        w,
        x_dot,
        y_dot,
        z_dot,
        df["u"].to_numpy(),
        df["v"].to_numpy(),
        df["w"].to_numpy(),
    ):
        print(
            f"{u_calc:>6.3f}, {x_dot_calc:>6.3f}, {u_ori:>6.3f},       {v_calc:>6.3f}, {y_dot_calc:>6.3f}, {v_ori:>6.3f},        {w_calc:>6.3f}, {z_dot_calc:>6.3f}, {w_ori:>6.3f}"
        )


if __name__ == "__main__":
    test_NED_velocities_to_BCS_velocities()
