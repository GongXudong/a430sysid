import numpy as np


def unwrap_euler_angle(
    euler_angle_raw: np.ndarray, threshold: float = 180.0, use_rad: bool = False
) -> np.ndarray:
    """
    解缠绕欧拉角，确保角度变化连续。

    参数:
    euler_angle_raw: 原始角度数据（单位：度），一维numpy数组
    threshold: 阈值，通常设为180度。对于噪声较大的数据，可以略小于180度（如175度）

    返回:
    解缠绕后的角度数据，一维numpy数组
    """
    # 假设 euler_angle_raw 是从CSV中读取的原始角度数据（单位：度）
    psi_unwrapped = np.zeros_like(euler_angle_raw)
    psi_unwrapped[0] = euler_angle_raw[0]  # 初始化第一个点

    # 用于累计需要补偿的360度倍数
    phase_correction = 0.0

    tmp_threshold = np.deg2rad(threshold) if use_rad else threshold
    tmp_one_circle = 2 * np.pi if use_rad else 360.0

    for i in range(1, len(euler_angle_raw)):
        delta = euler_angle_raw[i] - euler_angle_raw[i - 1]

        # 如果差值超过正阈值，说明发生了一个向下的跳变（例如从180度到-179度，实际变化是-359度）
        # 我们需要补偿一个+360度，使其变化量变为+1度
        if delta > tmp_threshold:
            phase_correction -= tmp_one_circle
        # 如果差值超过负阈值，说明发生了一个向上的跳变（例如从-179度到180度，实际变化是+359度）
        # 我们需要补偿一个-360度，使其变化量变为-1度
        elif delta < -tmp_threshold:
            phase_correction += tmp_one_circle

        # 应用补偿，得到解缠绕后的角度
        psi_unwrapped[i] = euler_angle_raw[i] + phase_correction

    # 现在 psi_unwrapped 就是一个连续的角度序列了
    return psi_unwrapped


def euler_rates_to_body_rates(
    phi: np.ndarray,
    theta: np.ndarray,
    phi_dot: np.ndarray,
    theta_dot: np.ndarray,
    psi_dot: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将欧拉角变化率转换为机体坐标系角速度 (p, q, r)

    参数:
    phi, theta: 平滑后的滚转角和俯仰角 (弧度)
    phi_dot, theta_dot, psi_dot: 欧拉角变化率 (弧度/秒)

    返回:
    p, q, r: 机体坐标系角速度 (弧度/秒)
    """

    # 确保输入是numpy数组
    # phi = np.asarray(phi)
    # theta = np.asarray(theta)
    # phi_dot = np.asarray(phi_dot)
    # theta_dot = np.asarray(theta_dot)
    # psi_dot = np.asarray(psi_dot)

    # 计算三角函数值
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # 避免除零错误（当theta接近±90度时）
    cos_theta_safe = np.where(
        np.abs(cos_theta) < 1e-10, np.sign(cos_theta) * 1e-10, cos_theta
    )

    # 正确的转换矩阵：欧拉角变化率 -> 机体角速度
    # [ p ]   [ 1,       0,         -sin(θ)    ]   [ ω_φ ]
    # [ q ] = [ 0,    cos(φ),    sin(φ)cos(θ) ] * [ ω_θ ]
    # [ r ]   [ 0,   -sin(φ),    cos(φ)cos(θ) ]   [ ω_ψ ]

    p = phi_dot - sin_theta * psi_dot
    q = cos_phi * theta_dot + sin_phi * cos_theta * psi_dot
    r = -sin_phi * theta_dot + cos_phi * cos_theta * psi_dot

    return p, q, r


def body_rates_to_euler_rates(
    phi: np.ndarray, theta: np.ndarray, p: np.ndarray, q: np.ndarray, r: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将机体坐标系角速度 (p, q, r) 转换为欧拉角变化率

    参数:
    phi, theta: 平滑后的滚转角和俯仰角 (弧度)
    p, q, r: 机体坐标系角速度 (弧度/秒)

    返回:
    phi_dot, theta_dot, psi_dot: 欧拉角变化率 (弧度/秒)
    """

    # phi = np.asarray(phi)
    # theta = np.asarray(theta)
    # p = np.asarray(p)
    # q = np.asarray(q)
    # r = np.asarray(r)

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    tan_theta = np.tan(theta)

    # 避免除零错误
    cos_theta_safe = np.where(
        np.abs(cos_theta) < 1e-10, np.sign(cos_theta) * 1e-10, cos_theta
    )

    # 正确的逆变换矩阵：机体角速度 -> 欧拉角变化率
    # [ ω_φ ]   [ 1,   sin(φ)tan(θ),   cos(φ)tan(θ) ]   [ p ]
    # [ ω_θ ] = [ 0,      cos(φ),         -sin(φ)    ] * [ q ]
    # [ ω_ψ ]   [ 0,   sin(φ)/cos(θ),   cos(φ)/cos(θ) ]   [ r ]

    phi_dot = p + sin_phi * tan_theta * q + cos_phi * tan_theta * r
    theta_dot = cos_phi * q - sin_phi * r
    psi_dot = (sin_phi / cos_theta_safe) * q + (cos_phi / cos_theta_safe) * r

    return phi_dot, theta_dot, psi_dot


def NED_velocities_to_BCS_velocities(
    x_dot: np.ndarray,
    y_dot: np.ndarray,
    z_dot: np.ndarray,
    phi: np.ndarray,
    theta: np.ndarray,
    psi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将东北地坐标系下的线速度转变为机体坐标系下的线速度。

    Args:
        x_dot (np.ndarray): 速度在东北地坐标系下的x分量
        y_dot (np.ndarray): 速度在东北地坐标系下的y分量
        z_dot (np.ndarray): 速度在东北地坐标系下的z分量
        phi (np.ndarray): 滚转角，单位：弧度
        theta (np.ndarray): 俯仰角，单位：弧度
        psi (np.ndarray): 偏航角，单位：弧度

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: u, v, w
    """
    # 计算三角函数值
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_psi = np.sin(psi)
    cos_psi = np.cos(psi)

    # [u] = [cosθcosψ,                cosθsinψ,                -sinθ   ]   [x_dot]
    # [v] = [sinϕsinθcosψ - cosϕsinψ, sinϕsinθsinψ + cosϕcosψ, sinϕcosθ] * [y_dot]
    # [w] = [cosϕsinθcosψ + sinϕsinψ, cosϕsinθsinψ - sinϕcosψ, cosϕcosθ]   [z_dot]
    u = (
        (cos_theta * cos_psi) * x_dot
        + (cos_theta * sin_psi) * y_dot
        - sin_theta * z_dot
    )
    v = (
        (sin_phi * sin_theta * cos_psi - cos_phi * sin_psi) * x_dot
        + (sin_phi * sin_theta * sin_psi + cos_phi * cos_psi) * y_dot
        + (sin_phi * cos_theta) * z_dot
    )
    w = (
        (cos_phi * sin_theta * cos_psi + sin_phi * sin_psi) * x_dot
        + (cos_phi * sin_theta * sin_psi - sin_phi * cos_psi) * y_dot
        + (cos_phi * cos_theta) * z_dot
    )

    return u, v, w
