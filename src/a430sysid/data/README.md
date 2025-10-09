# Data

## Pre-Processors

对给行轨迹的预处理器

### 1.unwrap_euler_angle_pre_processor.py

解缠绕滚转角$\phi$和偏航角$\psi$，针对滚转角和偏航角的范围为[-180, 180]，值从-180到180会发生突变的问题，该预处理器遇到突变就加360或者减360。最终达到的效果是：以连续滚转4周为例，滚转角会从0连续变到1440。

### 2.Interpolation_pre_processor.py

部分时刻的传感器值有误，剔除这部分值后，该预处理器对这些空位置进行插值。

### 3.smoothing_pre_processor.py

平滑传感器记录的轨迹，$x, y, z, \phi, \theta, \psi$.

### 4.differential_pre_processor.py

求$x, y, z, \phi, \theta, \psi$的微分。

### 5.calc_p_q_r_pre_processor.py

结合$\phi, \theta, \psi$的微分与$\phi, \theta$的值，求$p, q, r$.

### 6.calc_u_v_w_pre_processor.py

结合$x, y, z$的微分与$\phi, \theta, \psi$的值，求$u, v, w$和$vt, \alpha, \beta$.
