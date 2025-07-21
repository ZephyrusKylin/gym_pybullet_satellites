# environment/core/constants.py

"""
本文件定义了整个仿真项目中使用的常量。

它作为所有“魔法数字”的唯一来源，以提高代码的可读性和可维护性。
所有长度单位默认为 km，时间单位为 s，除非特别注明。
"""

from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth

# ===================================================================
# 物理常数 (Physical Constants)
# 我们从 poliastro 中引用，并赋予我们自己的变量名，
# 这样做的好处是，如果未来更换库或使用更精确的值，只需修改此处。
# ===================================================================

# 地球标准引力常数 (km^3 / s^2)
# 地球引力常数 (可能已存在)
EARTH_MU = 398600.4418 * (u.km**3 / u.s**2)

# -- 这是需要修改或添加的行 --
# 将地球半径定义为一个带单位的物理量
EARTH_RADIUS = 6378.137 * u.km

# 地球 J2 摄动系数 (无量纲)
EARTH_J2 = Earth.J2.value


# --- 仿真博弈常量 ---

# 最大的合理速度增量，用于机动规划的现实性检验
# 超过此值的机动方案将被认为是工程上不可行的
MAX_REASONABLE_DELTA_V = 10 * (u.km / u.s)

# ===================================================================
# 仿真与博弈常数 (Simulation & Game Constants)
# ===================================================================

# 默认的仿真时间步长 (秒)
DEFAULT_SIM_STEP_S = 60.0

# 默认的仿真起始历元
DEFAULT_EPOCH = Time("2025-07-16 00:00:00", scale="utc")

# 判定“拦截/打击”成功的最大接近距离 (km)
INTERCEPT_PROXIMITY_KM = 10.0

# 判定“伴飞/绕飞”成功的最大距离 (km)
ESCORT_PROXIMITY_KM = 50.0

# 观测传感器的最大有效观测距离 (km)
OBSERVATION_SENSOR_RANGE_KM = 2000.0

# 队形维持时，僚机与长机允许的最大位置偏差 (km)
FORMATION_MAX_DEVIATION_KM = 5.0