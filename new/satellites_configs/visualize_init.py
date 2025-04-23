import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import yaml



from Satellite_Rl_Env import BaseSatelliteEnv

EARTH_RADIUS = 6_371e3  # 米



def visualize_satellite_config(config_file):
    """
    可视化卫星初始位置及属性（局部区域展示，包含地球）
    :param config_file: YAML 配置文件路径
    """
    # 地球半径

    # 读取配置
    with open(config_file, 'r',encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 提取所有卫星位置
    red_positions = np.array([sat['initial_position'] for sat in config.get('red', [])])
    blue_positions = np.array([sat['initial_position'] for sat in config.get('blue', [])])
    all_positions = np.vstack((red_positions, blue_positions))

    # 计算显示区域范围：以卫星群中心为中心，添加缓冲，同时包含地球原点
    center = all_positions.mean(axis=0)
    buffer = 5e6  # 5000 km 缓冲
    mins = all_positions.min(axis=0) - buffer
    maxs = all_positions.max(axis=0) + buffer
    # 确保包含原点（地球中心）
    mins = np.minimum(mins, np.zeros(3))
    maxs = np.maximum(maxs, np.zeros(3))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制完整地球球体
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 50)
    x = EARTH_RADIUS * np.outer(np.cos(u), np.sin(v))
    y = EARTH_RADIUS * np.outer(np.sin(u), np.sin(v))
    z = EARTH_RADIUS * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='cyan', alpha=0.3, linewidth=0)

    # 类型对应颜色和大小
    marker_info = {
        'observer': {'color': 'green',  'size': 60},
        'strike':   {'color': 'red',    'size': 60},
        'defense':  {'color': 'orange', 'size': 80},
        'high_value': {'color': 'purple','size':100}
    }

    def plot_satellites(sats):
        for sat in sats:
            pos = np.array(sat['initial_position'], dtype=float)
            info = marker_info.get(sat['type'], {'color':'gray','size':50})
            ax.scatter(*pos, color=info['color'], s=info['size'], depthshade=True)
            ax.text(*(pos * 1.001), f"{sat['type'][0].upper()}{sat['id']}",
                    color=info['color'], fontsize=8)

    # 绘制卫星
    plot_satellites(config.get('red', []))
    plot_satellites(config.get('blue', []))

    # 设置显示范围
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.title('Satellite Configuration Visualization (Local Region with Earth)')
    plt.show()



def simulate_no_thrust(config_file, duration_sec=7200, timestep=10.0):
    """
    模拟在无推力下卫星轨迹，并可视化运动轨迹
    :param config_file: YAML 配置文件路径
    :param duration_sec: 总仿真时长（秒）
    :param timestep: 每步控制时间步长（秒）
    """
    # 初始化环境
    env = BaseSatelliteEnv(config_file, timestep=timestep, max_steps=int(duration_sec / timestep))
    obs, info = env.reset()
    n = env.n
    steps = int(duration_sec / timestep)

    # 存储轨迹
    trajectories = np.zeros((n, steps+1, 3))
    trajectories[:, 0, :] = obs.reshape(n, 9)[:, 0:3]

    # 零推力动作
    zero_action = np.zeros((n, 4), dtype=np.float32)
    # perp 均值方向为无特殊动作
    zero_action[:, 3] = 0

    # 迭代仿真
    for t in range(steps):
        obs, reward, done, truncated, info = env.step(zero_action)
        pos = obs.reshape(n,9)[:, 0:3]
        trajectories[:, t+1, :] = pos
        if done:
            break

    # 绘图
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制地球
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = EARTH_RADIUS * np.outer(np.cos(u), np.sin(v))
    y = EARTH_RADIUS * np.outer(np.sin(u), np.sin(v))
    z = EARTH_RADIUS * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='cyan', alpha=0.2, linewidth=0)

    # 轨迹颜色
    cmap = plt.get_cmap('tab10')
    for i in range(n):
        traj = trajectories[i]
        ax.plot(traj[:,0], traj[:,1], traj[:,2], color=cmap(i), label=f'Sat {i}')
        # 起点和终点标注
        ax.scatter(*traj[0], color=cmap(i), marker='o')
        ax.scatter(*traj[-1], color=cmap(i), marker='x')

    # 设置视区
    all_pts = trajectories.reshape(-1,3)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    buffer = 5e6
    ax.set_xlim(mins[0]-buffer, maxs[0]+buffer)
    ax.set_ylim(mins[1]-buffer, maxs[1]+buffer)
    ax.set_zlim(mins[2]-buffer, maxs[2]+buffer)
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.legend()
    plt.title('Trajectories under No-Thrust Simulation')
    plt.show()
# 调用示例
# visualize_satellite_config('scenario_5v3_longrange.yaml')

# 调用示例
# visualize_satellite_config("scenario_5v3.yaml")

if __name__ == "__main__":
    
    S = BaseSatelliteEnv('5vs3.yaml', timestep=10.0, max_steps=720)
    print(S.n_red, S.n_blue)
    
    
    # visualize_satellite_config("5vs3.yaml")


    simulate_no_thrust('5vs3.yaml', duration_sec=7200, timestep=10.0)