import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from astropy import units as u

def animate_orbit_validation(file_path, animation_interval=50):
    """
    通过动画可视化卫星轨道，并实时显示卫星位置和时间。

    Args:
        file_path (str): 包含位置/速度数据的CSV文件路径。
        animation_interval (int): 动画帧之间的间隔（毫秒），数值越小，动画越快。
    """
    print("🛰️ 开始处理轨道数据并准备动画...")

    # 1. 加载和解析数据 (与之前版本相同)
    try:
        df = pd.read_csv(file_path)
        df.columns = ['time_str', 'x', 'y', 'z', 'vx', 'vy', 'vz']
        df['time'] = pd.to_datetime(df['time_str'], errors='coerce')
        numeric_cols = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['time'] + numeric_cols, inplace=True)
        if df.empty:
            print("❌ 文件中无可用的有效数据。")
            return
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        return

    # 2. 确定并传播初始轨道 (与之前版本相同)
    initial_state = df.iloc[0]
    r0 = initial_state[['x', 'y', 'z']].values * u.km
    v0 = initial_state[['vx', 'vy', 'vz']].values * u.km / u.s
    initial_orbit = Orbit.from_vectors(Earth, r0, v0)
    
    elapsed_seconds = (df['time'] - df['time'].iloc[0]).dt.total_seconds()
    predicted_positions_list = []
    for t_sec in elapsed_seconds:
        propagated_orbit = initial_orbit.propagate(t_sec * u.s)
        predicted_positions_list.append(propagated_orbit.r.to(u.km).value)
    predicted_xyz = np.array(predicted_positions_list)

    # 3. 设置静态绘图背景
    print("正在生成3D轨道图...")
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制完整的真实与理论轨迹作为背景
    ax.plot(df['x'], df['y'], df['z'], label='Actual Path (from CSV)', color='blue', alpha=0.5)
    ax.plot(predicted_xyz[:, 0], predicted_xyz[:, 1], predicted_xyz[:, 2], 
            label='Predicted Path (from initial state)', color='red', linestyle='--', alpha=0.7)

    # 绘制地球
    u_sphere, v_sphere = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
    x_sphere = Earth.R.to(u.km).value * np.cos(u_sphere) * np.sin(v_sphere)
    y_sphere = Earth.R.to(u.km).value * np.sin(u_sphere) * np.sin(v_sphere)
    z_sphere = Earth.R.to(u.km).value * np.cos(v_sphere)
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='c', alpha=0.3)
    
    # 4. 创建动画元素
    # 初始的卫星高亮球体
    satellite_marker, = ax.plot([], [], [], 'o', color='yellow', markersize=10, markeredgecolor='black', label='Current Position')
    # 初始的时间文本
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, color='white', 
                          bbox=dict(facecolor='black', alpha=0.5))

    # 设置图表样式
    ax.set_xlabel('X (km)'), ax.set_ylabel('Y (km)'), ax.set_zlabel('Z (km)')
    ax.set_title('Animated Orbit Validation')
    ax.legend(loc='upper right')
    
    # 设置坐标轴范围
    max_range = np.array([df['x'].max()-df['x'].min(), df['y'].max()-df['y'].min(), df['z'].max()-df['z'].min()]).max() / 2.0
    mid_x, mid_y, mid_z = df[['x','y','z']].mean()
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 5. 定义动画更新函数
    def update(frame):
        # 获取当前帧的数据
        current_pos = df.iloc[frame]
        current_time = current_pos['time']
        
        # 更新卫星标记的位置
        satellite_marker.set_data_3d([current_pos['x']], [current_pos['y']], [current_pos['z']])
        
        # 更新时间文本
        time_text.set_text(f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        
        # 返回已更新的艺术家对象
        return satellite_marker, time_text

    # 6. 创建并启动动画
    # frames=len(df)表示动画的总帧数与数据点数相同
    # interval 控制帧之间的延迟（毫秒）
    # blit=True 是一种优化，可以使动画更流畅
    ani = FuncAnimation(fig, update, frames=len(df), interval=animation_interval, blit=True, repeat=True)

    plt.show()
    print("\n✅ 动画窗口已弹出。如果动画没有自动播放，请尝试拖动或缩放窗口。")

# --- 运行主函数 ---
if __name__ == '__main__':
    csv_file = 'H:\\old_D\\Work\\501_平台\\311\\S1_J2000_Position_Velocity.csv'
    # 你可以修改这里的数字来控制动画速度，50是比较适中的值
    animate_orbit_validation(csv_file, animation_interval=50)