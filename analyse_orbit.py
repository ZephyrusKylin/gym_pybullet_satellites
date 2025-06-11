import pandas as pd
import numpy as np

def analyze_maneuvers_from_energy(file_path, energy_change_threshold=5.0):
    """
    通过检测轨道能量的突变来分析卫星的轨道机动事件。
    新版本能正确处理第一列为日期时间格式的CSV文件。

    Args:
        file_path (str): 包含位置/速度数据的CSV文件路径。
        energy_change_threshold (float): 被认为是机动的最小能量变化阈值 (单位: km²/s²)。
    """
    print(f"🚀 正在分析卫星数据: {file_path}\n")

    # 地球引力常数 (km^3/s^2)
    MU_EARTH = 398600.4418

    # 1. 加载和准备数据
    try:
        # 先按原样读取CSV，不指定数据类型
        df = pd.read_csv(file_path, header=0) # header=0 假设第一行是列名
        
        # --- 全新的、更智能的数据类型转换 ---
        
        # 将原始列名重命名，方便后续调用
        original_columns = df.columns.tolist()
        df.columns = ['time_str', 'x', 'y', 'z', 'vx', 'vy', 'vz']

        # a) 单独处理时间列
        # 使用 to_datetime 智能解析时间字符串，无法解析的会变成 NaT (Not a Time)
        df['time'] = pd.to_datetime(df['time_str'], errors='coerce')

        # b) 单独处理数值列
        numeric_cols = ['x', 'y', 'z', 'vx', 'vy', 'vz']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # c) 删除任何在转换过程中出现问题的行（例如时间格式错误或数值列含文本）
        df.dropna(subset=['time'] + numeric_cols, inplace=True)
        # --- 数据处理结束 ---

    except FileNotFoundError:
        print(f"❌ 错误: 文件未找到 {file_path}")
        print("请确保CSV文件和Python脚本在同一个目录下。")
        return
    except Exception as e:
        print(f"❌ 读取或处理CSV文件时出错: {e}")
        return

    if df.empty:
        print("❌ 错误: 文件中没有有效的数值数据可供分析。请检查CSV文件内容和格式。")
        return

    # 为了方便理解，我们创建一个从0开始的“已用秒数”列
    df['elapsed_seconds'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()

    # 2. 计算关键物理量 (这部分逻辑不变)
    r_vec = df[['x', 'y', 'z']].values
    v_vec = df[['vx', 'vy', 'vz']].values
    df['r_mag'] = np.linalg.norm(r_vec, axis=1)
    df['v_mag'] = np.linalg.norm(v_vec, axis=1)
    df['energy'] = df['v_mag']**2 / 2 - MU_EARTH / df['r_mag']
    df['a'] = -MU_EARTH / (2 * df['energy'])
    
    # 3. 识别变轨事件
    df['energy_diff'] = df['energy'].diff().abs()
    maneuver_indices = df[df['energy_diff'] > energy_change_threshold].index

    # 4. 输出分析结果 (输出格式更新，使用更友好的时间戳)
    print("--- 🛰️ 轨道分析结果 ---")

    initial_state = df.iloc[0]
    print(f"\n## 初始轨道 (时刻: {initial_state['time'].strftime('%Y-%m-%d %H:%M:%S')})")
    print(f"  - 轨道能量: {initial_state['energy']:.2f} km²/s²")
    print(f"  - 半长轴 (a): {initial_state['a']:.2f} km")
    
    if not maneuver_indices.empty:
        print("\n" + "="*30)
        print("## 🔥 检测到轨道机动事件!")
        print("="*30)

        for i, idx in enumerate(maneuver_indices):
            state_after = df.loc[idx]
            state_before = df.loc[df.index[df.index.get_loc(idx)-1]] # 获取机动前一个点

            print(f"\n机动 #{i+1}")
            print(f"  - **变轨时刻**: {state_after['time'].strftime('%Y-%m-%d %H:%M:%S')} (数据点记录时刻)")
            
            print("\n  **变轨前状态**:")
            print(f"    - 轨道能量: {state_before['energy']:.2f} km²/s²")
            print(f"    - 半长轴 (a): {state_before['a']:.2f} km")

            print("\n  **变轨后状态**:")
            print(f"    - 轨道能量: {state_after['energy']:.2f} km²/s²")
            print(f"    - 半长轴 (a): {state_after['a']:.2f} km")
            
            v_before = state_before[['vx', 'vy', 'vz']].values
            v_after = state_after[['vx', 'vy', 'vz']].values
            delta_v = np.linalg.norm(v_after - v_before)
            print(f"\n  - **估算速度增量 (Δv)**: {delta_v:.4f} km/s")
    else:
        print("\n✅ 在提供的数据范围内未检测到明显的轨道机动。")

# --- 运行分析 ---
csv_file = 'H:\\old_D\\Work\\501_平台\\311\\S1_J2000_Position_Velocity.csv'
analyze_maneuvers_from_energy(csv_file)