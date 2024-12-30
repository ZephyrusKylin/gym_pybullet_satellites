import pybullet as p
import pybullet_data
import time
import math


def ori_env():
# 启动物理引擎
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    
    # 地球的实际半径（单位：千米）
    earth_radius = 6371  
    # 地球模型urdf的实际半径（单位：厘米）
    earth_model_radius = 0.5  
    # 缩放后的地球半径（单位：仿真中的尺寸）
    scaled_earth_radius = 6.371  # 这里设置为100单位，调整为合适的值

    # 计算urdf模型缩放因子
    scaling_factor = scaled_earth_radius / earth_model_radius  # 需要缩放的因子

    # 加载地球模型并进行缩放
    earth_id = p.loadURDF("sphere2.urdf", [0, 0, 0], globalScaling=scaling_factor)


    collision_shape_data = p.getCollisionShapeData(earth_id, -1)
    print(collision_shape_data)



    if earth_id != -1:
        print("地球模型加载成功！ID:", earth_id)
    else:
        print("地球模型加载失败！")
    p.resetBasePositionAndOrientation(earth_id, [0, 0, 0], [0, 0, 0, 1])
    p.setGravity(0, 0, 0)
    p.setRealTimeSimulation(0)

    # 设置相机视角，调整视距、角度和目标位置
    p.resetDebugVisualizerCamera(
        cameraDistance=50,  # 调整视距，根据场景大小适配
        cameraYaw=90,        # 水平旋转角度
        cameraPitch=-30,     # 俯仰角度
        cameraTargetPosition=[0, 0, 0]  # 相机对准地球的中心
    )

    # 模拟其他操作（例如加载卫星等）
    geo_altitude = 35786  # GEO轨道高度，单位：千米
    scaled_geo_altitude = 35.786

    geo_ratio = geo_altitude / scaled_geo_altitude

    orbital_radius = scaled_earth_radius + scaled_geo_altitude  # GEO轨道半径（仿真单位）

    satellite_start_pos = [orbital_radius, 0, 0]
    satellite_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

    # 加载卫星模型并缩放
    satellite_scaling = 1  # 卫星的缩放因子
    satellite = p.loadURDF("cube_small.urdf", satellite_start_pos, satellite_start_orientation, globalScaling=satellite_scaling)

    # collision_shape_data_sat = p.getCollisionShapeData(satellite, -1)
    # print(collision_shape_data_sat)
    # pos, ori = p.getBasePositionAndOrientation(satellite)
    # print(pos, ori)

    if satellite != -1:
        print("卫星模型加载成功！ID:", satellite)
    else:
        print("卫星模型加载失败！")

    # 设置敌方卫星初始轨道参数
    initial_orbit_altitude = 20000  # 初始轨道高度（单位：km）
    initial_orbit_inclination = 15  # 初始轨道倾角（单位：°）
    initial_orbit_longitude = 0  # 初始轨道经度（单位：°）

    # 计算敌方卫星初始位置和速度
    initial_position = [0, 0, initial_orbit_altitude + earth_radius]
    initial_velocity = [0, math.sqrt(398600.5 / (initial_orbit_altitude + earth_radius)), 0]

    # 设置缩放后敌方卫星初始轨道参数
    initial_orbit_altitude = 20000  # 初始轨道高度（单位：km）
    initial_orbit_inclination = 15  # 初始轨道倾角（单位：°）
    initial_orbit_longitude = 0  # 初始轨道经度（单位：°）

    # 计算缩放后敌方卫星初始位置和速度
    initial_position = [0, 0, initial_orbit_altitude + earth_radius]
    initial_velocity = [0, math.sqrt(398600.5 / (initial_orbit_altitude + earth_radius)), 0]

    satellite_scaling = 1  # 卫星的缩放因子
    satellite_enemy = p.loadURDF("cube_small.urdf", satellite_start_pos, satellite_start_orientation, globalScaling=satellite_scaling)

    # collision_shape_data_sat = p.getCollisionShapeData(satellite, -1)
    # print(collision_shape_data_sat)
    # pos, ori = p.getBasePositionAndOrientation(satellite)
    # print(pos, ori)

    if satellite_enemy != -1:
        print("敌方卫星模型加载成功！ID:", satellite_enemy)
    else:
        print("敌方卫星模型加载失败！")
    # 模拟主循环
    time_step = 1 / 30  # 时间步长，30fps
    orbital_speed = math.sqrt(398600.5 / (orbital_radius * geo_ratio))  # GEO轨道速度，km/s
    speed_factor = 1000  # 调整卫星速度
    angular_speed = orbital_speed / (orbital_radius * geo_ratio) * speed_factor  # 角速度，rad/s
    angle = 0

    while True:
        # 更新卫星位置
        angle += angular_speed * time_step
        new_x = orbital_radius * math.cos(angle)
        new_y = orbital_radius * math.sin(angle)
        p.resetBasePositionAndOrientation(satellite, [new_x, new_y, 0], satellite_start_orientation)

        # 绘制卫星轨迹
        if angle > 0:
            p.addUserDebugLine([new_x, new_y, 0], [orbital_radius * math.cos(angle - angular_speed * time_step), orbital_radius * math.sin(angle - angular_speed * time_step), 0], [1, 0, 0], 200)  # 红色轨迹
            # print("卫星位置:", new_x, new_y, 0)
            # print("卫星角度:", angle)

        
        # 获取地球的当前位置和方向
        position, orientation = p.getBasePositionAndOrientation(earth_id)
        # 重置地球的位置和方向为固定值
        p.resetBasePositionAndOrientation(earth_id, [0, 0, 0], [0, 0, 0, 1])


        # 步进仿真和暂停
        p.stepSimulation()
        time.sleep(time_step)

def change_orbit():


    # 启动物理引擎
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetDebugVisualizerCamera(
        cameraDistance=50,  # 调整视距，根据场景大小适配
        cameraYaw=90,        # 水平旋转角度
        cameraPitch=-30,     # 俯仰角度
        cameraTargetPosition=[0, 0, 0]  # 相机对准地球的中心
    )
    # 地球参数
    earth_radius = 6371  # 千米
    scaled_earth_radius = 6.371  # 仿真单位
    scaling_factor = scaled_earth_radius / 0.5  # 地球模型缩放

    # 加载地球模型
    earth_id = p.loadURDF("sphere2.urdf", [0, 0, 0], globalScaling=scaling_factor)
    p.resetBasePositionAndOrientation(earth_id, [0, 0, 0], [0, 0, 0, 1])

    # GEO轨道参数
    geo_altitude = 35786  # 千米
    scaled_geo_altitude = 35.786
    geo_ratio = geo_altitude / scaled_geo_altitude
    orbital_radius = scaled_earth_radius + scaled_geo_altitude

    # 加载自主卫星
    satellite_start_pos = [orbital_radius, 0, 0]
    satellite_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    satellite = p.loadURDF("cube_small.urdf", satellite_start_pos, satellite_start_orientation)

    # 敌方卫星初始参数
    enemy_start_altitude = 20000  # 千米
    scaled_enemy_altitude = enemy_start_altitude / geo_altitude * scaled_geo_altitude
    enemy_orbital_radius = scaled_earth_radius + scaled_enemy_altitude
    enemy_start_pos = [enemy_orbital_radius, 0, 0]
    satellite_enemy = p.loadURDF("cube_small.urdf", enemy_start_pos, satellite_start_orientation)

    # 模拟参数
    T = 10  # 变轨时间（秒）
    M = 5  # GEO轨道接近距离（仿真单位）
    time_step = 1 / 60

    # 速度与加速度
    mu = 398600.5  # 地球标准引力常数
    scale_factor = scaled_geo_altitude / geo_altitude
    geo_speed = math.sqrt(mu / (earth_radius + geo_altitude)) * scale_factor  # km/s
    enemy_speed = math.sqrt(mu / (earth_radius + enemy_start_altitude)) * scale_factor


    angle = 0
    elapsed_time = 0
    # 启用物理仿真
    p.changeDynamics(satellite, -1, mass=1)
    p.changeDynamics(satellite_enemy, -1, mass=1)
    p.setRealTimeSimulation(0)

    # 初始速度设置
    p.resetBaseVelocity(satellite_enemy, linearVelocity=[0, enemy_speed, 0])

    # 仿真主循环
    while True:
        # 获取卫星位置
        satellite_pos, _ = p.getBasePositionAndOrientation(satellite)
        satellite_enemy_pos, _ = p.getBasePositionAndOrientation(satellite_enemy)

        # 绘制轨迹
        p.addUserDebugLine(satellite_start_pos, satellite_pos, [0, 0, 1], 1)  # 蓝色轨迹
        p.addUserDebugLine(enemy_start_pos, satellite_enemy_pos, [1, 0, 0], 1)  # 红色轨迹

        elapsed_time += time_step
        angle += enemy_speed / enemy_orbital_radius * time_step

        # 敌方卫星变轨
        if elapsed_time >= T:
            enemy_velocity = p.getBaseVelocity(satellite_enemy)[0]
            angle = math.atan2(enemy_velocity[1], enemy_velocity[0])
            target_velocity = [geo_speed * -math.sin(angle), geo_speed * math.cos(angle), 0]
            p.resetBaseVelocity(satellite_enemy, linearVelocity=target_velocity)

        # 自主卫星开始追击
        if elapsed_time >= T and abs(satellite_enemy_pos[0] - orbital_radius) <= M:
            chase_direction = [
                satellite_enemy_pos[0] - satellite_pos[0],
                satellite_enemy_pos[1] - satellite_pos[1],
                0
            ]
            magnitude = math.sqrt(chase_direction[0]**2 + chase_direction[1]**2)
            if magnitude > 0:
                chase_direction = [c / magnitude for c in chase_direction]

            chase_speed = geo_speed * 0.2
            chase_velocity = [chase_direction[0] * chase_speed, chase_direction[1] * chase_speed, 0]
            p.resetBaseVelocity(satellite, linearVelocity=chase_velocity)

        # 检测碰撞
        if math.dist(p.getBasePositionAndOrientation(satellite)[0], satellite_enemy_pos) < 0.5:
            print("碰撞发生，仿真结束！")
            break

        # 步进仿真
        p.stepSimulation()
        time.sleep(time_step)

def change_orbit_trust():

    # 启动物理引擎
    p.connect(p.GUI)
    p.setRealTimeSimulation(0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # 常量定义
    G = 6.674e-11  # 万有引力常数
    M_earth = 5.972e24  # 地球质量（kg）

    # 地球参数
    earth_radius = 6371e3  # 米
    scaled_earth_radius = 6.371  # 仿真单位
    scaling_factor = scaled_earth_radius / 0.5  # 地球模型缩放

    # 加载地球模型
    earth_id = p.loadURDF("sphere2.urdf", [0, 0, 0], globalScaling=scaling_factor)
    p.resetBasePositionAndOrientation(earth_id, [0, 0, 0], [0, 0, 0, 1])

    # 加载卫星模型
    satellite_mass = 1000  # kg
    satellite = p.loadURDF("cube_small.urdf", [scaled_earth_radius + 35.786, 0, 0])
    satellite_enemy = p.loadURDF("cube_small.urdf", [scaled_earth_radius + 20.0, 0, 0])

    # 参数设置
    T = 10  # 敌方变轨时间（秒）
    M = 5  # GEO轨道接近距离（仿真单位）
    time_step = 1 / 60
    elapsed_time = 0

    def compute_gravity_force(mass, pos):
        distance = math.sqrt(pos[0]**2 + pos[1]**2)
        force_magnitude = G * M_earth * mass / (distance**2)
        force_direction = [-pos[0] / distance, -pos[1] / distance, 0]
        return [f * force_magnitude for f in force_direction]

    while True:
        # 获取卫星位置
        satellite_pos, _ = p.getBasePositionAndOrientation(satellite)
        satellite_enemy_pos, _ = p.getBasePositionAndOrientation(satellite_enemy)

        # 计算地球引力
        gravity_satellite = compute_gravity_force(satellite_mass, satellite_pos)
        gravity_enemy = compute_gravity_force(satellite_mass, satellite_enemy_pos)

        # 敌方卫星变轨
        if elapsed_time >= T:
            p.applyExternalForce(satellite_enemy, -1, [0, 500, 0], satellite_enemy_pos, p.WORLD_FRAME)

        # 自主卫星追击逻辑
        if elapsed_time >= T and abs(satellite_enemy_pos[0] - (scaled_earth_radius + 35.786)) <= M:
            chase_direction = [satellite_enemy_pos[0] - satellite_pos[0], satellite_enemy_pos[1] - satellite_pos[1], 0]
            magnitude = math.sqrt(chase_direction[0]**2 + chase_direction[1]**2)
            if magnitude > 0:
                chase_direction = [c / magnitude for c in chase_direction]
                thrust = [chase_direction[0] , chase_direction[1] , 0]
                p.applyExternalForce(satellite, -1, thrust, satellite_pos, p.WORLD_FRAME)

        # 施加引力
        p.applyExternalForce(satellite, -1, gravity_satellite, satellite_pos, p.WORLD_FRAME)
        p.applyExternalForce(satellite_enemy, -1, gravity_enemy, satellite_enemy_pos, p.WORLD_FRAME)

        # 检测碰撞
        if math.dist(satellite_pos, satellite_enemy_pos) < 0.5:
            print("碰撞发生，仿真结束！")
            break

        # 步进仿真
        elapsed_time += time_step
        p.stepSimulation()
        time.sleep(time_step)



def main():
    change_orbit()

if __name__ == '__main__':
    main()