import pybullet as p
import pybullet_data
import time
import math

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

collision_shape_data_sat = p.getCollisionShapeData(satellite, -1)
print(collision_shape_data_sat)
pos, ori = p.getBasePositionAndOrientation(satellite)
print(pos, ori)

if satellite != -1:
    print("卫星模型加载成功！ID:", satellite)
else:
    print("卫星模型加载失败！")
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
