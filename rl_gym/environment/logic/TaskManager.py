# environment/logic/task_manager.py

"""
本模块定义了 TaskManager 类，它是 logic 层的核心状态管理器。

其职责是接收由 maneuver_planner 生成的 ManeuverPlan，将其转化为
一个有状态的、可追踪的 ActiveTask，并在仿真时间流中精确地执行
任务中的每一次机动。
"""

from dataclasses import dataclass, field
from typing import Dict, List
from enum import Enum, auto

from astropy.time import Time, TimeDelta
from poliastro.twobody import Orbit

from environment.core.satellite import Satellite
from environment.logic.maneuver_planner import ManeuverPlan, ManeuverExecution
import numpy as np
class TaskStatus(Enum):
    """描述一个活动任务的当前状态。"""
    PENDING = auto()    # 任务已创建，等待执行第一个机动
    IN_PROGRESS = auto()# 任务正在执行中（适用于多脉冲机动）
    COMPLETED = auto()  # 所有机动已成功执行
    FAILED = auto()     # 任务因故失败（如燃料不足）
    CANCELLED = auto() # 新增：任务被外部指令主动取消
    
@dataclass
class ActiveTask:
    """
    一个内部数据结构，用于封装和追踪一个正在进行的具体任务。
    它将一个 ManeuverPlan 与一个特定的卫星关联起来，并维护其执行状态。
    """
    satellite_id: str
    plan: ManeuverPlan
    status: TaskStatus = TaskStatus.PENDING
    # 将待处理的机动列表单独管理，执行一个就移除一个
    pending_maneuvers: List[ManeuverExecution] = field(default_factory=list)

    def __post_init__(self):
        # 创建时，自动按执行时间对所有待处理机动进行排序
        self.pending_maneuvers = sorted(
            self.plan.maneuvers, key=lambda m: m.execution_time
        )

class TaskManager:
    """
    任务管理器。
    
    这是一个有状态的类，负责管理仿真中所有卫星的所有活动任务。
    它在每个仿真步骤中被调用，以检查并执行到期的轨道机动。
    """

    def __init__(self):
        """初始化一个空的 TaskManager。"""
        # 使用字典来存储任务，键为卫星ID，值为该卫星当前的 ActiveTask
        # 核心设计：一个卫星在同一时间只能执行一个任务
        self.active_tasks: Dict[str, ActiveTask] = {}

    def add_task(self, satellite_id: str, plan: ManeuverPlan, force: bool = False) -> bool:
        """
        为一个卫星分配一个新任务，可选择强制覆盖当前任务。

        Args:
            satellite_id (str): 将要执行任务的卫星的ID。
            plan (ManeuverPlan): 由 maneuver_planner 生成的机动计划。
            force (bool, optional): 如果为 True，且卫星当前有任务，
                                    则会先取消当前任务，再添加新任务。默认为 False。

        Returns:
            bool: 如果成功添加任务则返回 True。
        """
        if self.is_busy(satellite_id):
            if force:
                print(f"警告: 卫星 '{satellite_id}' 已有任务，执行强制替换。")
                self.cancel_task(satellite_id)
            else:
                print(f"警告: 卫星 '{satellite_id}' 已有任务，无法添加新任务 (使用 force=True 可强制替换)。")
                return False
        
        # ... (后续添加任务的逻辑不变) ...
        if not plan or not plan.maneuvers:
            print(f"警告: 尝试为卫星 '{satellite_id}' 添加一个空的机动计划。")
            return False

        task = ActiveTask(satellite_id=satellite_id, plan=plan)
        self.active_tasks[satellite_id] = task
        print(f"信息: 为卫星 '{satellite_id}' 添加新任务，总 Δv: {plan.total_delta_v:.4f}。")
        return True

    def is_busy(self, satellite_id: str) -> bool:
        """检查一个卫星当前是否有正在执行的任务。"""
        return satellite_id in self.active_tasks

    def get_task_status(self, satellite_id: str) -> TaskStatus | None:
        """获取卫星当前任务的状态。"""
        if task := self.active_tasks.get(satellite_id):
            return task.status
        return None
    def cancel_task(self, satellite_id: str) -> bool:
        """
        显式地中断并取消一个卫星当前正在执行的任务。

        Args:
            satellite_id (str): 要中断任务的卫星的ID。

        Returns:
            bool: 如果成功取消了任务，返回 True；如果该卫星原本就没有任务，返回 False。
        """
        if self.is_busy(satellite_id):
            task = self.active_tasks.pop(satellite_id) # 从活动任务字典中移除
            task.status = TaskStatus.CANCELLED
            print(f"信息: 中断并取消卫星 '{satellite_id}' 的当前任务。")
            return True
        return False
    def update(self, satellites: Dict[str, Satellite], current_time: Time, dt: TimeDelta):
        """
        更新所有活动任务的状态，并在需要时执行机动。
        这是 TaskManager 的核心方法，应在仿真主循环的每个时间步被调用。

        Args:
            satellites (Dict[str, Satellite]): 仿真世界中所有卫星对象的字典。
            current_time (Time): 当前仿真时间。
            dt (TimeDelta): 当前仿真步长。
        """
        # 创建一个待移除任务的列表，避免在迭代过程中修改字典
        completed_or_failed_tasks: List[str] = []

        # 遍历当前所有活动任务
        for sat_id, task in self.active_tasks.items():
            if not task.pending_maneuvers:
                # 如果没有待处理的机动，说明任务已完成
                task.status = TaskStatus.COMPLETED
                completed_or_failed_tasks.append(sat_id)
                print(f"信息: 卫星 '{sat_id}' 的任务已完成。")
                continue

            # 获取下一个将要执行的机动
            next_maneuver = task.pending_maneuvers[0]
            
            # 检查下一个机动的执行时间是否落在当前时间步之内
            if current_time <= next_maneuver.execution_time < current_time + dt:
                
                satellite = satellites.get(sat_id)
                if not satellite:
                    print(f"错误: 找不到ID为 '{sat_id}' 的卫星来执行任务。")
                    task.status = TaskStatus.FAILED
                    completed_or_failed_tasks.append(sat_id)
                    continue

                print(f"执行: 卫星 '{sat_id}' 在 {current_time} 执行机动。")
                
                # --- 核心执行逻辑 ---
                # # 1. 计算所需速度增量的大小
                # delta_v_vec = next_maneuver.delta_v
                # delta_v_mag = (sum(dv**2 for dv in delta_v_vec))**0.5
                # 1. 计算从当前时刻到精确点火时刻的时间差
                time_to_burn = next_maneuver.execution_time - current_time
                # 2. 将卫星精确传播到点火时刻
                orbit_at_burn_time = satellite.orbit.propagate(time_to_burn)
                # 3. 在精确的时刻和状态上，施加脉冲
                delta_v_vec = next_maneuver.delta_v
                new_velocity = orbit_at_burn_time.v + delta_v_vec
                
                # 4. 用点火时刻的真实状态，创建新的轨道
                new_orbit = Orbit.from_vectors(
                    attractor=orbit_at_burn_time.attractor,
                    r=orbit_at_burn_time.r,  # 使用点火时刻的位置
                    v=new_velocity,          # 使用点火后的速度
                    epoch=orbit_at_burn_time.epoch # 历元也来自点火时刻
                )
                satellite.update_orbit(new_orbit)
                # 5. 检查燃料并消耗 (这部分逻辑不变，但我们顺便修复一个次要bug) 
                # 修正：使用np.linalg.norm正确计算带单位的矢量大小
                delta_v_mag = np.linalg.norm(delta_v_vec)
                if not satellite.can_maneuver or satellite.fuel_mass < satellite.consume_fuel(delta_v_mag):
                    print(f"失败: 卫星 '{sat_id}' 燃料耗尽，无法执行机动。")
                    task.status = TaskStatus.FAILED
                    completed_or_failed_tasks.append(sat_id)
                    continue
                
                fuel_consumed = satellite.consume_fuel(delta_v_mag)
                print(f"  - 消耗燃料: {fuel_consumed:.4f}")
                # 注意：一个更复杂的模型可以在燃料不足时执行部分机动

                
                # 4. 从任务中移除已执行的机动
                task.pending_maneuvers.pop(0)
                
                # --- 新增的修正逻辑 ---
                # 5. 立刻检查任务是否已经完成
                if not task.pending_maneuvers:
                    # 如果待办列表已空，说明这是最后一次机动
                    task.status = TaskStatus.COMPLETED
                    # 立刻将其加入待清理列表
                    completed_or_failed_tasks.append(sat_id)
                else:
                    # 如果列表非空，说明还有后续机动
                    task.status = TaskStatus.IN_PROGRESS

        # 清理已完成或失败的任务
        for sat_id in completed_or_failed_tasks:
            del self.active_tasks[sat_id]