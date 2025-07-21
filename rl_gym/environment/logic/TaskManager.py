# environment/logic/task_manager.py

"""
本模块定义了 TaskManager 类，它是 logic 层的核心状态管理器。

其职责是接收由 maneuver_planner 生成的 ManeuverPlan，将其转化为
一个有状态的、可追踪的 ActiveTask，并在仿真时间流中精确地执行
任务中的每一次机动。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Callable
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

    def _validate_plan(self, plan: ManeuverPlan) -> bool:
        if not plan or not plan.maneuvers:
            return False
        
        # 检查执行时间顺序
        for i in range(1, len(plan.maneuvers)):
            if plan.maneuvers[i].execution_time <= plan.maneuvers[i-1].execution_time:
                return False
        
        # 检查delta_v有效性
        for maneuver in plan.maneuvers:
            if np.any(np.isnan(maneuver.delta_v.value)):
                return False
        
        return True
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
        if not self._validate_plan(plan):
            return False
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
# TaskManager.py (最终的、决定性的修正)

    def update(self, satellites, current_time, dt, propagator_func=None):
        """
        改进的update方法，使用更精确的时间处理
        """
        propagate = propagator_func if propagator_func is not None else (lambda orbit, time: orbit.propagate(time))
        completed_or_failed_tasks = []
        handled_satellite_ids = set()
        
        for sat_id, task in self.active_tasks.copy().items():
            if not task.pending_maneuvers:
                task.status = TaskStatus.COMPLETED
                completed_or_failed_tasks.append(sat_id)
                continue

            next_maneuver = task.pending_maneuvers[0]
            
            # 改进：使用更精确的时间窗口判断
            step_start = current_time
            step_end = current_time + dt
            burn_time = next_maneuver.execution_time
            
            # 只有当机动时间在当前步长内时才执行
            if step_start <= burn_time < step_end:
                satellite = satellites.get(sat_id)
                if not satellite:
                    task.status = TaskStatus.FAILED
                    completed_or_failed_tasks.append(sat_id)
                    continue

                # 燃料检查
                delta_v_vec = next_maneuver.delta_v
                delta_v_mag = np.linalg.norm(delta_v_vec)
                fuel_needed = satellite.fuel_mass_needed(delta_v_mag)

                if not satellite.can_maneuver or satellite.fuel_mass < fuel_needed:
                    task.status = TaskStatus.FAILED
                    completed_or_failed_tasks.append(sat_id)
                    continue

                # 改进：更精确的轨道传播
                # 1. 传播到精确的点火时刻
                time_to_burn = burn_time - step_start
                orbit_at_burn = propagate(satellite.orbit, time_to_burn)
                
                # 2. 施加机动
                new_velocity = orbit_at_burn.v + delta_v_vec
                orbit_after_burn = Orbit.from_vectors(
                    attractor=orbit_at_burn.attractor,
                    r=orbit_at_burn.r,
                    v=new_velocity,
                    epoch=burn_time  # 使用精确的点火时间
                )
                
                # 3. 传播到步长结束
                remaining_time = step_end - burn_time
                if remaining_time.to_value('s') > 0:
                    final_orbit = propagate(orbit_after_burn, remaining_time)
                else:
                    final_orbit = orbit_after_burn
                
                # 4. 更新卫星状态
                satellite.update_orbit(final_orbit)
                satellite.consume_fuel(delta_v_mag)
                handled_satellite_ids.add(sat_id)
                
                # 5. 更新任务状态
                task.pending_maneuvers.pop(0)
                if not task.pending_maneuvers:
                    task.status = TaskStatus.COMPLETED
                    completed_or_failed_tasks.append(sat_id)
                else:
                    task.status = TaskStatus.IN_PROGRESS

        # 清理完成的任务
        for sat_id in completed_or_failed_tasks:
            if sat_id in self.active_tasks:
                del self.active_tasks[sat_id]
                
        return handled_satellite_ids