import time
from env import SpaceCombatEnv  # 请确保你的类保存在 space_combat_env.py 中

# 创建环境
env = SpaceCombatEnv(config_path="config/satellites.yaml", dt=10, log_dir="logs/test_run")

# 重置环境
obs = env.reset()
print("Initial Observation Shape:", obs.shape)

# 运行 100 步模拟
for step in range(100):
    actions = env.rule_based_action()
    obs, reward, done, info = env.step(actions)
    print(f"Step {step:03d} | Reward: {reward:.2f}")

# 可视化轨迹
env.render_2d()

# 清理资源
env.close()
