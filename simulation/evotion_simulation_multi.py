# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from matplotlib.animation import FuncAnimation
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib
# import time
# import threading
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# WINDOW_X_POS = 750       # 窗口左上角 x 坐标
# WINDOW_Y_POS = 50       # 窗口左上角 y 坐标
# WINDOW_WIDTH = 1800     # 窗口宽度
# WINDOW_HEIGHT = 960     # 窗口高度

# def set_window_geometry(fig, x, y, width, height):
#     """设置Matplotlib窗口的位置和大小，兼容常用后端"""
#     try:
#         manager = fig.canvas.manager
#         backend = matplotlib.get_backend()
#         print(f"当前 Matplotlib 后端: {backend}")

#         if backend == 'tkagg':
#             manager.window.geometry(f"{width}x{height}+{x}+{y}")
#         elif backend == 'Qt5Agg' or backend == 'QtAgg':
#             manager.window.setGeometry(x, y, width, height)
#         elif backend == 'WXAgg':
#             manager.window.SetPosition((x, y))
#             manager.window.SetSize((width, height))
#         else:
#             print(f"警告: 未对后端 '{backend}' 进行窗口位置的特殊处理。")
#             manager.resize(width, height)
#     except Exception as e:
#         print(f"警告: 设置窗口位置时出错 - {e}")



# # --- 0. 初始打印 ---
# print("太空多智能体博弈对抗 - 决策演进可视化演示")
# print(f"Matplotlib version: {matplotlib.__version__}")

# # --- 1. 全局配置与参数 ---
# PRE_SIM_DURATION = 160; PRE_EVAL_DURATION = 80; POST_SIM_DURATION = 160; POST_EVAL_DURATION = 120
# PRE_SIM_END_FRAME = PRE_SIM_DURATION; PRE_EVAL_END_FRAME = PRE_SIM_END_FRAME + PRE_EVAL_DURATION; POST_SIM_END_FRAME = PRE_EVAL_END_FRAME + POST_SIM_DURATION; TOTAL_FRAMES = POST_SIM_END_FRAME + POST_EVAL_DURATION
# PRE_EVO_LOSS_FRAME_LOCAL = 80; PRE_EVO_KILL_FRAME_LOCAL = 110; POST_EVO_KILL_FRAME_LOCAL = 100
# EVAL_DELAY = 30; ZONE_LIMIT=10; STAR_COUNT=200; COLOR_FRIENDLY='#e74c3c'; COLOR_ENEMY='#3498db'; COLOR_DESTROYED='#95a5a6'; COLOR_EVOLVED='#f1c40f'; COLOR_ATTACK='#e67e22'

# # --- 2. 辅助函数定义 ---
# def generate_trajectories():
#     trajectories = {};start_A1=np.array([-8,-8,2]);start_A2=np.array([-8,8,-2]);start_A3=np.array([8,-8,-2]);start_E=np.array([0,0,0]);intercept_A1=np.array([-1,-1,0]);support_A2=np.array([-3,4,-1]);support_A3=np.array([4,-3,-1]);trajectories['pre_A1']=np.array([np.linspace(s,e,PRE_SIM_DURATION)for s,e in zip(start_A1,intercept_A1)]);trajectories['pre_A2']=np.array([np.linspace(s,e,PRE_SIM_DURATION)for s,e in zip(start_A2,support_A2)]);trajectories['pre_A3']=np.array([np.linspace(s,e,PRE_SIM_DURATION)for s,e in zip(start_A3,support_A3)]);trajectories['pre_E']=np.array([np.linspace(s,e,PRE_SIM_DURATION)for s,e in zip(start_E,start_E)]);end_post_A1=np.array([-5,-5,1]);end_post_A2=np.array([-5,5,-1]);end_post_A3=np.array([5,-5,-1]);trajectories['post_A1']=np.array([np.linspace(s,e,POST_SIM_DURATION)for s,e in zip(start_A1,end_post_A1)]);trajectories['post_A2']=np.array([np.linspace(s,e,POST_SIM_DURATION)for s,e in zip(start_A2,end_post_A2)]);trajectories['post_A3']=np.array([np.linspace(s,e,POST_SIM_DURATION)for s,e in zip(start_A3,end_post_A3)]);trajectories['post_E']=np.array([np.linspace(s,e,POST_SIM_DURATION)for s,e in zip(start_E,start_E)]);return trajectories

# def calculate_strategy_vector(loss_count, kill_frame_local, phase_duration, min_dist, coordination_score):
#     survivability=(3-loss_count)/3;efficiency=1.0-(kill_frame_local/phase_duration);aggressiveness=1.0-(min_dist/ZONE_LIMIT);return np.array([aggressiveness,coordination_score,survivability,efficiency])

# def print_evaluation_details(sim_type, strategy_vector, final_score, final_score_text):
#     aggressiveness,coordination_score,survivability,efficiency=strategy_vector;print(f"\n>>> 开始进行 [{sim_type}] 作战评估...");time.sleep(0.5);print("\n[1/3] 正在解析单步任务完成度...");time.sleep(0.5)
#     if sim_type=="演进前":print("  - 卫星-1: 最终拦截机动 -> 失败 (代价过高)")
#     else:print("  - 卫星-1/2/3: 防区外打击阵位机动 -> 成功")
#     time.sleep(0.5);print("\n[2/3] 正在汇总关键节点完成度...");time.sleep(0.5)
#     if sim_type=="演进前":print(f"  - 关键节点 (生存率): {survivability*100:.1f}%")
#     else:print(f"  - 关键节点 (生存率): {survivability*100:.1f}%")
#     time.sleep(0.5);print("\n[3/3] 正在生成整体任务完成度评估...");time.sleep(0.5)
#     print("="*50);print(f" {sim_type} 作战评估报告");print("="*50)
#     print(" 指标计算公式 (战术诊断):")
#     print("   - 生存率 = (3 - 损失数) / 3")
#     print(f"   - 效率   = 1.0 - (击毁用时 / 阶段总时长)")
#     # [修正] 增加协同性计算说明
#     print("   - 协同性 = 策略类型赋予的固定分值 (拦截=0.3, 协同=0.9)")
#     print("   - 攻击性 = 1.0 - (最近距离 / 作战区域尺寸)")
#     print("-"*50)
#     print(" 本次作战数据:")
#     print(f"   - 生存率: {survivability:.2f}");print(f"   - 效率:   {efficiency:.2f}");print(f"   - 协同性: {coordination_score:.2f}");print(f"   - 攻击性: {aggressiveness:.2f}");
#     print("="*50);print(" 整体任务评估 (战略结论):");print("   该分数是基于最终任务结果的直接评定。");print("\n   [最终得分判定规则]");print(f"   - 本次得分: {final_score}% ({final_score_text})");print("   - 判定依据: 根据是否有己方损失进行等级划分。");
    
#     if sim_type=="演进前":
#         print(f"  - 综合评估: 任务达成但存在缺陷。\n>>> 评估完成。最终得分: {final_score}%");print("\n>>> 开始进行记忆重构...");time.sleep(0.5);print("  - 解析评估数据链... 识别到‘生存率’为主要短板。");time.sleep(0.5);print("  - 触发记忆拓扑重构: 增加‘安全距离’权重...");time.sleep(0.5);print(">>> 记忆向量库已更新。新策略已生成。")
#     else:print(f"  - 综合评估: 高效达成所有任务目标。\n>>> 评估完成。最终得分: {final_score}%");print("\n>>> 新策略被验证为高效，强化相关记忆向量权重。")

# # --- 3. 全局数据初始化 ---
# trajectories = generate_trajectories()
# pre_evo_strategy = calculate_strategy_vector(1, PRE_EVO_KILL_FRAME_LOCAL, PRE_SIM_DURATION, 1.0, 0.3)
# post_evo_strategy = calculate_strategy_vector(0, POST_EVO_KILL_FRAME_LOCAL, POST_SIM_DURATION, 5.0, 0.9)
# agent1_destroyed = False




# # --- 4. 可视化设置 ---
# strategy_labels=['攻击性\nAggressiveness','协同性\nCoordination','生存率\nSurvivability','效率\nEfficiency'];
# fig=plt.figure(figsize=(18,9));fig.patch.set_facecolor('#1a1a1a');
# set_window_geometry(fig, WINDOW_X_POS, WINDOW_Y_POS, WINDOW_WIDTH, WINDOW_HEIGHT)
# gs=gridspec.GridSpec(2,2,figure=fig,hspace=0.3);ax_3d=fig.add_subplot(gs[:,0],projection='3d');ax_3d.set_facecolor('black');ax_3d.set_xlim([-ZONE_LIMIT,ZONE_LIMIT]);ax_3d.set_ylim([-ZONE_LIMIT,ZONE_LIMIT]);ax_3d.set_zlim([-ZONE_LIMIT,ZONE_LIMIT]);ax_3d.set_xlabel('沿轨方向 (In-Track)',color='white');ax_3d.set_ylabel('横向 (Cross-Track)',color='white');ax_3d.set_zlabel('径向 (Radial)',color='white');ax_3d.tick_params(axis='x',colors='white');ax_3d.tick_params(axis='y',colors='white');ax_3d.tick_params(axis='z',colors='white');stars=np.random.rand(STAR_COUNT,3)*2*ZONE_LIMIT-ZONE_LIMIT;ax_3d.scatter(stars[:,0],stars[:,1],stars[:,2],s=1,c='white',alpha=0.5);grid_x,grid_z=np.meshgrid(np.linspace(-ZONE_LIMIT,ZONE_LIMIT,10),np.linspace(-ZONE_LIMIT,ZONE_LIMIT,10));grid_y=-0.05*(grid_x**2+grid_z**2)-ZONE_LIMIT;ax_3d.plot_wireframe(grid_x,grid_y,grid_z,color='gray',alpha=0.2);sats_f1=ax_3d.scatter([],[],[],s=180,c=COLOR_FRIENDLY,label='我方卫星');sats_f2=ax_3d.scatter([],[],[],s=180,c=COLOR_FRIENDLY);sats_f3=ax_3d.scatter([],[],[],s=180,c=COLOR_FRIENDLY);sats_e=ax_3d.scatter([],[],[],s=180,marker='o',c=COLOR_ENEMY,label='敌方卫星');attack_lines=[ax_3d.plot([],[],[],lw=2,c=COLOR_ATTACK,alpha=0.8)[0] for _ in range(3)];ax_3d.legend();ax_eval=fig.add_subplot(gs[0,1]);ax_eval.set_facecolor('#2b2b2b');ax_eval.set_xlim([0,1]);ax_eval.set_ylim([0,1]);ax_eval.axis('off');eval_title=ax_eval.text(0.5,0.9,'自主评估反思模块',ha='center',va='center',fontsize=20,color='white',weight='bold');status_texts={'A1':ax_eval.text(0.1,0.7,'我方 卫星-1: ...',fontsize=14,color='white'),'A2':ax_eval.text(0.1,0.6,'我方 卫星-2: ...',fontsize=14,color='white'),'A3':ax_eval.text(0.1,0.5,'我方 卫星-3: ...',fontsize=14,color='white'),'E':ax_eval.text(0.1,0.4,'敌方目标: ...',fontsize=14,color='white'),'loss':ax_eval.text(0.5,0.25,'我方损失: 0',ha='center',fontsize=16,color='white'),'score':ax_eval.text(0.5,0.1,'整体任务评估: ...',ha='center',fontsize=18,color=COLOR_EVOLVED,weight='bold')};num_vars=len(strategy_labels);angles=np.linspace(0,2*np.pi,num_vars,endpoint=False).tolist();angles_closed=angles+angles[:1];ax_evo=fig.add_subplot(gs[1,1],polar=True);ax_evo.set_facecolor('#2b2b2b');ax_evo.text(0.5,1.3,'记忆重构演进模块',transform=ax_evo.transAxes,ha='center',va='center',fontsize=20,color='white',weight='bold');ax_evo.set_yticklabels([]);ax_evo.set_thetagrids(np.degrees(angles),strategy_labels,color='white',fontsize=12);ax_evo.set_rlim(0,1.0);evo_line_pre,=ax_evo.plot([],[],color=COLOR_ENEMY,linewidth=2,linestyle='dashed',label='演进前策略');evo_line_morph,=ax_evo.plot([],[],color=COLOR_EVOLVED,linewidth=2.5,label='演进后策略');evo_fill_morph=ax_evo.fill([],[],color=COLOR_EVOLVED,alpha=0.4);ax_evo.legend(loc='upper right',bbox_to_anchor=(1.3,1.1),labelcolor='white',frameon=False);evo_text=ax_evo.text(np.pi/2,1.33,"等待首次评估...",ha='center',va='center',fontsize=16,color='gray',transform=ax_evo.transData)


# # --- 5. 动画更新函数 ---
# def update(frame):
#     global agent1_destroyed
#     # [核心修正] 动画第0帧的“主复位”逻辑
#     if frame == 0:
#         agent1_destroyed = False
#         # 重置所有卫星的视觉状态
#         sats_f1.set_visible(True); sats_f1.set_color(COLOR_FRIENDLY)
#         sats_f2.set_visible(True)
#         sats_f3.set_visible(True)
#         sats_e.set_visible(True); sats_e.set_alpha(1)
#         # 重置所有文本
#         status_texts['A1'].set_text('我方 卫星-1: ...');status_texts['A1'].set_color('white')
#         status_texts['A2'].set_text('我方 卫星-2: ...');status_texts['A2'].set_color('white')
#         status_texts['A3'].set_text('我方 卫星-3: ...');status_texts['A3'].set_color('white')
#         status_texts['loss'].set_text('我方损失: 0')
#         status_texts['E'].set_text('敌方目标: 存活');status_texts['E'].set_color('white')
#         status_texts['score'].set_text('整体任务评估: ...')
#         # 重置雷达图和攻击线
#         evo_line_pre.set_data([],[]);evo_line_morph.set_data([],[])
#         evo_fill_morph[0].set_xy(np.array([[],[]]).T);evo_text.set_text("等待首次评估...")
#         for line in attack_lines:line.set_data_3d([],[],[])

#     # 阶段一: 演进前作战重现
#     if frame < PRE_SIM_END_FRAME:
#         f = frame
#         status_texts['score'].set_text('战场情景重现中...')
#         if not agent1_destroyed: sats_f1._offsets3d=([trajectories['pre_A1'][0,f]],[trajectories['pre_A1'][1,f]],[trajectories['pre_A1'][2,f]])
#         sats_f2._offsets3d=([trajectories['pre_A2'][0,f]],[trajectories['pre_A2'][1,f]],[trajectories['pre_A2'][2,f]]);sats_f3._offsets3d=([trajectories['pre_A3'][0,f]],[trajectories['pre_A3'][1,f]],[trajectories['pre_A3'][2,f]]);sats_e._offsets3d=([trajectories['pre_E'][0,f]],[trajectories['pre_E'][1,f]],[trajectories['pre_E'][2,f]]);
#         if not agent1_destroyed:status_texts['A1'].set_text('我方 卫星-1: 快速拦截中...')
#         status_texts['A2'].set_text('我方 卫星-2: 支援机动中...');status_texts['A3'].set_text('我方 卫星-3: 支援机动中...');
#         if f == PRE_EVO_LOSS_FRAME_LOCAL:
#             p_enemy = trajectories['pre_E'][:,f]
#             p_friendly1 = trajectories['pre_A1'][:,f]
#             attack_lines[0].set_data_3d([p_enemy[0], p_friendly1[0]], [p_enemy[1], p_friendly1[1]], [p_enemy[2], p_friendly1[2]])

#         if f >= PRE_EVO_LOSS_FRAME_LOCAL and not agent1_destroyed:
#             agent1_destroyed=True;sats_f1.set_color(COLOR_DESTROYED);status_texts['A1'].set_text('我方 卫星-1: 已被击毁');status_texts['A1'].set_color(COLOR_DESTROYED);status_texts['loss'].set_text('我方损失: 1')
#         if f > PRE_EVO_LOSS_FRAME_LOCAL + 10:
#             attack_lines[0].set_data_3d([],[],[])
#         if f >= PRE_EVO_KILL_FRAME_LOCAL:
#             if f==PRE_EVO_KILL_FRAME_LOCAL: p2=trajectories['pre_A2'][:,f];p3=trajectories['pre_A3'][:,f];pe=trajectories['pre_E'][:,f];attack_lines[1].set_data_3d([p2[0],pe[0]],[p2[1],pe[1]],[p2[2],pe[2]]);attack_lines[2].set_data_3d([p3[0],pe[0]],[p3[1],pe[1]],[p3[2],pe[2]])
#             sats_e.set_visible(False);status_texts['E'].set_text('敌方目标: 已击毁');status_texts['E'].set_color(COLOR_FRIENDLY);
#         if f > PRE_EVO_KILL_FRAME_LOCAL+10:
#              for line in attack_lines:line.set_data_3d([],[],[])
    
#     # 阶段二: 首次评估
#     elif frame < PRE_EVAL_END_FRAME:
#         f = frame - PRE_SIM_END_FRAME
#         if f < EVAL_DELAY:
#             if f == 1: threading.Thread(target=print_evaluation_details, args=("演进前",pre_evo_strategy,65,"有待提高")).start()
#             status_texts['score'].set_text('整体任务评估中...')
#         else:
#             status_texts['score'].set_text('整体任务评估: 65% (有待提高)')
#         if f == 1: values=np.concatenate((pre_evo_strategy,[pre_evo_strategy[0]]));evo_line_pre.set_data(angles_closed,values);evo_line_pre.set_alpha(0.7);evo_text.set_text("演进前策略已评估")
#         elif f > EVAL_DELAY+10: evo_text.set_text("记忆向量库更新中...")
    
#     # 阶段三: 演进后作战重现
#     elif frame < POST_SIM_END_FRAME:
#         f = frame - PRE_EVAL_END_FRAME
#         if f == 1: 
#             sats_f1.set_visible(True);sats_f1.set_color(COLOR_FRIENDLY);sats_e.set_visible(True);
#             status_texts['A1'].set_text('我方 卫星-1: ...');status_texts['A1'].set_color('white');status_texts['loss'].set_text('我方损失: 0');status_texts['E'].set_text('敌方目标: 存活');status_texts['E'].set_color('white');
#             evo_text.set_text("应用新策略重现中...")
#             for line in attack_lines:line.set_data_3d([],[],[])
#         status_texts['score'].set_text('战场情景重现中...')
#         sats_f1._offsets3d=([trajectories['post_A1'][0,f]],[trajectories['post_A1'][1,f]],[trajectories['post_A1'][2,f]]);sats_f2._offsets3d=([trajectories['post_A2'][0,f]],[trajectories['post_A2'][1,f]],[trajectories['post_A2'][2,f]]);sats_f3._offsets3d=([trajectories['post_A3'][0,f]],[trajectories['post_A3'][1,f]],[trajectories['post_A3'][2,f]]);sats_e._offsets3d=([trajectories['post_E'][0,f]],[trajectories['post_E'][1,f]],[trajectories['post_E'][2,f]]);
#         status_texts['A1'].set_text('我方 卫星-1: 安全协同机动...');status_texts['A2'].set_text('我方 卫星-2: 安全协同机动...');status_texts['A3'].set_text('我方 卫星-3: 安全协同机动...');
#         if f >= POST_EVO_KILL_FRAME_LOCAL:
#             if f==POST_EVO_KILL_FRAME_LOCAL: p1=trajectories['post_A1'][:,f];p2=trajectories['post_A2'][:,f];p3=trajectories['post_A3'][:,f];pe=trajectories['post_E'][:,f];attack_lines[0].set_data_3d([p1[0],pe[0]],[p1[1],pe[1]],[p1[2],pe[2]]);attack_lines[1].set_data_3d([p2[0],pe[0]],[p2[1],pe[1]],[p2[2],pe[2]]);attack_lines[2].set_data_3d([p3[0],pe[0]],[p3[1],pe[1]],[p3[2],pe[2]])
#             sats_e.set_visible(False);status_texts['E'].set_text('敌方目标: 已击毁');status_texts['E'].set_color(COLOR_FRIENDLY);
#         if f > POST_EVO_KILL_FRAME_LOCAL+10:
#              for line in attack_lines:line.set_data_3d([],[],[])

#     # 阶段四: 最终评估与演进分析
#     else:
#         f = frame - POST_SIM_END_FRAME
#         if f < EVAL_DELAY:
#             if f == 1: threading.Thread(target=print_evaluation_details, args=("演进后",post_evo_strategy,95,"优秀")).start()
#             status_texts['score'].set_text('整体任务评估中...')
#         else:
#             status_texts['score'].set_text('整体任务评估: 95% (优秀)')
#         if f == 1: evo_text.set_text("策略演进分析中...")
#         progress = f/POST_EVAL_DURATION; progress = min(progress, 1.0)
#         current_strategy = pre_evo_strategy + (post_evo_strategy - pre_evo_strategy) * progress
#         current_values = np.concatenate((current_strategy, [current_strategy[0]]));evo_line_morph.set_data(angles_closed, current_values);evo_fill_morph[0].set_xy(np.array([angles_closed, current_values]).T)

#     return [sats_f1,sats_f2,sats_f3,sats_e,evo_line_pre,evo_line_morph]+list(status_texts.values())

# # --- 6. 运行与保存 ---
# ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=40, blit=False, repeat=True)
# plt.tight_layout(pad=3.0)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import time
import threading
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

WINDOW_X_POS = 750       # 窗口左上角 x 坐标
WINDOW_Y_POS = 50       # 窗口左上角 y 坐标
WINDOW_WIDTH = 1800     # 窗口宽度
WINDOW_HEIGHT = 960     # 窗口高度

def set_window_geometry(fig, x, y, width, height):
    """设置Matplotlib窗口的位置和大小，兼容常用后端"""
    try:
        manager = fig.canvas.manager
        backend = matplotlib.get_backend()
        print(f"当前 Matplotlib 后端: {backend}")

        if backend == 'tkagg':
            manager.window.geometry(f"{width}x{height}+{x}+{y}")
        elif backend == 'Qt5Agg' or backend == 'QtAgg':
            manager.window.setGeometry(x, y, width, height)
        elif backend == 'WXAgg':
            manager.window.SetPosition((x, y))
            manager.window.SetSize((width, height))
        else:
            print(f"警告: 未对后端 '{backend}' 进行窗口位置的特殊处理。")
            manager.resize(width, height)
    except Exception as e:
        print(f"警告: 设置窗口位置时出错 - {e}")



# --- 0. 初始打印 ---
print("太空多智能体博弈对抗 - 决策演进可视化演示")
print(f"Matplotlib version: {matplotlib.__version__}")

# --- 1. 全局配置与参数 ---
PRE_SIM_DURATION = 160; PRE_EVAL_DURATION = 80; POST_SIM_DURATION = 160; POST_EVAL_DURATION = 120
PRE_SIM_END_FRAME = PRE_SIM_DURATION; PRE_EVAL_END_FRAME = PRE_SIM_END_FRAME + PRE_EVAL_DURATION; POST_SIM_END_FRAME = PRE_EVAL_END_FRAME + POST_SIM_DURATION; TOTAL_FRAMES = POST_SIM_END_FRAME + POST_EVAL_DURATION
PRE_EVO_LOSS_FRAME_LOCAL = 80; PRE_EVO_KILL_FRAME_LOCAL = 110; POST_EVO_KILL_FRAME_LOCAL = 100
EVAL_DELAY = 30; ZONE_LIMIT=10; STAR_COUNT=200; COLOR_FRIENDLY='#e74c3c'; COLOR_ENEMY='#3498db'; COLOR_DESTROYED='#95a5a6'; COLOR_EVOLVED='#f1c40f'; COLOR_ATTACK='#e67e22'

# --- 2. 辅助函数定义 ---
def generate_trajectories():
    trajectories = {};start_A1=np.array([-8,-8,2]);start_A2=np.array([-8,8,-2]);start_A3=np.array([8,-8,-2]);start_E=np.array([0,0,0]);intercept_A1=np.array([-1,-1,0]);support_A2=np.array([-3,4,-1]);support_A3=np.array([4,-3,-1]);trajectories['pre_A1']=np.array([np.linspace(s,e,PRE_SIM_DURATION)for s,e in zip(start_A1,intercept_A1)]);trajectories['pre_A2']=np.array([np.linspace(s,e,PRE_SIM_DURATION)for s,e in zip(start_A2,support_A2)]);trajectories['pre_A3']=np.array([np.linspace(s,e,PRE_SIM_DURATION)for s,e in zip(start_A3,support_A3)]);trajectories['pre_E']=np.array([np.linspace(s,e,PRE_SIM_DURATION)for s,e in zip(start_E,start_E)]);end_post_A1=np.array([-5,-5,1]);end_post_A2=np.array([-5,5,-1]);end_post_A3=np.array([5,-5,-1]);trajectories['post_A1']=np.array([np.linspace(s,e,POST_SIM_DURATION)for s,e in zip(start_A1,end_post_A1)]);trajectories['post_A2']=np.array([np.linspace(s,e,POST_SIM_DURATION)for s,e in zip(start_A2,end_post_A2)]);trajectories['post_A3']=np.array([np.linspace(s,e,POST_SIM_DURATION)for s,e in zip(start_A3,end_post_A3)]);trajectories['post_E']=np.array([np.linspace(s,e,POST_SIM_DURATION)for s,e in zip(start_E,start_E)]);return trajectories

def calculate_strategy_vector(loss_count, kill_frame_local, phase_duration, min_dist, coordination_score):
    survivability=(3-loss_count)/3;efficiency=1.0-(kill_frame_local/phase_duration);aggressiveness=1.0-(min_dist/ZONE_LIMIT);return np.array([aggressiveness,coordination_score,survivability,efficiency])

def print_evaluation_details(sim_type, strategy_vector, final_score, final_score_text):
    aggressiveness,coordination_score,survivability,efficiency=strategy_vector;print(f"\n>>> 开始进行 [{sim_type}] 作战评估...");time.sleep(0.5);print("\n[1/3] 正在解析单步任务完成度...");time.sleep(0.5)
    if sim_type=="演进前":print("  - 卫星-1: 最终拦截机动 -> 失败 (代价过高)")
    else:print("  - 卫星-1/2/3: 防区外打击阵位机动 -> 成功")
    time.sleep(0.5);print("\n[2/3] 正在汇总关键节点完成度...");time.sleep(0.5)
    if sim_type=="演进前":print(f"  - 关键节点 (生存率): {survivability*100:.1f}%")
    else:print(f"  - 关键节点 (生存率): {survivability*100:.1f}%")
    time.sleep(0.5);print("\n[3/3] 正在生成整体任务完成度评估...");time.sleep(0.5)
    print("="*50);print(f" {sim_type} 作战评估报告");print("="*50)
    print(" 指标计算公式 (战术诊断):")
    print("   - 生存率 = (3 - 损失数) / 3")
    print(f"   - 效率   = 1.0 - (击毁用时 / 阶段总时长)")
    # [修正] 增加协同性计算说明
    print("   - 协同性 = 策略类型赋予的固定分值 (拦截=0.3, 协同=0.9)")
    print("   - 攻击性 = 1.0 - (最近距离 / 作战区域尺寸)")
    print("-"*50)
    print(" 本次作战数据:")
    print(f"   - 生存率: {survivability:.2f}");print(f"   - 效率:   {efficiency:.2f}");print(f"   - 协同性: {coordination_score:.2f}");print(f"   - 攻击性: {aggressiveness:.2f}");
    print("="*50);print(" 整体任务评估 (战略结论):");print("   该分数是基于最终任务结果的直接评定。");print("\n   [最终得分判定规则]");print(f"   - 本次得分: {final_score}% ({final_score_text})");print("   - 判定依据: 根据是否有己方损失进行等级划分。");
    
    if sim_type=="演进前":
        print(f"  - 综合评估: 任务达成但存在缺陷。\n>>> 评估完成。最终得分: {final_score}%");print("\n>>> 开始进行记忆重构...");time.sleep(0.5);print("  - 解析评估数据链... 识别到‘生存率’为主要短板。");time.sleep(0.5);print("  - 触发记忆拓扑重构: 增加‘安全距离’权重...");time.sleep(0.5);print(">>> 记忆向量库已更新。新策略已生成。")
    else:print(f"  - 综合评估: 高效达成所有任务目标。\n>>> 评估完成。最终得分: {final_score}%");print("\n>>> 新策略被验证为高效，强化相关记忆向量权重。")

# --- 3. 全局数据初始化 ---
trajectories = generate_trajectories()
pre_evo_strategy = calculate_strategy_vector(1, PRE_EVO_KILL_FRAME_LOCAL, PRE_SIM_DURATION, 1.0, 0.3)
post_evo_strategy = calculate_strategy_vector(0, POST_EVO_KILL_FRAME_LOCAL, POST_SIM_DURATION, 5.0, 0.9)
agent1_destroyed = False




# --- 4. 可视化设置 ---
strategy_labels=['攻击性\nAggressiveness','协同性\nCoordination','生存率\nSurvivability','效率\nEfficiency'];
fig=plt.figure(figsize=(18,9));fig.patch.set_facecolor('#1a1a1a');
set_window_geometry(fig, WINDOW_X_POS, WINDOW_Y_POS, WINDOW_WIDTH, WINDOW_HEIGHT)
gs=gridspec.GridSpec(2,2,figure=fig,hspace=0.3);ax_3d=fig.add_subplot(gs[:,0],projection='3d');ax_3d.set_facecolor('black');ax_3d.set_xlim([-ZONE_LIMIT,ZONE_LIMIT]);ax_3d.set_ylim([-ZONE_LIMIT,ZONE_LIMIT]);ax_3d.set_zlim([-ZONE_LIMIT,ZONE_LIMIT]);ax_3d.set_xlabel('沿轨方向 (In-Track)',color='white');ax_3d.set_ylabel('横向 (Cross-Track)',color='white');ax_3d.set_zlabel('径向 (Radial)',color='white');ax_3d.tick_params(axis='x',colors='white');ax_3d.tick_params(axis='y',colors='white');ax_3d.tick_params(axis='z',colors='white');stars=np.random.rand(STAR_COUNT,3)*2*ZONE_LIMIT-ZONE_LIMIT;ax_3d.scatter(stars[:,0],stars[:,1],stars[:,2],s=1,c='white',alpha=0.5);grid_x,grid_z=np.meshgrid(np.linspace(-ZONE_LIMIT,ZONE_LIMIT,10),np.linspace(-ZONE_LIMIT,ZONE_LIMIT,10));grid_y=-0.05*(grid_x**2+grid_z**2)-ZONE_LIMIT;ax_3d.plot_wireframe(grid_x,grid_y,grid_z,color='gray',alpha=0.2);sats_f1=ax_3d.scatter([],[],[],s=180,c=COLOR_FRIENDLY,label='我方卫星');sats_f2=ax_3d.scatter([],[],[],s=180,c=COLOR_FRIENDLY);sats_f3=ax_3d.scatter([],[],[],s=180,c=COLOR_FRIENDLY);sats_e=ax_3d.scatter([],[],[],s=180,marker='o',c=COLOR_ENEMY,label='敌方卫星');attack_lines=[ax_3d.plot([],[],[],lw=2,c=COLOR_ATTACK,alpha=0.8)[0] for _ in range(3)];ax_3d.legend();ax_eval=fig.add_subplot(gs[0,1]);ax_eval.set_facecolor('#2b2b2b');ax_eval.set_xlim([0,1]);ax_eval.set_ylim([0,1]);ax_eval.axis('off');eval_title=ax_eval.text(0.5,0.9,'自主评估反思模块',ha='center',va='center',fontsize=20,color='white',weight='bold');status_texts={'A1':ax_eval.text(0.1,0.7,'我方 卫星-1: ...',fontsize=14,color='white'),'A2':ax_eval.text(0.1,0.6,'我方 卫星-2: ...',fontsize=14,color='white'),'A3':ax_eval.text(0.1,0.5,'我方 卫星-3: ...',fontsize=14,color='white'),'E':ax_eval.text(0.1,0.4,'敌方目标: ...',fontsize=14,color='white'),'loss':ax_eval.text(0.5,0.25,'我方损失: 0',ha='center',fontsize=16,color='white'),'score':ax_eval.text(0.5,0.1,'整体任务评估: ...',ha='center',fontsize=18,color=COLOR_EVOLVED,weight='bold')};num_vars=len(strategy_labels);angles=np.linspace(0,2*np.pi,num_vars,endpoint=False).tolist();angles_closed=angles+angles[:1];ax_evo=fig.add_subplot(gs[1,1],polar=True);ax_evo.set_facecolor('#2b2b2b');ax_evo.text(0.5,1.3,'记忆重构演进模块',transform=ax_evo.transAxes,ha='center',va='center',fontsize=20,color='white',weight='bold');ax_evo.set_yticklabels([]);ax_evo.set_thetagrids(np.degrees(angles),strategy_labels,color='white',fontsize=12);ax_evo.set_rlim(0,1.0);evo_line_pre,=ax_evo.plot([],[],color=COLOR_ENEMY,linewidth=2,linestyle='dashed',label='演进前策略');evo_fill_pre=ax_evo.fill([],[],color=COLOR_ENEMY,alpha=0.3) # MODIFICATION: Added fill for pre-evolution strategy
evo_line_morph,=ax_evo.plot([],[],color=COLOR_EVOLVED,linewidth=2.5,label='演进后策略');evo_fill_morph=ax_evo.fill([],[],color=COLOR_EVOLVED,alpha=0.4);ax_evo.legend(loc='upper right',bbox_to_anchor=(1.3,1.1),labelcolor='white',frameon=False);evo_text=ax_evo.text(np.pi/2,1.33,"等待首次评估...",ha='center',va='center',fontsize=16,color='gray',transform=ax_evo.transData)


# --- 5. 动画更新函数 ---
def update(frame):
    global agent1_destroyed
    # [核心修正] 动画第0帧的“主复位”逻辑
    if frame == 0:
        agent1_destroyed = False
        # 重置所有卫星的视觉状态
        sats_f1.set_visible(True); sats_f1.set_color(COLOR_FRIENDLY)
        sats_f2.set_visible(True)
        sats_f3.set_visible(True)
        sats_e.set_visible(True); sats_e.set_alpha(1)
        # 重置所有文本
        status_texts['A1'].set_text('我方 卫星-1: ...');status_texts['A1'].set_color('white')
        status_texts['A2'].set_text('我方 卫星-2: ...');status_texts['A2'].set_color('white')
        status_texts['A3'].set_text('我方 卫星-3: ...');status_texts['A3'].set_color('white')
        status_texts['loss'].set_text('我方损失: 0')
        status_texts['E'].set_text('敌方目标: 存活');status_texts['E'].set_color('white')
        status_texts['score'].set_text('整体任务评估: ...')
        # 重置雷达图和攻击线
        evo_line_pre.set_data([],[]);evo_line_morph.set_data([],[])
        evo_fill_pre[0].set_xy(np.array([[],[]]).T) # MODIFICATION: Reset the new pre-evolution fill
        evo_fill_morph[0].set_xy(np.array([[],[]]).T);evo_text.set_text("等待首次评估...")
        for line in attack_lines:line.set_data_3d([],[],[])

    # 阶段一: 演进前作战重现
    if frame < PRE_SIM_END_FRAME:
        f = frame
        status_texts['score'].set_text('战场情景重现中...')
        if not agent1_destroyed: sats_f1._offsets3d=([trajectories['pre_A1'][0,f]],[trajectories['pre_A1'][1,f]],[trajectories['pre_A1'][2,f]])
        sats_f2._offsets3d=([trajectories['pre_A2'][0,f]],[trajectories['pre_A2'][1,f]],[trajectories['pre_A2'][2,f]]);sats_f3._offsets3d=([trajectories['pre_A3'][0,f]],[trajectories['pre_A3'][1,f]],[trajectories['pre_A3'][2,f]]);sats_e._offsets3d=([trajectories['pre_E'][0,f]],[trajectories['pre_E'][1,f]],[trajectories['pre_E'][2,f]]);
        if not agent1_destroyed:status_texts['A1'].set_text('我方 卫星-1: 快速拦截中...')
        status_texts['A2'].set_text('我方 卫星-2: 支援机动中...');status_texts['A3'].set_text('我方 卫星-3: 支援机动中...');
        if f == PRE_EVO_LOSS_FRAME_LOCAL:
            p_enemy = trajectories['pre_E'][:,f]
            p_friendly1 = trajectories['pre_A1'][:,f]
            attack_lines[0].set_data_3d([p_enemy[0], p_friendly1[0]], [p_enemy[1], p_friendly1[1]], [p_enemy[2], p_friendly1[2]])

        if f >= PRE_EVO_LOSS_FRAME_LOCAL and not agent1_destroyed:
            agent1_destroyed=True;sats_f1.set_color(COLOR_DESTROYED);status_texts['A1'].set_text('我方 卫星-1: 已被击毁');status_texts['A1'].set_color(COLOR_DESTROYED);status_texts['loss'].set_text('我方损失: 1')
        if f > PRE_EVO_LOSS_FRAME_LOCAL + 10:
            attack_lines[0].set_data_3d([],[],[])
        if f >= PRE_EVO_KILL_FRAME_LOCAL:
            if f==PRE_EVO_KILL_FRAME_LOCAL: p2=trajectories['pre_A2'][:,f];p3=trajectories['pre_A3'][:,f];pe=trajectories['pre_E'][:,f];attack_lines[1].set_data_3d([p2[0],pe[0]],[p2[1],pe[1]],[p2[2],pe[2]]);attack_lines[2].set_data_3d([p3[0],pe[0]],[p3[1],pe[1]],[p3[2],pe[2]])
            sats_e.set_visible(False);status_texts['E'].set_text('敌方目标: 已击毁');status_texts['E'].set_color(COLOR_FRIENDLY);
        if f > PRE_EVO_KILL_FRAME_LOCAL+10:
             for line in attack_lines:line.set_data_3d([],[],[])
    
    # 阶段二: 首次评估
    elif frame < PRE_EVAL_END_FRAME:
        f = frame - PRE_SIM_END_FRAME
        if f < EVAL_DELAY:
            if f == 1: threading.Thread(target=print_evaluation_details, args=("演进前",pre_evo_strategy,65,"有待提高")).start()
            status_texts['score'].set_text('整体任务评估中...')
        else:
            status_texts['score'].set_text('整体任务评估: 65% (有待提高)')
        if f == 1: 
            values=np.concatenate((pre_evo_strategy,[pre_evo_strategy[0]]));
            evo_line_pre.set_data(angles_closed,values);
            evo_fill_pre[0].set_xy(np.array([angles_closed, values]).T) # MODIFICATION: Draw the pre-evolution fill area
            evo_line_pre.set_alpha(0.7);
            evo_text.set_text("演进前策略已评估")
        elif f > EVAL_DELAY+10: evo_text.set_text("记忆向量库更新中...")
    
    # 阶段三: 演进后作战重现
    elif frame < POST_SIM_END_FRAME:
        f = frame - PRE_EVAL_END_FRAME
        if f == 1: 
            sats_f1.set_visible(True);sats_f1.set_color(COLOR_FRIENDLY);sats_e.set_visible(True);
            status_texts['A1'].set_text('我方 卫星-1: ...');status_texts['A1'].set_color('white');status_texts['loss'].set_text('我方损失: 0');status_texts['E'].set_text('敌方目标: 存活');status_texts['E'].set_color('white');
            evo_text.set_text("应用新策略重现中...")
            for line in attack_lines:line.set_data_3d([],[],[])
        status_texts['score'].set_text('战场情景重现中...')
        sats_f1._offsets3d=([trajectories['post_A1'][0,f]],[trajectories['post_A1'][1,f]],[trajectories['post_A1'][2,f]]);sats_f2._offsets3d=([trajectories['post_A2'][0,f]],[trajectories['post_A2'][1,f]],[trajectories['post_A2'][2,f]]);sats_f3._offsets3d=([trajectories['post_A3'][0,f]],[trajectories['post_A3'][1,f]],[trajectories['post_A3'][2,f]]);sats_e._offsets3d=([trajectories['post_E'][0,f]],[trajectories['post_E'][1,f]],[trajectories['post_E'][2,f]]);
        status_texts['A1'].set_text('我方 卫星-1: 安全协同机动...');status_texts['A2'].set_text('我方 卫星-2: 安全协同机动...');status_texts['A3'].set_text('我方 卫星-3: 安全协同机动...');
        if f >= POST_EVO_KILL_FRAME_LOCAL:
            if f==POST_EVO_KILL_FRAME_LOCAL: p1=trajectories['post_A1'][:,f];p2=trajectories['post_A2'][:,f];p3=trajectories['post_A3'][:,f];pe=trajectories['post_E'][:,f];attack_lines[0].set_data_3d([p1[0],pe[0]],[p1[1],pe[1]],[p1[2],pe[2]]);attack_lines[1].set_data_3d([p2[0],pe[0]],[p2[1],pe[1]],[p2[2],pe[2]]);attack_lines[2].set_data_3d([p3[0],pe[0]],[p3[1],pe[1]],[p3[2],pe[2]])
            sats_e.set_visible(False);status_texts['E'].set_text('敌方目标: 已击毁');status_texts['E'].set_color(COLOR_FRIENDLY);
        if f > POST_EVO_KILL_FRAME_LOCAL+10:
             for line in attack_lines:line.set_data_3d([],[],[])

    # 阶段四: 最终评估与演进分析
    else:
        f = frame - POST_SIM_END_FRAME
        if f < EVAL_DELAY:
            if f == 1: threading.Thread(target=print_evaluation_details, args=("演进后",post_evo_strategy,95,"优秀")).start()
            status_texts['score'].set_text('整体任务评估中...')
        else:
            status_texts['score'].set_text('整体任务评估: 95% (优秀)')
        
        # MODIFICATION: This block now draws the final post-evolution strategy statically at the beginning of the phase.
        if f == 1: 
            evo_text.set_text("策略演进分析中...")
            post_values = np.concatenate((post_evo_strategy, [post_evo_strategy[0]]))
            evo_line_morph.set_data(angles_closed, post_values)
            evo_fill_morph[0].set_xy(np.array([angles_closed, post_values]).T)

    # MODIFICATION: Added evo_fill_pre[0] to the list of returned artists
    return [sats_f1,sats_f2,sats_f3,sats_e,evo_line_pre,evo_line_morph,evo_fill_pre[0]]+list(status_texts.values())

# --- 6. 运行与保存 ---
ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=40, blit=False, repeat=True)
plt.tight_layout(pad=3.0)
plt.show()