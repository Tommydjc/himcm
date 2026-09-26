import os
import sys

print("\n" + "=" * 65)
print("🚀 [Step 1/3] 正在启动 GAT-PPO 强化学习推理演示...")
print("=" * 65)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

try:
    from src.environment.layouts import get_office_layout
    from src.rl.env_wrapper import GraphEvacuationEnv
    from src.rl.gat_net import GATActorCritic
    print("✅ [Step 2/3] 核心环境与神经网络架构模块导入成功！")
except Exception as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

def run():
    checkpoint_path = os.path.join(PROJECT_ROOT, "src", "rl", "checkpoints", "gat_ppo_office.pt")
    if not os.path.exists(checkpoint_path):
        print(f"❌ 未找到权重文件: {checkpoint_path}")
        return

    device = torch.device('mps' if torch.backends.mps.is_available() else '1. 修复参数名：使用 n_agents=2 初始化环境
    graph = get_office_layout()
    try:
        env = GraphEvacuationEnv(graph=graph, n_agents=2)
    except TypeError:
        env = GraphEvacuationEnv(graph=graph)
    obs = env.reset()

    # 2. 载入模型权重
    policy = GATActorCritic(in_dim=6, hidden_dim=64, num_heads=4).to(device)
    try:
        policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
        policy.eval()
        print("✅ 权重载入成功！开始执行多消防员扫荡...")
    except Exception as e:
        print(f"⚠️ 权重加载提示: {e}")

    # 3. 逐步仿真执行
    print("-" * 65)
    step = 0
    trajectories = {0: [], 1: []}

    while step < 60:
        step += 1
        with torch.no_grad():
            # 兼容不同的动作选择方法名 (act / get_action / forward)
            if hasattr(policy, 'act'):
                actions = policy.act(obs, deterministic=False)
            elif hasattr(policy, 'get_action'):
             nv.step(actions)
        if len(step_result) == 4:
            next_obs, rewards, done, info = step_result
        else:
            next_obs, rewards, terminated, truncated, info = step_result
            done = terminated or truncated
        obs = next_obs

        # 获取当前消防员位置
        positions = getattr(env, 'agent_positions', getattr(env, 'agents', [f'Node_{actions[0]}', f'Node_{actions[1]}']))
        for a_id in range(2):
            pos_val = positions[a_id] if a_id < len(positions) else "unknown"
            trajectories[a_id].append(pos_val)

        if step <= 5 or step % 5 == 0:
            print(f"⏱️ 第 {step:02d} 步 | 消防员 1: {trajectories[0][-1]} | 消防员 2: {trajectories[1][-1]}")

        if done or (isinstance(info, dict) and info.get("all_cleared", False)):
            print("-" * 65)
            print(f"🎉 任务达成！全楼在第 {step} 步完成清空！")
            break

    print("=" * 65)
    print("📊 [Step 3/3] 最终行动路径汇总 (Acstr(p) for p in trajectories[a_id]])}")
    print("=" * 65)

if __name__ == '__main__':
    run()
