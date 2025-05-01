from transcendence_env import TranscendenceGridEnv
from agent import FrustrationAgent

env = TranscendenceGridEnv()
agent = FrustrationAgent(env)

episode_rewards = []

for ep in range(10):  # run a few episodes
    obs = env.reset()
    agent.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.choose_action(obs, env.agent_pos, env.goal)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    print(f"Episode {ep+1}: Reward = {total_reward:.2f}")
    episode_rewards.append(total_reward)
