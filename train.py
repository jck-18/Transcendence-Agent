from transcendence_env import TranscendenceGridEnv
env = TranscendenceGridEnv()

obs = env.reset()
done = False

total_reward = 0
while not done:
    action = env.action_space.sample()  # for now: random
    obs, reward, done, _ = env.step(action)
    total_reward += reward

print("Episode finished with reward:", total_reward)
