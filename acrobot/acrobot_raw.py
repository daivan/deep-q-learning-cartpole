import gym
env = gym.make('Acrobot-v1')
for i_episode in range(20):
    observation = env.reset()
    for t in range(10000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()