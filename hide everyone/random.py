import gym
env = gym.make('CartPole-v0')
env.reset()
env.render()
for _ in range(1000):
	env.step(env.action_space.sample())
env.close()