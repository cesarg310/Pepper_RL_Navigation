from pepperLaserEnv import pepperEnv
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

#Objetivo 
ox = 3.5
oy = 0

env = pepperEnv(obx = ox, oby = oy)

model = PPO.load("laser30of")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1, warn=False)

env.close()

