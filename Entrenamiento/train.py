import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from pepperObjEnv1 import pepperEnv1
from pepperObjEnv2 import pepperEnv2
from pepperObjEnv3 import pepperEnv3
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
import torch as th


def make_env(cl):
    
    def _init():
        ox = list(np.linspace(-5,-2,5))
        x = list(np.linspace(2,5,5))
        oy = list(np.linspace(-5,5,15))

        for i in range(len(x)):
            ox.append(x[i])

        if cl == 0:
            env = pepperEnv1(render=False, obx = ox, oby = oy)
        elif cl == 1:
            env = pepperEnv2(render=False, obx = ox, oby = oy)
        
        elif cl == 2:
            env = pepperEnv3(render=False, obx = ox, oby = oy)

        return env

    return _init


if __name__ == "__main__":

    num_cpu = 3 
    env = DummyVecEnv([make_env(i) for i in range(num_cpu)])

    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[dict(pi=[128, 128, 128], vf=[128, 128, 128])])

    model = PPO("MlpPolicy", env, verbose=1, n_steps=150, tensorboard_log= "./model1",batch_size= 50, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=51000)

    model.save('model1')

    print('Modelo guardado')

