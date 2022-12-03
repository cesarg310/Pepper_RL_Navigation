import gym
import numpy as np
from qibullet import SimulationManager
from gym import spaces
import pybullet

class pepperEnv1(gym.Env):

    def __init__(self, render = False, obx = 1, oby = 1):

        super(pepperEnv1, self).__init__()
        self.obx = obx
        self.oby = oby
        self.render = render

        #Inicio simulacion
        self.simulation_manager = SimulationManager()
        self.client1 = self.simulation_manager.launchSimulation(gui=self.render, use_shared_memory=True)
        self.pepper = self.simulation_manager.spawnPepper(self.client1, spawn_ground_plane=True)

        low = np.array([-10, -10, -np.pi, 0, 0, 0],dtype=np.float32,)
        high = np.array([10, 10, np.pi, 3, 3, 3],dtype=np.float32,)

        #Observaciones
        self.observation_space = spaces.Box(low,high,dtype=np.float32)

        vel_low = np.array([0, -3, -1],dtype=np.float32,)
        vel_high = np.array([3, 3, 1],dtype=np.float32,)

        #Acciones
        self.action_space = spaces.Box(vel_low,vel_high,dtype=np.float32)

        #Posicion objetivo

        posx = np.random.choice(obx)
        posy = np.random.choice(oby)
        self.objetivo = [posx,posy]

        self.pepper.subscribeLaser()

        self.left, self.front, self.right = self.laserInfo()

        #Estado actual
        self.state = self.get_obs()

        #print(self.state)

        #Pasos
        self.steps = 0
    
    def get_obs(self):
        pos = self.get_pos()
        return  np.array([pos[0], pos[1], pos[2], min(self.right), min(self.front), min(self.left)])

    def get_pos(self):
        x, y, theta = self.pepper.getPosition()
        return [self.objetivo[0]-x, self.objetivo[1]-y, theta]


    #Realizar movimientos
    def mover(self, action):
        
        self.pepper.move(action[0],action[1],action[2])

    def step(self, action):

        info = {}

        self.mover(action)

        self.left, self.front, self.right = self.laserInfo()

        self.state = self.get_obs()

        self.steps += 1

        d = np.linalg.norm(np.array(self.state[0:2]))

        terminated = bool(
            d < 1
            or self.steps > 250
            or min(self.left) < 0.5
            or min(self.right) < 0.5
            or min(self.front) < 0.5
            or d >= 10
        )

        if d < 1:
            reward = 2 - self.steps*0.01
            print('1 LLego al objetivo')

        elif min(self.left) < 0.5 or min(self.right) < 0.5 or min(self.front) < 0.5:
            reward = -5
            print('1 Se choco con un obstaculo')

        elif d >= 10 or (self.steps > 250 and d >= 10):
            reward = -4
            print('1 Se fue muy lejos')

        elif self.steps > 250:
            reward = -1-(2*d)/10
            print('1 No llego a ningun lado')

        else:
            reward = 0


        return self.state, reward, terminated, info

    def laserInfo(self):
        
        left = self.pepper.getLeftLaserValue()
        rigth = self.pepper.getRightLaserValue()
        front = self.pepper.getFrontLaserValue()

        return left, front, rigth

    def reset(self):

        self.simulation_manager.resetSimulation(self.client1)
        self.pepper = self.simulation_manager.spawnPepper(self.client1, spawn_ground_plane=True)


        #Muros
        Muro3 = pybullet.loadURDF("wallY.urdf",basePosition=[0, 5.5, 1])
        Muro4 = pybullet.loadURDF("wallY.urdf",basePosition=[0, -5.5, 1])
        Muro1 = pybullet.loadURDF("wallX.urdf",basePosition=[5.5, 0, 1])
        Muro2 = pybullet.loadURDF("wallX.urdf",basePosition=[-5.5, 0, 1])
    
        # Obstaculo
        obs1 = pybullet.loadURDF("obs.urdf",basePosition=[2.2, 0, 0.5])
        obs2 = pybullet.loadURDF("obs.urdf",basePosition=[3.2, 3.2, 0.5])
        obs3 = pybullet.loadURDF("obs.urdf",basePosition=[-3.2, -2.2, 0.5])
        obs4 = pybullet.loadURDF("obs.urdf",basePosition=[-1.2, 2.2, 0.5])
        obs5 = pybullet.loadURDF("obs.urdf",basePosition=[1.2, -3.2, 0.5])
        obs6 = pybullet.loadURDF("obs.urdf",basePosition=[-2.2, 3.2, 0.5])
        obs7 = pybullet.loadURDF("obs.urdf",basePosition=[-1.2, -2.2, 0.5])

        self.pepper.goToPosture("Stand", 0.6)

        posx = np.random.choice(self.obx)
        posy = np.random.choice(self.oby)
        self.objetivo = [posx,posy]

        self.pepper.subscribeLaser()

        self.left, self.front, self.right = self.laserInfo()

        #Estado inicial
        self.state = self.get_obs()

        self.steps = 0

        return self.state

    def close(self):

        #Parar simulacion 
        self.simulation_manager.stopSimulation(self.client1)