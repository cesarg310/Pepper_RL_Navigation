import gym
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from gym import spaces
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

class pepperEnv(gym.Env):

    def callback_laser(self, data):

        self.right = min(data.ranges[0:15])
        self.front = min(data.ranges[23:38])
        self.left  = min(data.ranges[46:61])
        if self.right > 3:
            self.right = 3
        if self.front > 3:
            self.front = 3
        if self.left > 3:
            self.left = 3


    def callback_odom(self, data):
        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y
        self.theta = data.pose.pose.orientation.z

    def __init__(self, obx = 1, oby = 1):

        super(pepperEnv, self).__init__()

        rospy.init_node('agent')


        self.sub2 = rospy.Subscriber('/laser',LaserScan, self.callback_laser)

        self.sub1 = rospy.Subscriber('/odom',Odometry, self.callback_odom)

        self.init_pos = [0,0,0]

        self.x, self.y, self.theta = 0,0,0
        self.right, self.front, self.left = 0,0,0

        while self.init_pos[0] == 0 and self.init_pos[1] == 0 and self.init_pos[2] == 0:

            self.init_pos = [self.x, self.y, self.theta]
    

        self.obx = obx
        self.oby = oby

        low = np.array([-10, -10, -np.pi, 0, 0, 0],dtype=np.float32,)
        high = np.array([10, 10, np.pi, 3, 3, 3],dtype=np.float32,)

        #Observaciones
        self.observation_space = spaces.Box(low,high,dtype=np.float32)

        vel_low = np.array([0, -3, -1],dtype=np.float32,)
        vel_high = np.array([3, 3, 1],dtype=np.float32,)

        #Acciones
        self.action_space = spaces.Box(vel_low,vel_high,dtype=np.float32)

        print(self.x, ',', self.y, ',', self.theta)

        self.objetivo = [self.obx,self.oby]

        pos = [self.objetivo[0]- self.x, self.objetivo[1] - self.y, self.theta]

        #Estado actual
        self.state = np.array([pos[0], pos[1], pos[2], self.right, self.front, self.left])

        print('Estado inicial: ', self.state)

        #Pasos
        self.steps = 0

    def move(self, action):
        rospy.sleep(1)
        twistmessage = Twist()
        twistmessage.linear.x = action[0]
        twistmessage.linear.y = action[1]
        twistmessage.angular.z = action[2]
        self.cmdPublisher.publish(twistmessage)
        
    def step(self, action):

        info = {}
        print(action)

        self.move(action)

        pos = [self.objetivo[0]- self.x, self.objetivo[1] - self.y, self.theta]

        #Estado actual
        self.state = np.array([pos[0], pos[1], pos[2], self.right, self.front, self.left])

        print('Estado :', self.state)

        self.steps += 1

        print('Frente = ', self.front)

        d = np.linalg.norm(np.array(self.state[0:2]))

        terminated = bool(
            d < 1
            or self.steps > 250
            or self.left < 0.5
            or self.right < 0.5
            or self.front < 0.5
            or d >= 10
        )

        if d < 1:
            reward = 2 - self.steps*0.01
            print('LLego al objetvio')

        elif self.left < 0.5 or self.right < 0.5 or self.front < 0.5:
            reward = -5
            print('Se choco con un obstaculo')

        elif d >= 10 or (self.steps > 250 and d >= 10):
            reward = -4
            print('Se fue muy lejos')

        elif self.steps > 250:
            reward = -1-(2*d)/10
            print('No llego a ningun lado')

        else:
            reward = 0

        return np.array(self.state, dtype=np.float32), reward, terminated, info

    def reset(self):

        self.init_pos = [0,0,0]

        self.x, self.y, self.theta = 0,0,0
        self.right, self.front, self.left = 0,0,0

        self.sub2 = rospy.Subscriber('/laser',LaserScan, self.callback_laser)

        self.sub1 = rospy.Subscriber('/odom',Odometry, self.callback_odom)

        while self.init_pos[0] == 0 and self.init_pos[1] == 0 and self.init_pos[2] == 0:

            self.init_pos = [self.x, self.y, self.theta] 

        print('Objetivo = ', [self.obx, self.oby])

        print(self.x, ',', self.y, ',', self.theta)

        print('Derecha = ', self.right)
        print('Izquierda = ', self.left)
        print('Frente = ', self.front)

        self.objetivo = [self.obx,self.oby]

        pos = [self.objetivo[0]- self.x, self.objetivo[1] - self.y, self.theta]

        #Estado actual
        self.state = np.array([pos[0], pos[1], pos[2], self.right, self.front, self.left])

        print('Estado inicial al reiniciar: ', self.state)

        #Pasos
        self.steps = 0


        return np.array(self.state, dtype=np.float32)