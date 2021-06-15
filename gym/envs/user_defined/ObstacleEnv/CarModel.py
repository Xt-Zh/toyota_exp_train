import os

import numpy as np
import tensorflow as tf
from easydict import EasyDict as edict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class VehicleDynamics(object):
    def __init__(self,bs):
        """
        初始化
        :param bs: 即batch_size
        """
        self.obses = None # tensor, shape:(bs, 6)
        self.actions = None # tensor, shape(bs, 2)
        self.frequency = 10
        self.safety = np.ones(shape=(bs, 1),dtype=int)

        self.car_info = edict({'width': 2,  # x_length
                               'height': 0.8  # y_length
                               })
        self.constraint = edict({'max_y': 6.0,
                                 'min_y': 0.0,
                                 'max_speed': 10.0,
                                 'max_x': 30.0,
                                 'min_x': -30.0,
                                 })
        self.obstacle_info = edict(dict(x=10,
                                        y=0.5,
                                        width=5,
                                        height=2))


        self.vehicle_params = dict(C_f=-155495.0,  # front wheel cornering stiffness [N/rad]
                                   C_r=-155495.0,  # rear wheel cornering stiffness [N/rad]
                                   a=1.19,  # distance from CG to front axle [m]
                                   b=1.46,  # distance from CG to rear axle [m]
                                   mass=1520.,  # mass [kg]
                                   I_z=2642.,  # Polar moment of inertia at CG [kg*m^2]
                                   miu=0.8,  # tire-road friction coefficient
                                   g=9.81,  # acceleration of gravity [m/s^2]
                                   )
        a, b, mass, g = self.vehicle_params['a'], self.vehicle_params['b'], \
                        self.vehicle_params['mass'], self.vehicle_params['g']
        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        self.vehicle_params.update(dict(F_zf=F_zf,
                                        F_zr=F_zr))

        self.expected_speed = 5

    def f_xu(self, states, actions, tau):  # states and actions are tensors, [[], [], ...]
        """
        动力学模型

        :param states:  shape:(bs,6) : vx, vy, r, x, y, phi
        :param actions: shape:(bs,2) : steer, acceleration
        :param tau:  单步时间步长, float
        :return:
        """

        v_x, v_y, r, x, y, phi = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4], states[:, 5]
        phi = phi * np.pi / 180.
        steer, a_x = actions[:, 0], actions[:, 1]
        C_f = tf.convert_to_tensor(self.vehicle_params['C_f'], dtype=tf.float32)
        C_r = tf.convert_to_tensor(self.vehicle_params['C_r'], dtype=tf.float32)
        a = tf.convert_to_tensor(self.vehicle_params['a'], dtype=tf.float32)
        b = tf.convert_to_tensor(self.vehicle_params['b'], dtype=tf.float32)
        mass = tf.convert_to_tensor(self.vehicle_params['mass'], dtype=tf.float32)
        I_z = tf.convert_to_tensor(self.vehicle_params['I_z'], dtype=tf.float32)
        miu = tf.convert_to_tensor(self.vehicle_params['miu'], dtype=tf.float32)
        g = tf.convert_to_tensor(self.vehicle_params['g'], dtype=tf.float32)

        F_zf, F_zr = b * mass * g / (a + b), a * mass * g / (a + b)
        F_xf = tf.where(a_x < 0, mass * a_x / 2, tf.zeros_like(a_x))
        F_xr = tf.where(a_x < 0, mass * a_x / 2, mass * a_x)
        miu_f = tf.sqrt(tf.square(miu * F_zf) - tf.square(F_xf)) / F_zf
        miu_r = tf.sqrt(tf.square(miu * F_zr) - tf.square(F_xr)) / F_zr
        alpha_f = tf.atan((v_y + a * r) / (v_x+1e-8)) - steer
        alpha_r = tf.atan((v_y - b * r) / (v_x+1e-8))

        next_state = [v_x + tau * (a_x + v_y * r),
                      (mass * v_y * v_x + tau * (a * C_f - b * C_r) * r - tau * C_f * steer * v_x - tau * mass * tf.square(v_x) * r) / (mass * v_x - tau * (C_f + C_r)),
                      (-I_z * r * v_x - tau * (a * C_f - b * C_r) * v_y + tau * a * C_f * steer * v_x) / (tau * (tf.square(a) * C_f + tf.square(b) * C_r) - I_z * v_x),
                      x + tau * (v_x * tf.cos(phi) - v_y * tf.sin(phi)),
                      y + tau * (v_x * tf.sin(phi) + v_y * tf.cos(phi)),
                      (phi + tau * r) * 180 / np.pi]


        return tf.stack(next_state, 1), tf.stack([alpha_f, alpha_r, miu_f, miu_r], 1)

    def prediction(self, x_1, u_1, frequency):
        x_next, next_params = self.f_xu(x_1, u_1, 1 / frequency)
        return x_next, next_params

    def rollout_out(self, actions):  # obses and actions are tensors, think of actions are in range [-1, 1]
        # actions shape: (bs, 2)
        with tf.name_scope('model_step') as scope:
            rewards  = self._get_reward(actions[:,1])
            self.obses, _ = self.prediction(self.obses, actions, self.frequency)
            is_not_safe = self._judge_collision(self.obses[:,3],self.obses[:,4])

        return self.obses, rewards, is_not_safe

    def _get_reward(self,action):
        """
        根据当前状态获取reward（还没有考虑动作的影响）
        :return: shape:(bs,)
        """
        y = self.obses[:,4]
        vx = self.obses[:,0]
        phi = self.obses[:,5]
        reward = (tf.square(y - self.constraint.min_y)+ tf.square(vx - self.expected_speed) + tf.square( phi * np.pi / 180) + tf.square(action)) / self.frequency
        return reward

    # def _get_next_observations(self, actions):
    #     new_states, _ = self.prediction(self.obses, actions, self.frequency)
    #     return new_states

    def _judge_collision(self, xs, ys):
        obs_x = self.obstacle_info.x
        obs_width = self.obstacle_info.width
        obs_y = self.obstacle_info.y
        obs_height = self.obstacle_info.height

        is_collapse = ((((obs_x <= xs) & (xs <= obs_x + obs_width)) |
                        ((obs_x <= xs + self.car_info.width) & (xs + self.car_info.width <= obs_x + obs_width))) &
                       (((obs_y <= ys) & (ys <= obs_y + obs_height)) |
                        ((obs_y <= ys + self.car_info.height) & (ys + self.car_info.height <= obs_y + obs_height)))) | \
                      (ys < self.constraint.min_y) & (ys < self.constraint.max_y)

        # if obs_x <= xs <= obs_x + obs_width or \
        #         obs_x <= xs + self.car_info.width <= obs_x + obs_width:
        #     if obs_y <= ys <= obs_y + obs_height or \
        #             obs_y <= ys + self.car_info.height <= obs_y + obs_height:
        #         return True
        # return False

        return is_collapse

    def reset(self, obses):  # input are all tensors
        self.obses = obses
        self.actions = None
        self.reward_info = None

if __name__ == '__main__':
    from gym.envs.user_defined.ObstacleEnv.CarObstacle import CarEnv
    env=CarEnv()
    obs_list = []
    obs = env.reset()
    for i in range(50):
        obs_list.append(obs)
    while True:
        action = np.array([-0.05, 2], dtype=np.float32)
        obs, reward, done, info = env.step(action)
        env.render()
        if done: env.reset()
    # obses = np.stack(obs_list, 0)
    # dym=VehicleDynamics(bs=10)
    # actions = tf.tile(tf.constant([[0.5, -0.5]], dtype=tf.float32), tf.constant([len(obses), 1]))
    # a,b=dym.prediction(obses,actions,2.0)
    # print(a)
    # print(b)
