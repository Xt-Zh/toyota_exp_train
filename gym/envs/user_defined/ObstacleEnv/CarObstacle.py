import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from easydict import EasyDict as edict
from numpy.random import normal, uniform

import gym
from gym.envs.user_defined.ObstacleEnv.CarModel import VehicleDynamics

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class CarSimpleEnv(gym.Env):
    def __init__(self):
        metadata = {'render.modes': []}
        self.action_space = gym.spaces.Box(low=1, high=1, shape=(1,))  # 动作空间
        self.observation_space = gym.spaces.Box(low=-10, high=10, shape=(1,))  # 状态空间
        self.obs = self._reset_init_state()
        self.action = 0.0
        self.done = False

    def reset(self):
        self.obs = self._reset_init_state()
        self.action = 0.0
        print(self.obs)
        return self.obs

    def step(self, action):
        reward = self._get_reward()
        done = self._get_done()
        self.obs = self._get_next_observation(action)
        info = {'action': action}  # 用于记录训练过程中的环境信息,便于观察训练状态
        return self.obs, reward, done, info

        # 根据需要设计相关辅助函数

    def render(self, **kwargs):

        car_x_length = 2
        car_y_length = 2

        plt.ion()
        plt.cla()
        plt.title("Avoiding Obstacle")
        ax = plt.axes(xlim=(-10, 10),
                      ylim=(-10, 10))
        plt.axis("equal")
        # plt.axis('off')

        plt.plot([-10, 10], [0, 0])
        plt.plot([-10, 10], [5, 5])
        ax.add_patch(plt.Rectangle((self.obs, 0), car_x_length, car_y_length, edgecolor='black', facecolor='none'))

        text_x, text_y = -5, 8
        plt.text(text_x, text_y, f'x: {self.obs:.2f}m')

        plt.show()
        plt.pause(0.5)

    def _get_next_observation(self, action):
        return self.obs + action

    def _get_reward(self):
        reward = 1
        return reward

    def _get_done(self):
        if self.obs < -10 or self.obs > 10:
            self.done = True
        else:
            self.done = False
        return self.done

    def _reset_init_state(self):
        return float(np.random.randn() * 5)


class CarEnv(gym.Env):
    def __init__(self):
        # state: v_x, v_y, r, x, y, phi
        metadata = {'render.modes': []}

        self.action_space = gym.spaces.Box(low=-np.inf * np.ones(2),
                                           high=-np.inf * np.ones(2),
                                           shape=(2,))  # 动作空间
        self.observation_space = gym.spaces.Box(low=-np.inf * np.ones(6),
                                                high=-np.inf * np.ones(6),
                                                shape=(6,))  # 状态空间
        self.obs = self._reset_init_state()
        self.action = None
        self.reward = None

        self.value = 0
        self.done = False

        self.dynamics = VehicleDynamics(bs=10)

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

        self.plot_option = edict(dict(xmin=-15,
                                      xmax=30,
                                      ymin=-2,
                                      ymax=15,
                                      ))

        self.expected_speed = 5

        self.frequency = 10

        self.fig = plt.figure(num='one', figsize=(10, 5))

    def reset(self):
        self.obs = self._reset_init_state()
        self.action = np.array([0.0, 0.0], dtype=np.float32)
        self.reward = self._get_reward()

        plt.clf()
        return self.obs

    def step(self, action: np.ndarray):
        # action: steer, a_x
        reward = self._get_reward(action[0])
        done = self._get_done()
        obs = self._get_next_observation(action)
        self.obs = obs
        self.action = action
        self.reward = reward
        info = {'action': action, 'done_info': done, 'reward_info': 0.0}  # 用于记录训练过程中的环境信息,便于观察训练状态
        return obs, reward, done, info

    def render(self, **kwargs):
        x, y, phi = self.obs[3], self.obs[4], self.obs[5]
        plt.ion()
        plt.figure('one')
        plt.cla()

        plt.title("Avoiding Obstacle")
        ax = plt.axes()
        ax.set_aspect('equal')
        ax.set_xlim([min(x - 2, self.plot_option.xmin), max(x + 2, self.plot_option.xmax)])
        ax.set_ylim([self.plot_option.ymin, self.plot_option.ymax])

        plt.axhline(y=self.constraint['max_y'], lw=2, color='k')
        plt.axhline(y=self.constraint['min_y'], lw=2, color='k')
        plt.axhline(y=0.5 * (self.constraint['min_y'] + self.constraint['max_y']), lw=1, ls='--', color='k')

        obstacle_x = self.obstacle_info.x
        obstacle_y = self.obstacle_info.y
        obstacle_width = self.obstacle_info.width
        obstacle_height = self.obstacle_info.height

        ax.add_patch(plt.Rectangle((obstacle_x, obstacle_y),
                                   obstacle_width,
                                   obstacle_height, edgecolor='black', facecolor='black'))

        Cos = lambda x: np.cos(x * np.pi / 180.0)
        Sin = lambda x: np.sin(x * np.pi / 180.0)

        width = self.car_info.width
        height = self.car_info.height

        plt.plot([x, x + width * Cos(phi)], [y, y + width * Sin(phi)], 'r')
        plt.plot([x, x - height * Sin(phi)], [y, y + height * Cos(phi)], 'r')
        plt.plot([x + width * Cos(phi), x + width * Cos(phi) - height * Sin(phi)],
                 [y + width * Sin(phi), y + width * Sin(phi) + height * Cos(phi)], 'r')
        plt.plot([x - height * Sin(phi), x + width * Cos(phi) - height * Sin(phi)],
                 [y + height * Cos(phi), y + width * Sin(phi) + height * Cos(phi)], 'r')

        text_x = self.plot_option.xmin + 5
        text_xr = self.plot_option.xmax - 10
        text_y = self.plot_option.ymax - 1
        plt.text(text_x, text_y, f'x: {self.obs[3]:.2f}m')
        plt.text(text_x, text_y - 1, f'y: {self.obs[4]:.2f}m')
        plt.text(text_x, text_y - 2, f'phi:{self.obs[5]:.2f}')
        plt.text(text_x, text_y - 3, f'v_x: {self.obs[0]:.2f}m/s')
        plt.text(text_x, text_y - 4, f'v_y: {self.obs[1]:.2f}m/s')
        plt.text(text_x, text_y - 5, f'r: {self.obs[3]:.2f}')

        plt.text(text_xr, text_y, f'reward: {self.reward:.3f}')
        plt.text(text_xr, text_y - 1, f'action: {float(self.action[0]):.3f} {float(self.action[1]):.3f}')
        plt.text(text_xr, text_y - 2, f'value: {float(self.value):.3f}')

        plt.show()
        plt.pause(0.05)

    def _get_next_observation(self, action):
        state = tf.convert_to_tensor(self.obs[np.newaxis, :], dtype=tf.float32)
        action = tf.convert_to_tensor(action[np.newaxis, :], dtype=tf.float32)
        new_state, _ = self.dynamics.prediction(state, action, self.frequency)
        return np.array(new_state).flatten()

    def _get_reward(self,action=0.0):
        y = self.obs[4]
        vx = self.obs[0]
        phi = self.obs[5]
        reward = ((y - self.constraint.min_y) ** 2 + (vx - self.expected_speed) ** 2 + (phi * np.pi/ 180) ** 2 + action ** 2) / self.frequency
        # reward: 到底部车道线的距离的平方 + 与期望速度之差的平方 + 转向角的平方
        return reward

    def _judge_collision(self, x, y):
        obs_x = self.obstacle_info.x
        obs_width = self.obstacle_info.width
        obs_y = self.obstacle_info.y
        obs_height = self.obstacle_info.height
        if obs_x <= x <= obs_x + obs_width or \
                obs_x <= x + self.car_info.width <= obs_x + obs_width:
            if obs_y <= y <= obs_y + obs_height or \
                    obs_y <= y + self.car_info.height <= obs_y + obs_height:
                return True
        return False

    def _get_done(self):
        v_x, v_y, _, x, y, _ = self.obs

        if x < self.constraint.min_x or \
                x > self.constraint.max_x or \
                y > self.constraint.max_y or \
                y < self.constraint.min_y or \
                self._judge_collision(x, y):

            self.done = True
        else:
            self.done = False
        return self.done

    def _reset_init_state(self):
        # v_x_mean = 2.0
        # v_y_mean = 0.3
        # r_mean = 0.0
        # x_mean = -10.0
        # y_mean = 0.5
        # phi_mean = 0
        #
        # state_ = np.hstack([normal(v_x_mean, 1.0, ),
        #                     normal(v_y_mean, 0.3, ),
        #                     normal(r_mean, 1, ),
        #                     normal(x_mean, 5, ),
        #                     normal(y_mean, 1, ),
        #                     normal(phi_mean, 10, ),
        #                     ]).astype(np.float32)

        state_ = np.hstack([uniform(0, 5, ),
                            uniform(0, 5, ),
                            uniform(-1, 1, ),
                            uniform(-10, 5, ),
                            uniform(1, 5, ),
                            uniform(-1, 1, ),
                            ]).astype(np.float32)

        return state_

    def rollout_out(self, actions):  # obses and actions are tensors, think of actions are in range [-1, 1]
        with tf.name_scope('model_step') as scope:
            rewards = self._get_reward(actions[0])
            self.obs = self._get_next_observation(actions)
            is_safe = self._judge_collision(self.obs[3], self.obs[4])

        return self.obs, rewards, is_safe

    def set_value(self, v):
        self.value = v


def test_env():
    env = CarSimpleEnv()
    obs = env.reset()
    i = 0
    done = 0
    while True:
        action = 1.0 if np.random.random() < 0.5 else -1.0
        obs, reward, done, info = env.step(action)
        env.render()
        if done: env.reset()
        print(obs, reward, done, info)


if __name__ == '__main__':
    test_env()
