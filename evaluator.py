#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: evaluator.py
# =====================================

import logging
import os

import matplotlib.pyplot as plt
import numpy as np

import gym
from preprocessor import Preprocessor
from utils.misc import TimerStat

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Evaluator(object):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')

    def __init__(self, policy_cls, env_id, args):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self.args = args
        self.env = gym.make(env_id)
        self.policy_with_value = policy_cls(self.args)
        self.iteration = 0
        if self.args.mode == 'training':
            self.log_dir = self.args.log_dir + '/evaluator'
        else:
            self.log_dir = self.args.test_log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.preprocessor = Preprocessor((self.args.obs_dim,), self.args.obs_preprocess_type,
                                         self.args.reward_preprocess_type,
                                         self.args.obs_scale, self.args.reward_scale, self.args.reward_shift,
                                         gamma=self.args.gamma)

        self.writer = self.tf.summary.create_file_writer(self.log_dir)
        self.stats = {}
        self.eval_timer = TimerStat()
        self.eval_times = 0

    def get_stats(self):
        self.stats.update(dict(eval_time=self.eval_timer.mean))
        return self.stats

    def load_weights(self, load_dir, iteration):
        self.policy_with_value.load_weights(load_dir, iteration)

    def load_ppc_params(self, load_dir):
        self.preprocessor.load_params(load_dir)

    def evaluate_saved_model(self, model_load_dir, ppc_params_load_dir, iteration):
        self.load_weights(model_load_dir, iteration)
        self.load_ppc_params(ppc_params_load_dir)

    def run_an_episode(self, mode, steps=None, render=True, ):
        reward_list = []
        reward_info_dict_list = []
        action_list = []
        done = 0
        obs = self.env.reset()
        # if render: self.env.render()
        if steps is not None:
            for _ in range(steps):
                processed_obs = self.preprocessor.tf_process_obses(obs)
                action = self.policy_with_value.compute_mode(processed_obs[np.newaxis, :])
                value = self.policy_with_value.compute_value_net(processed_obs[np.newaxis, :])
                self.env.set_value(value)
                obs, reward, done, info = self.env.step(action.numpy()[0])
                reward_info_dict_list.append(info['reward_info'])
                if render:
                    if mode == 'test':
                        xmin, xmax, ymin, ymax = (self.env.plot_option.xmin, self.env.plot_option.xmax,
                                                  self.env.boundary_info.min_y, self.env.boundary_info.max_y)

                        xs = np.linspace(xmin, xmax, int((xmax - xmin) / 0.1 + 1))
                        ys = np.linspace(ymin, ymax, int((ymax - ymin) / 0.1 + 1))

                        xs, ys = np.meshgrid(xs, ys)

                        length = xs.shape[0] * xs.shape[1]

                        plot_states = np.vstack([np.ones(length) * obs[0],
                                         np.ones(length) * obs[1],
                                         np.ones(length) * obs[2],
                                         xs.flatten(),
                                         ys.flatten(),
                                         np.ones(length) * obs[5]]).T

                        processed_obs = self.preprocessor.tf_process_obses(plot_states)
                        value = self.policy_with_value.compute_value_net(processed_obs)
                        value = np.array(value).reshape(xs.shape[0], xs.shape[1])

                        self.env.render(xs=xs, ys=ys, value=value)
                    else:
                        self.env.render()

                reward_list.append(reward)
                action_list.append(action[0])

                # debuging
                # print(done, info['done_info'])

                if done:
                    break

                # if info['done_info'] == 'good_done': break
        else:
            while not done:
                processed_obs = self.preprocessor.tf_process_obses(obs)
                action = self.policy_with_value.compute_mode(processed_obs[np.newaxis, :])
                obs, reward, done, info = self.env.step(action.numpy()[0])
                reward_info_dict_list.append(info['reward_info'])
                if render: self.env.render()
                reward_list.append(reward)
        episode_return = sum(reward_list)
        episode_len = len(reward_list)
        info_dict = dict()
        info_dict.update(dict(episode_return=episode_return,
                              episode_len=episode_len))
        return info_dict

    def run_n_episode(self, n, mode):
        # list_of_return = []
        # list_of_len = []

        list_of_info_dict = []
        for _ in range(n):
            logger.info('logging {}-th episode'.format(_))
            info_dict = self.run_an_episode(mode, self.args.fixed_steps, self.args.eval_render,)
            list_of_info_dict.append(info_dict.copy())
        n_info_dict = dict()
        for key in list_of_info_dict[0].keys():
            info_key = list(map(lambda x: x[key], list_of_info_dict))
            mean_key = sum(info_key) / len(info_key)
            n_info_dict.update({key: mean_key})


        return n_info_dict




    def set_weights(self, weights):
        self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def run_evaluation(self, iteration,mode='train'):
        with self.eval_timer:
            self.iteration = iteration

            if mode =='train':
                self.draw_feasible_states(iteration,mode='value')
                self.draw_feasible_states(iteration,mode='action')

            n_info_dict = self.run_n_episode(self.args.num_eval_episode, mode)
            with self.writer.as_default():
                for key, val in n_info_dict.items():
                    self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)
                for key, val in self.get_stats().items():
                    self.tf.summary.scalar("evaluation/{}".format(key), val, step=self.iteration)
                self.writer.flush()
        if self.eval_times % self.args.eval_log_interval == 0:
            logger.info('Evaluator_info: {}, {}'.format(self.get_stats(), n_info_dict))
        self.eval_times += 1

    def compute_action_from_batch_obses(self, path):
        obses = np.load(path)
        preprocess_obs = self.preprocessor.np_process_obses(obses)
        action = self.policy_with_value.compute_mode(preprocess_obs)
        action_np = action.numpy()
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(range(action_np.shape[0]), action_np[:, 0])
        plt.show()
        a = 1

    def draw_feasible_states(self,n,mode='value'):
        fig = plt.figure(num='eval', figsize=(10, 5))

        plt.title("Feasible States")
        ax = plt.axes()
        ax.set_aspect('equal')

        xmin, xmax, ymin, ymax = (self.env.plot_option.xmin, self.env.plot_option.xmax,
                                  self.env.boundary_info.min_y, self.env.boundary_info.max_y)

        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        plt.axhline(y=self.env.boundary_info['max_y'], lw=2, color='k')
        plt.axhline(y=self.env.boundary_info['min_y'], lw=2, color='k')
        plt.axhline(y=0.5 * (self.env.constraint['min_y'] + self.env.constraint['max_y']), lw=1, ls='--', color='k')

        obstacle_x = self.env.obstacle_info.x
        obstacle_y = self.env.obstacle_info.y
        obstacle_width = self.env.obstacle_info.width
        obstacle_height = self.env.obstacle_info.height

        ax.add_patch(plt.Rectangle((obstacle_x, obstacle_y),
                                   obstacle_width,
                                   obstacle_height, edgecolor='black', facecolor='black'))



        xs = np.linspace(xmin,xmax,int((xmax-xmin)/0.1 +1))
        ys = np.linspace(ymin,ymax,int((ymax-ymin)/0.1 +1))

        xs,ys = np.meshgrid(xs,ys)

        length = xs.shape[0] * xs.shape[1]

        obs = np.vstack([np.ones(length) * self.env.expected_speed,
                         np.zeros(length),
                         np.zeros(length),
                        xs.flatten(),
                         ys.flatten(),
                         np.zeros(length)]).T



        processed_obs = self.preprocessor.tf_process_obses(obs)

        if mode == 'value':
            value = self.policy_with_value.compute_value_net(processed_obs)
            value = np.array(value).reshape(xs.shape[0],xs.shape[1])
        elif mode == 'action':
            value = self.policy_with_value.compute_mode(processed_obs)
            value = np.array(value)[:,1].reshape(xs.shape[0],xs.shape[1])

        fig_plot=ax.contourf(xs, ys, value, 100, linestyles=":", cmap='rainbow')
        plt.colorbar(fig_plot,orientation = 'horizontal')

        if not os.path.exists(os.path.join(self.args.model_dir,mode)):
            os.mkdir(os.path.join(self.args.model_dir,mode))
        fig.savefig(os.path.join(self.args.model_dir,mode,f'{mode}-{n}'))

        plt.close(fig=fig)




if __name__ == '__main__':
    atest_trained_model('./results/toyota3lane/experiment-2021-01-03-12-38-00/models',
                        './results/toyota3lane/experiment-2021-01-03-12-38-00/models', 100000)
