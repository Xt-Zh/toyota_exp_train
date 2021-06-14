#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: ampc.py
# =====================================

import logging

import numpy as np

from gym.envs.user_defined.ObstacleEnv.CarModel import VehicleDynamics
from preprocessor import Preprocessor
from utils.misc import TimerStat, args2envkwargs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class FeasibleLearner(object):
    import tensorflow as tf
    tf.config.optimizer.set_experimental_options({'constant_folding': True,
                                                  'arithmetic_optimization': True,
                                                  'dependency_optimization': True,
                                                  'loop_optimization': True,
                                                  'function_optimization': True,
                                                  })

    def __init__(self, policy_cls, args):
        self.args = args
        self.policy_with_value = policy_cls(self.args)
        self.batch_data = {}
        self.all_data = {}
        self.M = self.args.M
        self.num_rollout_list_for_policy_update = self.args.num_rollout_list_for_policy_update

        self.model = VehicleDynamics(bs=self.args.replay_batch_size)
        self.preprocessor = Preprocessor((self.args.obs_dim,),
                                         self.args.obs_preprocess_type,
                                         self.args.reward_preprocess_type,
                                         self.args.obs_scale,
                                         self.args.reward_scale,
                                         self.args.reward_shift,
                                         gamma=self.args.gamma)
        self.grad_timer = TimerStat()
        self.stats = {}
        self.info_for_buffer = {}
        self.test_min = 0

    def get_stats(self):
        return self.stats

    def get_info_for_buffer(self):
        return self.info_for_buffer

    def get_batch_data(self, batch_data, rb, indexes):
        self.batch_data = {'batch_obs': batch_data[0].astype(np.float32),
                           'batch_actions': batch_data[1].astype(np.float32),
                           'batch_rewards': batch_data[2].astype(np.float32),
                           'batch_obs_tp1': batch_data[3].astype(np.float32),
                           'batch_dones': batch_data[4].astype(np.float32),
                           # 'batch_ref_index': batch_data[5].astype(np.int32)
                           }

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def model_rollout_for_update(self, start_obses, ite):
        start_obses = self.tf.tile(start_obses, [self.M, 1])
        self.model.reset(start_obses)

        rewards_sum = self.tf.zeros((start_obses.shape[0],))
        obses = start_obses
        processed_obses = self.preprocessor.tf_process_obses(obses)
        # print(processed_obses.shape)
        value_pred = self.policy_with_value.compute_value_net(processed_obses)

        gamma = 0.95
        discount = 1.0
        # print(obses)

        safe_info = self.tf.ones_like(processed_obses[:, 0],dtype=self.tf.bool)

        for i in range(self.num_rollout_list_for_policy_update[0]):
            processed_obses = self.preprocessor.tf_process_obses(obses)
            actions, _ = self.policy_with_value.compute_action(processed_obses)
            obses, rewards, now_safe_info = self.model.rollout_out(actions)
            # print(rewards)
            safe_info = safe_info & now_safe_info
            # print(i,safe_info[:10],now_safe_info[:10])
            rewards_sum += discount * self.preprocessor.tf_process_rewards(rewards)
            discount *= gamma

        # print(obses)

        # print(rewards_sum)
        # print(safe_info)

        now_state = self.preprocessor.tf_process_obses(obses)
        now_state_value = self.policy_with_value.compute_value_net(now_state)


        # print(now_state_value)

        # print(self.tf.reduce_max(rewards_sum))
        # print(obses.shape)
        # print(processed_obses.shape)

        # index_violate=self.tf.argmin(safe_info)
        # print('index',int(index_violate))
        # if index_violate==0: # 全部安全
        #     pass
        #     # print('now_state_value',float(value_pred[0:1]))
        # else: # 第index_violate项不安全
        #     print('\033[32m now_violate_value \033[0m',float(value_pred[index_violate:index_violate+1]))


        # policy loss
        policy_loss = self.tf.reduce_mean(rewards_sum + now_state_value * discount)
        # print(policy_loss)

        # print('reward_sum')
        # print(rewards_sum)

        rewards_absorb = self.tf.where(safe_info == True, self.tf.ones_like(rewards_sum) * (50), rewards_sum)
        target = self.tf.stop_gradient(rewards_absorb + now_state_value * discount)

        # print(target)

        # target = self.tf.where(safe_info == 0,
        #                        self.tf.ones_like(rewards_sum) * 50,
        #                        -rewards_sum+ now_state_value * discount)
        # target = self.tf.stop_gradient(-rewards_absorb )

        # if index_violate == 0:
        #     pass
        #     # print('\n reward_absorb',float(rewards_absorb[0]),
        #     #   '\n now_state_value',float(now_state_value[0]),
        #     #   '\n target',float(target[0]))
        # else:
        #     print('\033[32m \n violate reward_absorb \033[0m', float(rewards_absorb[0]),
        #           '\033[32m \n violate now_state_value \033[0m', float(now_state_value[0]),
        #           '\033[32m \n violate target \033[0m', float(target[0]))

        # value的目的是学习一个正值
        value_loss = self.tf.reduce_mean(self.tf.square(value_pred - self.tf.clip_by_value(target, 0, 10000)))

        # print(value_loss)


        return value_loss, policy_loss

    # @tf.function
    def forward_and_backward(self, mb_obs, ite):
        with self.tf.GradientTape(persistent=True) as tape:
            value_loss, policy_loss = self.model_rollout_for_update(mb_obs, ite)

        with self.tf.name_scope('policy_gradient') as scope:
            policy_grad = tape.gradient(policy_loss, self.policy_with_value.policy.trainable_weights)
        with self.tf.name_scope('value_gradient') as scope:
            value_grad = tape.gradient(value_loss, self.policy_with_value.value_net.trainable_weights)

        return policy_grad, value_grad, policy_loss, value_loss

    def export_graph(self, writer):
        mb_obs = self.batch_data['batch_obs']
        self.tf.summary.trace_on(graph=True, profiler=False)
        self.forward_and_backward(mb_obs, self.tf.convert_to_tensor(0, self.tf.int32),
                                  self.tf.zeros((len(mb_obs),), dtype=self.tf.int32))
        with writer.as_default():
            self.tf.summary.trace_export(name="policy_forward_and_backward", step=0)

    def compute_gradient(self, samples, rb, indexs, iteration):
        self.get_batch_data(samples, rb, indexs)
        mb_obs = self.tf.constant(self.batch_data['batch_obs'])
        iteration = self.tf.convert_to_tensor(iteration, self.tf.int32)
        # mb_ref_index = self.tf.constant(self.batch_data['batch_ref_index'], self.tf.int32)

        with self.grad_timer:
            policy_grad, value_grad, policy_loss, value_loss = self.forward_and_backward(mb_obs, iteration)

            policy_grad, policy_grad_norm = self.tf.clip_by_global_norm(policy_grad, self.args.gradient_clip_norm)
            value_grad, value_grad_norm = self.tf.clip_by_global_norm(value_grad, self.args.gradient_clip_norm)

        self.stats.update(dict(
            iteration=iteration,
            grad_time=self.grad_timer.mean,
            value_loss=value_loss.numpy(),
            policy_loss=policy_loss.numpy(),
            policy_grads_norm=policy_grad_norm.numpy(),
            value_grad_norm=value_grad_norm.numpy(),
        ))

        grads = value_grad + policy_grad

        return list(map(lambda x: x.numpy(), grads))


if __name__ == '__main__':
    pass
