#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : z
# @FileName: ampc.py
# =====================================

import logging

import numpy as np
from gym.envs.user_defined.EmerBrake.models import EmBrakeModel

from preprocessor import Preprocessor
from utils.misc import TimerStat

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONSTRAINTS_CLIP_MINUS = -0.1


class MyBrakingLearner(object):
    import tensorflow as tf
    tf.config.optimizer.set_experimental_options({'constant_folding': True,
                                                  'arithmetic_optimization': True,
                                                  'dependency_optimization': True,
                                                  'loop_optimization': True,
                                                  'function_optimization': True,
                                                  })

    def __init__(self, policy_cls, value_cls, args):
        self.args = args
        self.policy_with_value = policy_cls(self.args)
        self.value_function = value_cls(self.args)
        self.batch_data = {}
        self.all_data = {}
        self.M = self.args.M
        self.num_rollout_list_for_policy_update = self.args.num_rollout_list_for_policy_update

        self.model = EmBrakeModel()
        self.preprocessor = Preprocessor((self.args.obs_dim,), self.args.obs_preprocess_type,
                                         self.args.reward_preprocess_type,
                                         self.args.obs_scale, self.args.reward_scale, self.args.reward_shift,
                                         gamma=self.args.gamma)
        self.grad_timer = TimerStat()
        self.stats = {}
        self.info_for_buffer = {}
        # self.constraint_total_dim = args.num_rollout_list_for_policy_update[0] * self.model.constraints_num

    def get_stats(self):
        return self.stats

    def get_info_for_buffer(self):
        return self.info_for_buffer

    def get_batch_data(self, batch_data):
        self.batch_data = {'batch_obs': batch_data[0].astype(np.float32),
                           'batch_actions': batch_data[1].astype(np.float32),
                           'batch_rewards': batch_data[2].astype(np.float32),
                           'batch_obs_tp1': batch_data[3].astype(np.float32),
                           'batch_dones': batch_data[4].astype(np.float32)
                           }

    def get_weights(self):
        return self.policy_with_value.get_weights()

    def set_weights(self, weights):
        return self.policy_with_value.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def model_rollout_for_update(self, start_obses):
        self.model.reset(start_obses)
        obses = start_obses
        reward_list = []
        for step in range(self.num_rollout_list_for_policy_update[0]):
            processed_obses = self.preprocessor.tf_process_obses(obses)
            actions = self.policy_with_value.compute_action(processed_obses)
            values = self.value_function.compute_action(processed_obses)
            obses, rewards = self.model.rollout_out(actions)
            reward_list.append(rewards)
        reward_list.append(self.value_function(obses))
        total_reward = self.tf.reduce_sum(reward_list)

        return total_reward

    def forward_and_backward(self, mb_obs, ite):
        with self.tf.GradientTape(persistent=True) as tape:
            actor_loss, value_loss = self.model_rollout_for_update(mb_obs)

        with self.tf.name_scope('policy_gradient') as scope:
            actor_grad = tape.gradient(actor_loss, self.policy_with_value.policy.trainable_weights)
            critic_grad = tape.gradient(value_loss, self.value_function.trainable_weights)

        return actor_grad, critic_grad, actor_loss, value_loss

    def compute_gradient(self, samples, rb, indexs, iteration):
        self.get_batch_data(samples, rb, indexs)
        mb_obs = self.tf.constant(self.batch_data['batch_obs'])
        iteration = self.tf.convert_to_tensor(iteration, self.tf.int32)
        # mb_ref_index = self.tf.constant(self.batch_data['batch_ref_index'], self.tf.int32)

        with self.grad_timer:
            actor_grad, critic_grad, actor_loss, value_loss = self.forward_and_backward(mb_obs, iteration)

            actor_grad, actor_grad_norm = self.tf.clip_by_global_norm(actor_grad, self.args.gradient_clip_norm)
            critic_grad, critic_grad_norm = self.tf.clip_by_global_norm(critic_grad, self.args.gradient_clip_norm)

        self.stats.update(dict(
            iteration=iteration,
            grad_time=self.grad_timer.mean,
            actor_loss=actor_loss.numpy(),
            value_loss=value_loss.numpy(),
            actor_grad_norm=actor_grad_norm.numpy(),
            critic_grad_norm=critic_grad_norm.numpy()
        ))

        grads = actor_grad + critic_grad

        return list(map(lambda x: x.numpy(), grads))


if __name__ == '__main__':
    pass
