#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import numpy as np
from gym.envs.user_defined.EmerBrake.models import MyOriginBrakeModel

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

    def __init__(self, actor_critic_cls, args):
        self.args = args

        self.actor_critic = actor_critic_cls(self.args)

        self.batch_data = {}
        self.all_data = {}
        self.M = self.args.M
        self.num_rollout_list_for_policy_update = self.args.num_rollout_list_for_policy_update

        self.model = MyOriginBrakeModel()
        self.preprocessor = Preprocessor((self.args.obs_dim,), self.args.obs_preprocess_type,
                                         self.args.reward_preprocess_type,
                                         self.args.obs_scale, self.args.reward_scale, self.args.reward_shift,
                                         gamma=self.args.gamma)
        self.grad_timer = TimerStat()
        self.stats = {}
        self.info_for_buffer = {}

    def get_stats(self):
        return self.stats

    def get_info_for_buffer(self):
        return self.info_for_buffer

    def get_batch_data(self, batch_data):
        self.batch_data = {'batch_obs': batch_data[0].astype(np.float32), }

    def get_weights(self):
        return self.actor_critic.get_weights()

    def set_weights(self, weights):
        return self.actor_critic.set_weights(weights)

    def set_ppc_params(self, params):
        self.preprocessor.set_params(params)

    def model_rollout_for_update(self, start_obses):
        self.model.reset(start_obses)
        obses = start_obses
        critic_target = self.tf.zeros([start_obses.shape[0], 1])
        # safe_info = np.ones([start_obses.shape[0], 1])
        gamma = 0.9
        discount = 1.0
        for step in range(self.num_rollout_list_for_policy_update[0]):
            # processed_obses = self.preprocessor.tf_process_obses(obses)
            actions, _ = self.actor_critic.compute_action(obses)  # -1<=actions<=1
            obses, rewards = self.model.rollout_out(actions)
            critic_target += discount * rewards
            # assert np.all(rewards>=0)
            discount *= gamma
        # 在rollout的过程中和actor_loss的计算中都不采用罚函数吸收态，直接计算即可
        critic_target += discount * self.actor_critic.compute_value(obses)


        actor_loss = self.tf.reduce_mean(critic_target)  # 目的是最小化\sum l(x,u)+V

        # 在计算critic_loss的时候将那些已经到达不安全状态的情况设置为吸收态值
        safe_info = self.tf.expand_dims(self.model.judge_safety(), axis=1)  # 0为不安全，1为安全
        critic_target = self.tf.where(safe_info == 0,
                                      self.tf.ones_like(critic_target) * self.model.reward_absorb,
                                      critic_target)
        critic_loss = self.tf.stop_gradient(critic_target) - self.actor_critic.compute_value(start_obses)
        critic_loss = self.tf.reduce_mean(self.tf.square(critic_loss) / 2)

        return actor_loss, critic_loss

    @tf.function
    def forward_and_backward(self, mb_obs, ite):
        with self.tf.GradientTape(persistent=True) as tape:
            actor_loss, value_loss = self.model_rollout_for_update(mb_obs)

        with self.tf.name_scope('policy_gradient') as scope:
            actor_grad = tape.gradient(actor_loss, self.actor_critic.actor.trainable_weights)
            critic_grad = tape.gradient(value_loss, self.actor_critic.critic.trainable_weights)

        return actor_grad, critic_grad, actor_loss, value_loss

    def compute_gradient(self, samples, rb, indexs, iteration):
        self.get_batch_data(samples)
        mb_obs = self.tf.constant(self.batch_data['batch_obs'])
        iteration = self.tf.convert_to_tensor(iteration, self.tf.int32)

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
