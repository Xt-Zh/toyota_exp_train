#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
from gym import spaces
from tensorflow.keras.optimizers.schedules import PolynomialDecay

from model import MLPNet

NAME2MODELCLS = dict([('MLP', MLPNet),])

class ActorCritic4Braking(tf.Module):
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], 'GPU')

    def __init__(self, args):
        super().__init__()
        self.args = args
        obs_dim, act_dim = self.args.obs_dim, self.args.act_dim
        n_hiddens, n_units, hidden_activation = self.args.num_hidden_layers, \
                                                self.args.num_hidden_units, \
                                                self.args.hidden_activation
        value_model_cls, policy_model_cls = NAME2MODELCLS[self.args.value_model_cls], \
                                            NAME2MODELCLS[self.args.policy_model_cls]

        # 定义actor
        self.actor = policy_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, act_dim * 2, name='actor',
                                       output_activation=self.args.policy_out_activation)
        actor_lr_schedule = PolynomialDecay(*self.args.policy_lr_schedule)
        self.actor_optimizer = self.tf.keras.optimizers.Adam(actor_lr_schedule, name='adam_opt')

        # 定义critic
        self.critic = policy_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, act_dim * 2, name='critic')
        critic_lr_schedule = PolynomialDecay(*self.args.value_lr_schedule)
        self.critic_optimizer = self.tf.keras.optimizers.Adam(critic_lr_schedule, name='adam_opt')

        # 组合起来
        self.models = (self.actor, self.critic)
        self.optimizers = (self.actor_optimizer,self.critic_optimizer)

    def save_weights(self, save_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.save(save_dir + '/ckpt_ite' + str(iteration))
    def load_weights(self, load_dir, iteration):
        model_pairs = [(model.name, model) for model in self.models]
        optimizer_pairs = [(optimizer._name, optimizer) for optimizer in self.optimizers]
        ckpt = self.tf.train.Checkpoint(**dict(model_pairs + optimizer_pairs))
        ckpt.restore(load_dir + '/ckpt_ite' + str(iteration) + '-1')
    def get_weights(self):
        return [model.get_weights() for model in self.models]
    def set_weights(self, weights):
        for i, weight in enumerate(weights):
            self.models[i].set_weights(weight)

    @tf.function
    def apply_gradients(self, iteration, grads):
        actor_len = len(self.actor.trainable_weights)
        critic_len = len(self.critic.trainable_weights)
        actor_grad, critic_grad = grads[:actor_len], grads[critic_len:]
        # self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_weights))
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_weights))

    @tf.function
    def compute_mode(self, obs):
        logits = self.actor(obs)
        mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1) #取前一半是为什么？
        return mean

    @tf.function
    def compute_action(self, obs):
        with self.tf.name_scope('compute_action') as scope:
            logits = self.actor(obs)
            assert self.args.deterministic_policy
            mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
            return mean, 0.

    def compute_value(self, obs):
        with tf.name_scope("compute_value") as scope:
            logits = self.critic(obs)
            mean, log_std = self.tf.split(logits, num_or_size_splits=2, axis=-1)
            return mean
