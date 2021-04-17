#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/9/1
# @Author  : Yang Guan (Tsinghua Univ.)
# @FileName: policy.py
# =====================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import PolynomialDecay

from model import MLPNet

NAME2MODELCLS = dict([('MLP', MLPNet), ])


class Policy4Toyota(tf.Module):
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors
    tf.config.experimental.set_visible_devices([], 'GPU')

    def __init__(self, args):
        super().__init__()
        self.args = args
        obs_dim, act_dim = self.args.obs_dim, self.args.act_dim
        n_hiddens, n_units, hidden_activation = self.args.num_hidden_layers, self.args.num_hidden_units, self.args.hidden_activation
        value_model_cls, policy_model_cls = NAME2MODELCLS[self.args.value_model_cls], \
                                            NAME2MODELCLS[self.args.policy_model_cls]
        self.policy = policy_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, act_dim * 2, name='policy',
                                       output_activation=self.args.policy_out_activation)
        policy_lr_schedule = PolynomialDecay(*self.args.policy_lr_schedule)
        self.policy_optimizer = self.tf.keras.optimizers.Adam(policy_lr_schedule, name='adam_opt')

        self.obj_v = value_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, 1, name='obj_v')
        self.con_v = value_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, 1, name='con_v')

        obj_value_lr_schedule = PolynomialDecay(*self.args.value_lr_schedule)
        self.obj_value_optimizer = self.tf.keras.optimizers.Adam(obj_value_lr_schedule, name='objv_adam_opt')

        con_value_lr_schedule = PolynomialDecay(*self.args.value_lr_schedule)
        self.con_value_optimizer = self.tf.keras.optimizers.Adam(con_value_lr_schedule, name='conv_adam_opt')

        self.models = (self.obj_v, self.con_v, self.policy,)
        self.optimizers = (self.obj_value_optimizer, self.con_value_optimizer, self.policy_optimizer)

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
        obj_v_len = len(self.obj_v.trainable_weights)
        con_v_len = len(self.con_v.trainable_weights)
        obj_v_grad, con_v_grad, policy_grad = grads[:obj_v_len], \
                                              grads[obj_v_len:obj_v_len + con_v_len], \
                                              grads[obj_v_len + con_v_len:]
        self.obj_value_optimizer.apply_gradients(zip(obj_v_grad, self.obj_v.trainable_weights))
        self.con_value_optimizer.apply_gradients(zip(con_v_grad, self.con_v.trainable_weights))
        self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))

    @tf.function
    def compute_mode(self, obs):
        logits = self.policy(obs)
        mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        return mean

    @tf.function
    def compute_action(self, obs):
        with self.tf.name_scope('compute_action') as scope:
            logits = self.policy(obs)
            assert self.args.deterministic_policy
            mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
            return mean, 0.


    @tf.function
    def compute_obj_v(self, obs):
        with self.tf.name_scope('compute_obj_v') as scope:
            return tf.squeeze(self.obj_v(obs), axis=1)

    @tf.function
    def compute_con_v(self, obs):
        with self.tf.name_scope('compute_con_v') as scope:
            return tf.squeeze(self.con_v(obs), axis=1)

class Policy4Feasible(tf.Module):
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    tfb = tfp.bijectors
    tf.config.experimental.set_visible_devices([], 'GPU')

    def __init__(self, args):
        super().__init__()
        self.args = args
        obs_dim, act_dim = self.args.obs_dim, self.args.act_dim
        n_hiddens, n_units, hidden_activation = self.args.num_hidden_layers, self.args.num_hidden_units, self.args.hidden_activation
        value_model_cls, policy_model_cls = NAME2MODELCLS[self.args.value_model_cls], \
                                            NAME2MODELCLS[self.args.policy_model_cls]
        self.policy = policy_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, act_dim * 2, name='policy',
                                       output_activation=self.args.policy_out_activation)
        policy_lr_schedule = PolynomialDecay(*self.args.policy_lr_schedule)
        self.policy_optimizer = self.tf.keras.optimizers.Adam(policy_lr_schedule, name='adam_opt')


        # TODO: 只留一个Value网络
        self.value_net=value_model_cls(obs_dim, n_hiddens, n_units, hidden_activation, 1, name='value_net')

        value_net_lr_schedule = PolynomialDecay(*self.args.value_lr_schedule)
        self.value_net_optimizer = self.tf.keras.optimizers.Adam(value_net_lr_schedule, name='value_net_adam_opt')

        self.models = (self.value_net, self.policy,)
        self.optimizers = (self.value_net_optimizer, self.policy_optimizer)

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
        value_net_len = len(self.value_net.trainable_weights)
        value_net_grad, policy_grad = grads[:value_net_len], grads[value_net_len:]
        self.value_net_optimizer.apply_gradients(zip(value_net_grad, self.value_net.trainable_weights))
        self.policy_optimizer.apply_gradients(zip(policy_grad, self.policy.trainable_weights))

    @tf.function
    def compute_mode(self, obs):
        logits = self.policy(obs)
        mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
        return mean

    @tf.function
    def compute_action(self, obs):
        with self.tf.name_scope('compute_action') as scope:
            logits = self.policy(obs)
            assert self.args.deterministic_policy
            mean, _ = self.tf.split(logits, num_or_size_splits=2, axis=-1)
            return mean, 0.


    @tf.function
    def compute_value_net(self, obs):
        with self.tf.name_scope('compute_value_net') as scope:
            return tf.squeeze(self.value_net(obs), axis=1)


if __name__ == '__main__':
    pass
