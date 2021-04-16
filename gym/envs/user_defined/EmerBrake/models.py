import numpy as np
import tensorflow as tf
from tensorflow import logical_and


class EmBrakeModel(object):
    def __init__(self):
        self.constraints_num = 1

    def rollout_out(self, actions):
        with tf.name_scope('model_step') as scope:
            self.actions = self._action_transformation_for_end2end(actions)
            rewards, constraints = self.compute_rewards(self.obses, self.actions)
            self.obses = self.f_xu(self.obses, self.actions)
            # self.reward_info.update({'final_rew': rewards.numpy()[0]})

            return self.obses, rewards, constraints

    def compute_rewards(self, obses, actions):
        # rewards = -0.01 * tf.square(actions[:, 0]) # tf.square(obses[:, 0]) + tf.square(obses[:, 1]) +
        rewards = -0.01 * tf.square(obses[:, 1] - 5.0)  # r=-0.01(v-5)^2
        constraints = -obses[:, 0]
        return rewards, constraints

    def _action_transformation_for_end2end(self, actions):
        clipped_actions = tf.clip_by_value(actions, -1.05, 1.05)
        acc = 5.0 * clipped_actions
        return acc

    def f_xu(self, x, u, frequency=10.0):
        d, v = tf.cast(x[:, 0], dtype=tf.float32), tf.cast(x[:, 1], dtype=tf.float32)
        a = tf.cast(u[:, 0], dtype=tf.float32)
        frequency = tf.convert_to_tensor(frequency)
        next_state = [d - 1 / frequency * v, v + 1 / frequency * a]
        return tf.stack(next_state, 1)

    def reset(self, obses):  # input are all tensors
        self.obses = obses
        self.actions = None
        self.reward_info = None


# 这里有两个BrakeModel,第一个可能有些问题，第二个是当时跑出比较好效果的那个model
class MyBrakeModel(object):
    def __init__(self):
        self.constraints_num = 1
        self.reward_absorb = 10

    def judge_safety(self, new_state_value):
        # 在评估过程中，满足其中任意一种都被认为是不安全状态：
        # 1. 违反真实世界中的约束条件（到障碍物的距离小于0）
        # 2. 该点的价值函数值大于10（正常状况下不可能大于，因此我们知道它到了不可行区域）
        self.safe_info = tf.where(self.obses[:, 0] < 0, tf.zeros_like(self.obses[:, 0]),
                                  tf.ones_like(self.obses[:, 0]))
        self.safe_info = self.safe_info * tf.where(new_state_value[:, 0] >= 1.0 * self.reward_absorb,
                                                   tf.zeros_like(self.obses[:, 0]),
                                                   tf.ones_like(self.obses[:, 0]))

        return self.safe_info

    def judge_true_safety(self):
        self.safe_info = tf.where(self.obses[:, 0] < 0, tf.zeros_like(self.obses[:, 0]),
                                  tf.ones_like(self.obses[:, 0]))
        return self.safe_info

    def rollout_out(self, actions):
        with tf.name_scope('model_step') as scope:
            self.actions = self._action_transformation_for_end2end(actions)  # 传入的action是-1到1的，现在拉到-5到5
            rewards = self.compute_rewards(self.obses, self.actions)
            self.obses = self.f_xu(self.obses, self.actions)

            return self.obses, rewards

    def compute_rewards(self, obses, actions):
        rewards = 0.01 * tf.square(actions)
        return rewards

    def _action_transformation_for_end2end(self, actions):
        clipped_actions = tf.clip_by_value(actions, -1.05, 1.05)
        acc = 5.0 * clipped_actions
        return acc

    def f_xu(self, x, u, frequency=10.0):
        d, v = tf.cast(x[:, 0], dtype=tf.float32), tf.cast(x[:, 1], dtype=tf.float32)
        a = tf.cast(u[:, 0], dtype=tf.float32)
        frequency = tf.convert_to_tensor(frequency)
        next_state = [d - 1 / frequency * v, v + 1 / frequency * a]
        return tf.stack(next_state, 1)

    def reset(self, obses):  # input are all tensors
        self.obses = obses
        self.judge_true_safety()
        self.actions = None
        self.reward_info = None


# 用这个Model跑出了比较好的效果。
class MyOriginBrakeModel(object):
    def __init__(self):
        self.constraints_num = 1
        self.reward_absorb = 10

    def judge_safety(self):
        self.safe_info = tf.where(self.obses[:, 0] < 0, tf.zeros_like(self.obses[:, 0]),
                                  tf.ones_like(self.obses[:, 0]))
        return self.safe_info

    def rollout_out(self, actions):
        with tf.name_scope('model_step') as scope:
            self.actions = self._action_transformation_for_end2end(actions)  # 传入的action是-1到1的，现在拉到-5到5
            rewards = self.compute_rewards(self.obses, self.actions)
            self.obses = self.f_xu(self.obses, self.actions)

            return self.obses, rewards

    def compute_rewards(self, obses, actions):
        rewards = 0.01 * tf.square(actions)
        return rewards

    def _action_transformation_for_end2end(self, actions):
        clipped_actions = tf.clip_by_value(actions, -1.05, 1.05)
        acc = 5.0 * clipped_actions
        return acc

    def f_xu(self, x, u, frequency=10.0):
        d, v = tf.cast(x[:, 0], dtype=tf.float32), tf.cast(x[:, 1], dtype=tf.float32)
        a = tf.cast(u[:, 0], dtype=tf.float32)
        frequency = tf.convert_to_tensor(frequency)
        next_state = [d - 1 / frequency * v, v + 1 / frequency * a]
        return tf.stack(next_state, 1)

    def reset(self, obses):  # input are all tensors
        self.obses = obses
        self.judge_safety()
        self.actions = None
        self.reward_info = None
