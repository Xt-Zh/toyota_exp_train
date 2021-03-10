# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# logging.basicConfig(level=logging.INFO)
#
# logging.debug(u"deb")
# logging.info(u"inf")
# logging.warning(u"warn")
# logging.error(u"err")
# logging.critical(u"cri")

# 屏蔽tensorflow输出的log信息
# 注意：代码在import tensorflow之前
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
print("python的版本信息：",sys.version)

import tensorflow as tf

'''
验证GPU相对于CPU,在并行计算优势明显
'''
n=80000000

# 创建在 CPU 环境上运算的 2 个矩阵
with tf.device('/cpu:0'):
    cpu_a = tf.random.normal([1, n])
    cpu_b = tf.random.normal([n, 1])
    print(cpu_a.device, cpu_b.device)

# 创建使用 GPU 环境运算的 2 个矩阵
with tf.device('/gpu:0'):
    gpu_a = tf.random.normal([1, n])
    gpu_b = tf.random.normal([n, 1])
    print(gpu_a.device, gpu_b.device)

import timeit

def cpu_run(): # CPU 运算函数
    with tf.device('/cpu:0'):
        c = tf.matmul(cpu_a, cpu_b)
    return c

def gpu_run():# GPU 运算函数
    with tf.device('/gpu:0'):
        c = tf.matmul(gpu_a, gpu_b)
    return c

# 第一次计算需要热身，避免将初始化时间结算在内
cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print('首先计算10次（含热身环境）的平均时间,CPU计算消耗时间：%.3fms,GPU计算消耗时间：%.3fms!'%(cpu_time*1000, gpu_time*1000) )

#正式计算10次，取平均时间
cpu1_time = timeit.timeit(cpu_run, number=30)
gpu1_time = timeit.timeit(gpu_run, number=30)

print('正式计算10次的平均时间,CPU计算消耗时间：%.3fms,GPU计算消耗时间：%.3fms!'%(cpu1_time*1000, gpu1_time*1000))