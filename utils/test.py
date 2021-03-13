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

a='../results'
b='toyota3lane/2021-03-10-12-29-40/logs'
c=os.path.join(a,b)
print(c)
print(os.listdir(c))