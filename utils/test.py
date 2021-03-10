import logging


import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.INFO)

logging.debug(u"deb")
logging.info(u"inf")
logging.warning(u"warn")
logging.error(u"err")
logging.critical(u"cri")

import tensorflow as tf
a=tf.constant([1,2,3])
a=tf.expand_dims(a,0)
b=tf.constant([5,7,6])
b=tf.expand_dims(b,0)
c=tf.concat([a,b],0)

import numpy as np
e=a.numpy().ravel()
print(a)
print(e)
print( 1 in e)
print(1 in a)
print(c)