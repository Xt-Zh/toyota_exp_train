import numpy as np
import tensorflow as tf
a=np.arange(5)[:,None]
a=tf.convert_to_tensor(a)
print(a)

b=np.array([[0,1,4,0,0]]).T
b=tf.convert_to_tensor(b)
print(b)
print(a==b)
print(a<3)



c=(a<=1)
d=(a>=3)
print(c)
print(d)
print(c|d)