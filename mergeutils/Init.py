import os
import random
import tensorflow as tf

def set_seeds(seed:int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def set_proj():
    os.environ['PROJ_LIB'] = './mergeutils/proj'