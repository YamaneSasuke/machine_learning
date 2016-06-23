# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 15:54:21 2016

@author: yamane
"""

import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers, utils
import chainer.functions as F