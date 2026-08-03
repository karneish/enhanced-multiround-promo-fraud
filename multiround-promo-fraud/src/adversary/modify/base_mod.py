import torch
import numpy as np

from numpy import random
from utils.utils_func import verPrint

###########################################
### BASE ADVERSARIAL MODIFICATION CLASS ###
###########################################
class BaseAdversarialMod():
    def __init__(self, verbose=0):
        pass

    def modify_seeds(self, graph, node_data, edge_data, seed_ids, modified_ids, **kwargs):
        return None