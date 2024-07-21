import os, sys, csv
sys.path.insert(1, os.path.join(os.path.abspath(os.path.join(sys.path[0], os.pardir)), ".."))

from matplotlib import pyplot as plt
import numpy as np

from mylib import NeuroData, qgraph
from figure_style import *
import data_path