import os, sys, csv
sys.path.insert(1, os.path.join(os.path.abspath(os.path.join(sys.path[0], os.pardir)), ".."))

from matplotlib import pyplot as plt
import numpy as np

from mylib import NeuroData, qgraph
from figure_style import *
import data_path


class Stimuli:

    def __init__(self, S, data_dir=data_path.stim_resp):
        self._num_neuron = 1000
        self._max_num_stim = 500
        # number of neurons receive stimulation
        self.S = S
        # (list of indices) indices of the group of neurons that never receive any stimulations
        self.obsvgp_ind = list(range(100,200))+list(range(600,1000))
        # (list of indices) indices of the group of neurons that can receive stimulations, in this group, some neurons may not receive stimulations, see `self.stim_ON_inds` and `self.stimOFF_inds` below
        self.stimgp_ind = list(range(100))+list(range(200,600))
        # (list of list of indices) indices of the neurons that actually receive stimulation (must be within `self.stimgp_ind`), one list for each element in `S`
        self.stim_ON_inds = [self.read_nidx(os.path.join(data_dir, "Nstim/Nstim_{}_0.txt".format(s))) for s in S]
        # (list of list of indices) indices of the neurons that can receive no stimulation (must be within `self.stimgp_ind`), one list for each element in `S`
        self.stimOFF_inds = [np.setdiff1d(self.stimgp_ind, ind) for ind in self.stim_ON_inds]
        # (list of list of indices) sorting of neuron indices used for raster plots, one list for each element in `S`
        self.ind_sortings = [np.concatenate([stim_ON_ind, stimOFF_ind, self.obsvgp_ind]) for stim_ON_ind,stimOFF_ind in zip(self.stim_ON_inds,self.stimOFF_inds)]
        self.ind_sortings_colors = [["darkorange" for _ in range(s)]+["k" for _ in range(self._max_num_stim-s)]+["darkgreen" for _ in range(len(self.obsvgp_ind))] for s in self.S]

    def read_nidx(self, filename):
        return np.array(list(csv.reader(open(filename, "r"), delimiter=" "))).flatten().astype(int)
