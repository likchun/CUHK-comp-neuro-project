import os
import numpy as np
from mylib import NeuroData


class EntropyFinder:

    def load_data(self, params, directory, Nstim, trial, t_range):
        self._dir = directory
        self._params = params
        self._Nstim = Nstim
        self._trial = trial
        self._t_range = t_range
        self._Astim = 2

        self.get_response()
        self.get_activity()
        self.get_Prob_response()
        self.get_Prob_condresp()
        self.get_H_response()
        self.get_H_condresp()
        self.get_mutual_information()

    def get_response(self):
        self.response = np.zeros((len(self._Nstim),len(self._trial))) # to store responses, [row: stimuli, col: trials]
        for j,tr in enumerate(self._trial):
            for i,Ns in enumerate(self._Nstim):
                nd = NeuroData(os.path.join(self._dir, "{0},{1},{2},{3}/[{4},0,{5},{6}]".format(*self._params, Ns, self._Astim, tr)))
                nd.retain_dynamics(*self._t_range)
                # get neurons that have never been stimulated, total = 500
                nd.apply_neuron_mask(list(range(100,200))+list(range(600,1000)))
                self.response[i][j] = len(nd.dynamics.spike_count[nd.dynamics.spike_count>0])
        self.xbins = np.arange(self.response.min(),self.response.max()+1,1)
        return self.response

    def get_activity(self):
        self.activity = np.mean(self.response, axis=1)
        return self.activity

    def get_Prob_response(self):
        self.P_response = np.histogram(self.response.flatten(), bins=self.xbins, density=True)[0]
        return self.P_response

    def get_Prob_condresp(self):
        self.P_condresp = np.array([np.histogram(self.response[i], bins=self.xbins, density=True)[0] for i in range(len(self._Nstim))])
        return self.P_condresp

    entropy = lambda self, Prob: -np.sum([p*np.log2(p) for p in Prob if p!=0])

    def get_H_response(self):
        self.H_response = self.entropy(self.P_response)
        return self.H_response

    def get_H_condresp(self):
        # assume that all stimuli have equal probability to appear, P(S)=constant
        self.H_condresp = np.mean([self.entropy(P) for P in self.P_condresp])
        return self.H_condresp

    def get_mutual_information(self):
        self.mutual_information = self.H_response - self.H_condresp
        return self.mutual_information
