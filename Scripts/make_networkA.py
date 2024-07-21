import pandas as pd
import numpy as np
import random


N = 1000 # number of neurons
p = 0.01 # connection probability
N_ei_ratio = 4 # neuron EI ratio
kin_ei_ratio = 4 # in-degree EI ratio = (num of EXC presynaptic neuron)/(num of INH presynaptic neuron)
rng_seed = 1


np.random.seed(rng_seed)
random.seed(rng_seed)
neurons = np.arange(N,dtype=int)
neutype = np.zeros(N,dtype=int)+1
neutype[:int(N/(1+N_ei_ratio))] = -1
adjmat = np.zeros((N,N))

def get_data_frame():
    exc_pool = neurons[int(N/(1+N_ei_ratio)):]
    inh_pool = neurons[:int(N/(1+N_ei_ratio))]
    draw_exc = int(N*p*kin_ei_ratio/(1+kin_ei_ratio))
    draw_inh = int(N*p/(1+kin_ei_ratio))
    # choose EXC/INH inputs (controlled by "kin_ei_ratio") from EXC/INH neurons (controlled by "N_ei_ratio")
    inh_presyn_neuron_ind = [np.random.choice(np.delete(inh_pool, n), draw_inh, replace=False) if len(inh_pool)>n else np.random.choice(inh_pool, draw_inh, replace=False) for n in range(N)]
    exc_presyn_neuron_ind = [np.random.choice(np.delete(exc_pool, n-len(inh_pool)), draw_exc, replace=False) if len(inh_pool)<n+1 else np.random.choice(exc_pool, draw_exc, replace=False) for n in range(N)]
    return pd.DataFrame(
        data=np.array([neurons, neutype, exc_presyn_neuron_ind, inh_presyn_neuron_ind], dtype=object).T,
        columns=["neuron_index", "neuron_type", "exc_presyn_neuron_ind", "inh_presyn_neuron_ind"])

df = get_data_frame()

for i in range(N):
    for j_e in df.loc[i]["exc_presyn_neuron_ind"]: adjmat[i][j_e] = +1
    for j_i in df.loc[i]["inh_presyn_neuron_ind"]: adjmat[i][j_i] = -1


# The resulting adjacency matrix result is:
adjmat # <----
# ^^^^