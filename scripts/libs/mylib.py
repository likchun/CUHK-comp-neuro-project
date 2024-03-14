"""
MyLib
-----
Tools & templates

Last update: 10 Mar, 2024 (pm)
"""

import os
import csv
import math
import json
import yaml
import itertools
import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from powerlaw import Fit
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy import signal
from scipy import stats

MAXVAL_uint16_t = 65535
MAXVAL_uint8_t = 255


t_range_ = lambda duration_ms, stepsize_ms, transient_ms=0: np.arange(int(duration_ms/stepsize_ms)+1)[int(transient_ms/stepsize_ms):]*stepsize_ms
"""time scale: milliseconds (ms)"""

def load_spike_steps(filepath:str, delimiter='\t', incspkc=True):
    if incspkc: return np.array([np.array(x,dtype=int)[1:] for x in csv.reader(open(filepath,'r'),delimiter=delimiter)],dtype=object)
    else: return np.array([np.array(x,dtype=int) for x in csv.reader(open(filepath,'r'),delimiter=delimiter)],dtype=object)

def load_spike_steps2(filepath:str, dtype=None):
    try:
        with open(filepath, "r") as file:
            data = [np.fromstring(line.rstrip(), dtype=dtype, sep=" ") for line in file.readlines()]
        return data
    except FileNotFoundError: raise

def save_spike_steps(spike_steps:np.ndarray, filepath:str, delimiter='\t', incspkc=True):
    with open(filepath,'w') as f:
        if incspkc: [[f.write("{}".format(len(spks))), [f.write("{}{}".format(delimiter, s)) for s in spks], f.write('\n')] for spks in spike_steps]
        else: [[f.write("{}".format(spks[0])), [f.write("{}{}".format(delimiter, s)) for s in spks[1:]], f.write('\n')] for spks in spike_steps]

def load_time_series(filepath:str, num_neuron:int, dtype=np.float32):
    time_series = np.fromfile(filepath,dtype=dtype)
    return time_series.reshape((int(time_series.shape[0]/num_neuron),num_neuron)).T

def load_stimulus(filepath:str, returnInfo=False):
    """Return a tuple with three elements:
    - [0] indices of stimulated neurons, np.array
    - [1] stimulus time series, np.array
    - [2] (optional) stimulus info, np.array, contains:
        - [0] time step size (ms)
        - [1] stimulated duration (ms)
        - [2] non-stimulated duration before stimulus (ms)
        - [3] non-stimulated duration after stimulus (ms)
        - [4] stimulus name/type
        - [5] driving frequency f_d (Hz)
        - [6] stimulus amplitude A
        - [7] stimulus bias B"""
    content = np.array(list(csv.reader(open(filepath,'r',newline=''),delimiter='\t')),dtype=object)
    stim_info = content[0]
    for i in [0,1,2,3,5,6,7]: stim_info[i] = float(stim_info[i])
    if returnInfo: return np.array(content[1],dtype=int), np.array(content[2],dtype=float), stim_info
    else: return np.array(content[1],dtype=int), np.array(content[2],dtype=float)


def get_spike_count(spike_steps:np.ndarray):
    return np.array([x.size for x in spike_steps])

def get_mean_spike_rate(spike_steps:np.ndarray, duration_ms:float):
    """unit: Hz"""
    return np.array(np.array([x.size for x in spike_steps]))/duration_ms*1000

def get_average_spike_rate_time_histogram(spike_steps:np.ndarray, stepsize_ms:float, duration_ms:float, binsize_ms:float):
    """Return a tuple:
    - mid-point of time bins
    - network average spike rate (number of spikes in time bin divided by bin size and number of neurons)\n
    units: s, Hz"""
    num_bin = int(duration_ms//binsize_ms)+1
    numspikes, binedges = np.histogram(np.hstack(spike_steps*stepsize_ms),np.linspace(0,binsize_ms*num_bin,num_bin))
    return ((binedges[1:]+binedges[:-1])/2/1000)[:-1], (numspikes/(binedges[1:]-binedges[:-1])*1000/spike_steps.size)[:-1] # remove the last bin, which cannot be fitted into the time interval for the given bin size

def get_average_spike_rate_time_curve(spike_steps:np.ndarray, stepsize_ms:float, duration_ms:float, kernelstd_ms:float):
    """Return a tuple:
    - time point
    - network average spike rate (Gaussian kernel filtered collapsed spike train)\n
    units: s, Hz"""
    spike_train = np.zeros((spike_steps.size,int(duration_ms/stepsize_ms)+1),dtype=int) # +1 to include the initial time step t0
    for i,s in enumerate(spike_train): s[spike_steps[i]] = 1
    return t_range_(duration_ms,stepsize_ms)/1000, ndimage.gaussian_filter1d(spike_train.mean(axis=0),kernelstd_ms/stepsize_ms)/(stepsize_ms/1000)

def get_num_spiked_neuron_time_histogram(spike_steps:np.ndarray, stepsize_ms:float, duration_ms:float, binsize_ms:float):
    """Return a tuple:
    - mid-point of time bins
    - number of neurons that spike within each time bin\n
    units: s, number of neurons"""
    spike_steps = spike_steps*stepsize_ms
    num_bin = int(duration_ms//binsize_ms)+1
    bins = np.linspace(0,binsize_ms*num_bin,num_bin)
    num_spk_in_bin = np.array([(np.histogram(s,bins=bins)[0]).clip(max=1) for s in spike_steps])
    return ((bins[1:]+bins[:-1])/2/1000)[:-1], np.sum(num_spk_in_bin,axis=0)[:-1]

def get_interspike_intervals(spike_steps:np.ndarray, stepsize_ms:float):
    """unit: s"""
    return np.array([np.diff(x)/1000 for x in spike_steps*stepsize_ms],dtype=object)

def get_network_spike_times(spike_steps:np.ndarray, stepsize_ms:float):
    return np.unique(np.hstack(spike_steps))*stepsize_ms

def get_interevent_intervals(spike_steps:np.ndarray, stepsize_ms:float):
    return np.diff(get_network_spike_times(spike_steps, stepsize_ms))

def get_interevent_intervals_insteps(spike_steps:np.ndarray):
    return np.diff(np.unique(np.hstack(spike_steps)))

def get_spike_train(spike_steps:np.ndarray, num_steps:int):
    spike_train = np.zeros((spike_steps.size,num_steps+1),dtype=int) # +1 to include the initial time step t0
    for i,s in enumerate(spike_train): s[spike_steps[i]] = 1
    return spike_train[:,1:]

def trim_spike_steps(spike_steps:np.ndarray, start_t_ms:float, end_t_ms:float, stepsize_ms:float):
    """Return spike steps"""
    return np.array([np.array(ss[np.where((start_t_ms/stepsize_ms < np.array(ss)) & (np.array(ss) < end_t_ms/stepsize_ms))], dtype=int) for ss in spike_steps], dtype=object)


def get_C_measure(binsize_ms:float, spike_steps:np.ndarray, stepsize_ms:float, duration_ms:float):
    boolmask = [False if s.size!=0 else True for s in spike_steps]
    num_bin = int(duration_ms//binsize_ms)+1
    bins = np.linspace(0,binsize_ms*num_bin,num_bin)
    numspike, binedge = np.histogram(np.hstack(spike_steps*stepsize_ms), bins)
    avgfr = (numspike/(binedge[1:]-binedge[:-1])*1000/spike_steps.size)[:-1]
    numspikes = np.array([np.histogram(np.hstack(spike_steps[i]*stepsize_ms), bins)[0] if spike_steps[i].size!=0 else np.zeros(numspike.size) for i in range(spike_steps.size)])
    fr = (numspikes/(binedge[1:]-binedge[:-1])*1000)[:,:-1]

    m1, m2 = [fr[i].mean() for i in range(spike_steps.size)], avgfr.mean()
    return np.array([( np.dot(avgfr,fr[i]) / avgfr.size - m1[i]*m2 ) / ( np.sqrt(np.mean((fr[i]-m1[i])**2)) * np.sqrt(np.mean((avgfr-m2)**2)) ) for i in range(spike_steps.size)])



def get_autocovariance_given_lag(x:np.ndarray, lag:int):
    """Return the autocovariance for a given time lag."""
    if x.size < 2: raise ValueError("error: need at least two elements to calculate the autocovariance")
    x_centered = x - np.mean(x)
    a = np.pad(x_centered, pad_width=(0, lag), mode="constant")
    b = np.pad(x_centered, pad_width=(lag, 0), mode="constant")
    return np.dot(a, b) / x.size

def get_autocovariance(x:np.ndarray, neglags=False):
    """Return the autocovariance for all time lags."""
    if x.size < 2: raise ValueError("error: need at least two elements to calculate the autocovariance")
    x_centered = x - np.mean(x)
    if not neglags: return np.correlate(x_centered, x_centered, mode="full")[x.size - 1:] / x.size
    else: return np.correlate(x_centered, x_centered, mode="full") / x.size

def power_spectral_density(x:np.ndarray, sampfreq_Hz:float):
    """Return a tuple:
    - sampling frequency (Hz)
    - power spectral density"""
    return signal.periodogram(x, sampfreq_Hz, scaling="density")

def power_spectral_density_normalized(x:np.ndarray, sampfreq_Hz:float):
    """The power spectrum is normalized by dividing all densities
    with the total area under curve, such that the graph may be 
    interpreted as a probability density function of the frequencies.

    Return a tuple:
    - sampling frequency (Hz)
    - normalized power spectral density"""
    freq, Sxx = signal.periodogram(x, sampfreq_Hz, scaling="density")
    return freq, Sxx/(Sxx.sum()*np.abs(freq[-1]-freq[-2]))

class FilterButterworth():

    def __init__(self, N:int, Wn, fs:float, btype="lowpass", **option):
        self.butterworth_coeff = signal.butter(N=N, Wn=Wn, fs=fs, btype=btype, **option)

    def filter_signal(self, x):
        return signal.lfilter(*self.butterworth_coeff, x)

def prob_dens(data:np.ndarray, binsize:float, min_val="auto", max_val="auto"):
    """Return a tuple:\n
    - mid-point of bins
    - probability density"""
    start_from_min,start_from_max = False,False
    if min_val == "auto": min_val = np.amin(data)
    else: start_from_min = True
    if max_val == "auto": max_val = np.amax(data)
    else: start_from_max = True
    num_bin = math.ceil((max_val-min_val)/binsize)
    if num_bin < 0: raise ValueError("variable \"num_bin\" is negative, possible erroneous argument \"min_val\" or \"max_val\"")
    if start_from_max: bins = np.linspace(max_val-num_bin*binsize-binsize/2, max_val+binsize/2, num_bin+1)
    elif start_from_max and start_from_min: bins = np.linspace(min_val-binsize/2, max_val+binsize/2, num_bin+1)
    else: bins = np.linspace(min_val-binsize/2, min_val+num_bin*binsize+binsize/2, num_bin+1)
    densities, binedges = np.histogram(data, bins=bins, density=True)
    return (binedges[1:]+binedges[:-1])/2, densities

def prob_dens_CustomBin(data:np.ndarray, bins:np.ndarray):
    """Return a tuple:\n
    - mid-point of bins
    - probability density"""
    densities, binedges = np.histogram(data, bins=bins, density=True)
    return (binedges[1:]+binedges[:-1])/2, densities

def cumu_dens(data:np.ndarray):
    """Return a tuple:\n
    - data in discrete step
    - cumulative probability density"""
    data_sorted = np.sort(data)
    return np.concatenate([data_sorted,data_sorted[[-1]]]), np.arange(data_sorted.size+1)

def get_linbins(min_max_val:tuple[float,float], binsize=None, num_bin=None, return_xcenters=False):
    min_val, max_val = min_max_val
    if num_bin is None: num_bin = int((max_val-min_val)/binsize)
    binedges = np.linspace(min_val, max_val, num_bin)
    if return_xcenters: return binedges, (binedges[1:]+binedges[:-1])/2
    else: return binedges

def get_logbins(min_max_val=None, data=None, binsize=None, num_bin=None, return_xcenters=False):
    """`binsize` in log-scale.\n
    Return an array or a tuple:\n
    - bin edges
    - mid-point of bins (if `return_xcenters` = `True`)\n
    Note:
    - specify either `min_max_val` or `data` only
    - specify either `num_bin` or `binsize` only"""
    if data is not None and min_max_val is None:
        vmin, vmax = np.log10(data.min()), np.log10(data.max())
    elif min_max_val is not None and data is None:
        vmin, vmax = np.log10(min_max_val[0]), np.log10(min_max_val[1])
    else: raise ValueError("specify either data or vlim only")
    if num_bin is not None and binsize is None:
        binsize = (vmax-vmin) / num_bin
    elif binsize is not None and num_bin is None:
        num_bin = np.floor((vmax-vmin) / binsize).astype(int)
    else: raise ValueError("specify either num_bin or binsize only")
    bin_min, bin_max = vmin - binsize/2, vmax + binsize/2
    binedges = 10**np.arange(bin_min, bin_max+binsize, binsize)
    if return_xcenters:
        xcenters = np.sqrt(binedges[:-1]*binedges[1:])
        return binedges, xcenters
    return binedges

def get_linlogbins(min_max_val:tuple[float,float], linbinsize:float, logbinsize:float, return_xcenters=False, print_cutoff=False):
    """Linear-bins are used when binsize of log-bins is smaller than that of linear-bins.
    The linear-log-bins cutoff value depends on `linbinsize` and `logbinsize`. `logbinsize` in log-scale.\n
    Set `print_cutoff` = `True` to show the linear-log bin cutoff value.\n
    Return an array or a tuple:\n
    - bin edges
    - xcenters of linear-bins and log-bins (if `return_xcenters` = `True`)\n
    """
    min_val, max_val = min_max_val
    linbins, xmidpts = get_linbins(min_max_val=(min_val,max_val), binsize=linbinsize, return_xcenters=True)
    logbins, xcenters = get_logbins(min_max_val=(min_val,max_val), binsize=logbinsize, return_xcenters=True)
    linlogbin_cutoff = linbinsize
    logbinsize_s = np.diff(logbins)
    cut_logbins = logbins[np.argmax(logbinsize_s >= linlogbin_cutoff):]
    cut_xcenters = xcenters[np.argmax(logbinsize_s >= linlogbin_cutoff):]
    cut_linbins = linbins[np.argwhere(linbins < cut_logbins[0]).flat]
    cut_xmidpts = xmidpts[np.argwhere(linbins < cut_logbins[0]).flat][:-1]
    cut_logbins *= 10**logbinsize / (cut_logbins[0] / cut_linbins[-1])
    cut_xcenters = np.sqrt(cut_logbins[:-1]*cut_logbins[1:])
    if print_cutoff: print("linear-log bin cutoff: {:5f} ms".format((cut_linbins[-1]+cut_logbins[0])/2))
    if return_xcenters: return np.hstack([cut_linbins,cut_logbins]), np.hstack([cut_xmidpts,(cut_linbins[-1]+cut_logbins[0])/2,cut_xcenters])
    else: return np.hstack((cut_linbins,cut_logbins))

def break_at_discontinuity(sequence:np.ndarray, lower_discont:float, upper_discont:float, threshold=.999):
    sequence = sequence.copy()
    sequence[np.where(sequence <= threshold*lower_discont)[0]] = np.nan
    sequence[np.where(sequence >= threshold*upper_discont)[0]] = np.nan
    return sequence

def fill_lowerleft_triangular_matrix(LLtrimatrix_flatten:np.ndarray):
    size = int(np.sqrt(len(LLtrimatrix_flatten)*2))+1
    mask = np.tri(size,dtype=bool,k=-1) # or np.arange(n)[:,None] > np.arange(n)
    LLmatrix = np.zeros((size,size),dtype=float)
    LLmatrix[mask] = LLtrimatrix_flatten
    return LLmatrix



class NeuronalNetwork:

    def __init__(self):
        self.size = None
        self.adjacency_matrix = None
        self.link_ij = None
        self._neuron_type = None

    def adjacency_matrix_from_file(self, filepath:str):
        """
        Read an adjacency matrix from a file, which stores
        only nonzero elements in each row, with format:
            j i w_ij, separated by ` ` (whitespace),
        "j" is the pre-synaptic neuron index,
        "i" is the post-synaptic neuron index,
        "w_ji" is the synaptic weight of the link directing
        from j to i. Our neuron index runs from 1 to N.
        """
        content = list(csv.reader(open(filepath,'r',newline=''),delimiter=' '))
        self.size = int(content[0][0])                                     # the first row is the network size/number of neurons
        self.adjacency_matrix = np.zeros((self.size,self.size))
        for x in content[1:]:                                              # the remaining rows are the links with
            #                         "j"          "i"           "wij"     # non-zero synaptic weights
            self.adjacency_matrix[int(x[1])-1][int(x[0])-1] = float(x[2])   # "-1" as our index runs from 1 to N
        self.link_ij = np.vstack(np.nonzero(self.adjacency_matrix))

    def adjacency_matrix_from(self, adjacency_matrix):
        """Import our adjacency matrix directly from
        a 2D list or numpy array."""
        self.adjacency_matrix = np.array(adjacency_matrix)
        self.size = self.adjacency_matrix.shape[0]
        self.link_ij = np.vstack(np.nonzero(self.adjacency_matrix))

    def adjacency_matrix_to_file(self, filename: str):
        """Write the adjacency matrix into a file.
        See `adjacency_matrix_from_file` for format."""
        delimiter = ' '
        with open(filename,'w') as f:
            f.write(str(self.size))
            for j, i in itertools.product(range(self.size),range(self.size)):
                if self.adjacency_matrix[i][j] != 0:
                    f.write('\n{:d}{}{:d}{}{:.10f}'.format(j+1, delimiter, \
                        i+1, delimiter, self.adjacency_matrix[i][j]))

    def scale_synaptic_weights(self, scale:float, neuron_type="all"):
        """Multiply all synaptic weights by a common factor: `scale`.
        Set `neuron_type` to `all`, `exc` or `inh` to scale all synaptic weights,
        only excitatory weights or only inhibitory only"""
        if neuron_type == "all": self.adjacency_matrix *= scale
        elif neuron_type == "exc": self.adjacency_matrix[self.adjacency_matrix > 0] *= scale
        elif neuron_type == "inh": self.adjacency_matrix[self.adjacency_matrix < 0] *= scale
        else: raise ValueError("invalid argument \"neuron_type\"")

    def presynaptic_neuron_index(self, start_from=0, presynaptic_neuron_type="all"):
        """With `start_from` = x, all neuron indexes start from x.
        Count only EXC or INH presynaptic neurons using `presynaptic_neuron_type`."""
        if presynaptic_neuron_type == "all": return np.array([np.argwhere(self.adjacency_matrix[i] != 0).flatten()-start_from for i in range(self.size)], dtype=object)
        elif presynaptic_neuron_type == "exc": return np.array([np.argwhere(self.adjacency_matrix[i] > 0).flatten()-start_from for i in range(self.size)], dtype=object)
        elif presynaptic_neuron_type == "inh": return np.array([np.argwhere(self.adjacency_matrix[i] < 0).flatten()-start_from for i in range(self.size)], dtype=object)

    @property
    def synaptic_weights(self):
        return np.hstack(self.adjacency_matrix[self.adjacency_matrix != 0])

    @property
    def synaptic_weights_inh(self):
        return np.hstack(self.adjacency_matrix[self.adjacency_matrix < 0])

    @property
    def synaptic_weights_exc(self):
        return np.hstack(self.adjacency_matrix[self.adjacency_matrix > 0])

    @property
    def neuron_type(self):
        """Each neuron can be classified into excitatory if the
        synaptic weights of all outgoing links are positive,
        or inhibitory if those are negative."""
        if self._neuron_type is None: self._neuron_type = np.array(["uncl" if np.all(col == 0) \
            else ("exc" if np.all(col >= 0) else ("inh" if np.all(col <= 0) \
                else "none")) for col in self.adjacency_matrix.T])
        return self._neuron_type

    @property
    def num_link(self):
        """Find the number of links with non-zero synaptic weight."""
        return len(np.hstack(self.adjacency_matrix)[np.hstack(self.adjacency_matrix) != 0])

    @property
    def num_link_exc(self):
        """Find the number of links with positive synaptic weight."""
        return len(np.hstack(self.adjacency_matrix)[np.hstack(self.adjacency_matrix) > 0])

    @property
    def num_link_inh(self):
        """Find the number of links with negative synaptic weight."""
        return len(np.hstack(self.adjacency_matrix)[np.hstack(self.adjacency_matrix) < 0])

    @property
    def connection_prob(self):
        """Find the connection probability defined by [number of links/total number of possible links]."""
        return self.num_link/self.size**2

    @property
    def in_degree(self):
        """Find the incoming degrees, denoted as k_in, of all
        neurons. The i-th element of the returned array is
        the incoming degree of the (i+1)th neuron (since
        our neuron index starts from 1).
        """
        return np.array([len(row[row != 0]) for row in self.adjacency_matrix])

    @property
    def in_degree_exc(self):
        """Find the excitatory incoming degree of all neurons.
        We only consider the incoming links with weight > 0."""
        return np.array([len(row[row > 0]) for row in self.adjacency_matrix])

    @property
    def in_degree_inh(self):
        """Find the excitatory incoming degree of all neurons.
        We only consider the incoming links with weight < 0."""
        return np.array([len(row[row < 0]) for row in self.adjacency_matrix])

    @property
    def out_degree(self):
        """Find the outgoing degrees, denoted as k_out, of all
        neurons. The i-th element of the returned array is the
        outgoing degree of the (i+1)th neuron (since our neuron
        index starts from 1)."""
        return np.array([len(row[row != 0]) for row in self.adjacency_matrix.T])

    @property
    def synaptic_input(self):
        """Find the sum of synaptic weights of incoming links."""
        return np.sum(self.adjacency_matrix, axis=1)

    @property
    def synaptic_input_inh(self):
        """Find the sum of synaptic weights of inhibitory incoming links."""
        return np.sum(self.adjacency_matrix.clip(max=0), axis=1)

    @property
    def synaptic_input_exc(self):
        """Find the sum of synaptic weights of excitatory incoming links."""
        return np.sum(self.adjacency_matrix.clip(min=0), axis=1)

    @property
    def synaptic_output(self):
        """Find the sum of synaptic weights of outgoing links."""
        return np.sum(self.adjacency_matrix, axis=0)

    @property
    def mean_inweight(self):
        """Find the mean synaptic weight of incoming links."""
        return np.array([np.sum(row)/len(row[row != 0]) if len(row[row != 0]) > 0 \
            else None for row in self.adjacency_matrix])

    @property
    def mean_inweight_inh(self):
        """Find the mean synaptic weight of inhibitory incoming links."""
        return np.array([np.sum(row.clip(max=0))/len(row[row < 0]) if len(row[row < 0]) > 0 \
            else None for row in self.adjacency_matrix])

    @property
    def mean_inweight_exc(self):
        """Find the mean synaptic weight of excitatory incoming links."""
        return np.array([np.sum(row.clip(min=0))/len(row[row > 0]) if len(row[row > 0]) > 0 \
            else None for row in self.adjacency_matrix])

    @property
    def mean_outweight(self):
        """Find the mean synaptic weight of outgoing links."""
        return np.array([np.sum(col)/len(col[col != 0]) if len(col[col != 0]) > 0 \
            else None for col in self.adjacency_matrix.T])


class NeuronalDynamics:

    def __init__(self, directory:str, stepsize_ms:float, duration_ms:float):
        self._directory = directory
        self._stepsize_ms = stepsize_ms
        self._duration_ms = duration_ms
        self._num_step = int(duration_ms/stepsize_ms)
        self._num_step_orig = int(duration_ms/stepsize_ms)
        self.spike_steps = load_spike_steps(os.path.join(self._directory,"spks.txt"))
        self._spike_times = None
        self._spike_train = None
        self._spike_count = None
        self._mean_spike_rate = None
        self._interspike_intervals = None
        self._network_spike_times = None
        self._interevent_intervals = None
        self._interevent_intervals_insteps = None
        self._avg_spike_rate_binned = None
        self._avg_spike_rate_gauskern = None
        self.time_series = None
        self.analysis = None

    def _reset_all(self):
        self._spike_times = None
        self._spike_train = None
        self._spike_count = None
        self._mean_spike_rate = None
        self._interspike_intervals = None
        self._network_spike_times = None
        self._interevent_intervals = None
        self._interevent_intervals_insteps = None
        self._avg_spike_rate_binned = None
        self._avg_spike_rate_gauskern = None

    @property
    def spike_steps(self): return self._spike_steps

    @spike_steps.setter
    def spike_steps(self, _spike_steps): self._spike_steps = _spike_steps

    @property
    def spike_times(self):
        """time scale: milliseconds (ms)"""
        if self._spike_times is None: self._spike_times = self._spike_steps*self._stepsize_ms
        return self._spike_times

    @property
    def spike_train(self):
        if self._spike_train is None: self._spike_train = get_spike_train(self._spike_steps,self._num_step+1)
        return self._spike_train

    @property
    def spike_count(self):
        if self._spike_count is None: self._spike_count = get_spike_count(self._spike_steps)
        return self._spike_count

    @property
    def mean_spike_rate(self):
        """frequency scale: Hertz (Hz)"""
        if self._mean_spike_rate is None: self._mean_spike_rate = get_mean_spike_rate(self._spike_steps,self._duration_ms)
        return self._mean_spike_rate

    @property
    def interspike_intervals(self):
        """time scale: seconds (s)"""
        if self._interspike_intervals is None: self._interspike_intervals = get_interspike_intervals(self._spike_steps,self._stepsize_ms)
        return self._interspike_intervals

    @property
    def network_spike_times(self):
        """time scale: milliseconds (ms)"""
        if self._network_spike_times is None: self._network_spike_times = get_network_spike_times(self._spike_steps, self._stepsize_ms)
        return self._network_spike_times

    @property
    def interevent_intervals(self):
        """time scale: milliseconds (ms)"""
        if self._interevent_intervals is None: self._interevent_intervals = get_interevent_intervals(self._spike_steps, self._stepsize_ms)
        return self._interevent_intervals

    @property
    def interevent_intervals_insteps(self):
        """time scale: milliseconds (ms)"""
        if self._interevent_intervals_insteps is None: self._interevent_intervals_insteps = get_interevent_intervals_insteps(self._spike_steps)
        return self._interevent_intervals_insteps

    def average_spike_rate_time_histogram(self, binsize_ms:float):
        """time scale: seconds (s), frequency scale: Hertz (Hz)"""
        return get_average_spike_rate_time_histogram(self._spike_steps,self._stepsize_ms,self._duration_ms,binsize_ms)

    def average_spike_rate_time_curve(self, kernelstd_ms:float):
        """time scale: seconds (s), frequency scale: Hertz (Hz)"""
        return get_average_spike_rate_time_curve(self._spike_steps,self._stepsize_ms,self._duration_ms,kernelstd_ms)

    class _time_series_handler:

        def __init__(self, num_neuron:int, directory:str, potential:str, adaptation:str,
                     conductance_exc:str, conductance_inh:str, synap_current:str, stoch_current:str):
            potential_filepath = os.path.join(directory,potential)
            adaptation_filepath = os.path.join(directory,adaptation)
            conductance_exc_filepath = os.path.join(directory,conductance_exc)
            conductance_inh_filepath = os.path.join(directory,conductance_inh)
            synap_current_filepath = os.path.join(directory,synap_current)
            stoch_current_filepath = os.path.join(directory,stoch_current)
            self._num_neuron = num_neuron
            self._num_step_beg_trim = 0
            self._num_step_end_trim = 0
            self._v_filepath = potential_filepath
            self._u_filepath = adaptation_filepath
            self._isyn_filepath = synap_current_filepath
            self._ge_filepath = conductance_exc_filepath
            self._gi_filepath = conductance_inh_filepath
            self._istc_filepath = stoch_current_filepath

        @property
        def membrane_potential(self):
            try:
                x = load_time_series(self._v_filepath, num_neuron=self._num_neuron)
                return x[:,self._num_step_beg_trim:x.shape[1]-self._num_step_end_trim]
            except FileNotFoundError: print("warning: no membrane potential series file \"{}\"".format(self._v_filepath))

        @property
        def recovery_variable(self):
            try:
                x = load_time_series(self._u_filepath, num_neuron=self._num_neuron)
                return x[:,self._num_step_beg_trim:x.shape[1]-self._num_step_end_trim]
            except FileNotFoundError: print("warning: no recovery variable series file \"{}\"".format(self._u_filepath))

        @property
        def presynaptic_current(self):
            try:
                x = load_time_series(self._isyn_filepath, num_neuron=self._num_neuron)
                return x[:,self._num_step_beg_trim:x.shape[1]-self._num_step_end_trim]
            except FileNotFoundError: print("warning: no presynaptic current series file \"{}\"".format(self._synapi_filepath))

        @property
        def conductance_exc(self):
            try:
                x = load_time_series(self._ge_filepath, num_neuron=self._num_neuron)
                return x[:,self._num_step_beg_trim:x.shape[1]-self._num_step_end_trim]
            except FileNotFoundError: print("warning: no presynaptic EXC conductance series file \"{}\"".format(self._ge_filepath))

        @property
        def conductance_inh(self):
            try:
                x = load_time_series(self._gi_filepath, num_neuron=self._num_neuron)
                return x[:,self._num_step_beg_trim:x.shape[1]-self._num_step_end_trim]
            except FileNotFoundError: print("warning: no presynaptic INH conductance series file \"{}\"".format(self._gi_filepath))

        @property
        def stochastic_current(self):
            try:
                x = load_time_series(self._istc_filepath, num_neuron=self._num_neuron)[:,self._num_step_beg_trim:-self._num_step_end_trim]
                return x[:,self._num_step_beg_trim:x.shape[1]-self._num_step_end_trim]
            except FileNotFoundError: print("warning: no stochastic current series file \"{}\"".format(self._stochi_filepath))

        def set_trim_duration(self, trim_beg_t_ms:float, trim_end_t_ms:float, stepsize_ms:float):
            self._num_step_beg_trim = int(trim_beg_t_ms/stepsize_ms)
            self._num_step_end_trim = int(trim_end_t_ms/stepsize_ms)

    def time_series_data_from_file(self, num_neuron:int, directory:str,
            potential="memp.bin", adaptation="recv.bin", conductance_exc="gcde.bin",
            conductance_inh="gcdi.bin", synap_current="curr.bin", stoch_current="stoc.bin"):
        self.time_series = self._time_series_handler(num_neuron, directory, potential, adaptation,
            conductance_exc, conductance_inh, synap_current, stoch_current)

    def init_data_analysis(self):
        self.analysis = self._analysis(self)

    class _analysis:

        def __init__(self, outer_instance):
            self.outer = outer_instance

        def C_measure(self, binsize_ms:float):
            return get_C_measure(binsize_ms, self.outer._spike_steps, self.outer._stepsize_ms, self.outer._duration_ms)

        @property
        def KS_test_results(self):
            """Load the K-S test info and K-S test results for different xmax as a tuple:
            - `ks_test_info`: dict (json)
                - `fit_type`: str, can be `powerlaw`, `exp`
                - `KS`: float, KS value
                - `fit_param1`: float, if powerlaw: fitted exponent
                - `fit_xmin`: float, min value for which the fitted function starts
                - `fit_xmax`: float, max value for which the fitted function starts
                - `xmin_search_range`: tuple, range of min values searched for fitting
                - `xmax_search_range`: tuple, range of max values searched for fitting
                - `xmax_search_skip`: int, skips in xmax_search_range
                - `discrete_KSfit`: bool
            - `ks_test_results`: pandas DataFrame
                - columns: `"fit_exponent"`, `"fit_xmin"`, `"fit_xmax"`, `"KS"`
                - rows: `fit_xmax` in `xmax_search_range` with `xmax_search_skip`
            """
            file_directory = self.outer._directory
            ks_test_info = json.load(open(os.path.join(file_directory,"KS_test_info.json"), "r"))
            ks_test_results = pd.read_pickle(os.path.join(file_directory,"KS_test_results.pdDF"))
            return ks_test_info, ks_test_results


class QuickGraph:

    def bar_chart_INT(self, dataINT, ax=None, **options):
        label,count = np.unique(dataINT,return_counts=True)
        if ax is not None:
            ax.bar(label,count,align="center",**options)
            ax.set_axisbelow(True)
            ax.grid(True)
        return label,count

    def prob_dens_plot(self, data, binsize:float, ax=None, plotZero=True, min_val="auto", max_val="auto", **options):
        xcenters,probdens = prob_dens(np.hstack(data).flatten(), binsize, min_val, max_val)
        if not plotZero: xcenters[np.argwhere(probdens==0)] = np.nan # remove zero densities
        if not plotZero: probdens[np.argwhere(probdens==0)] = np.nan # remove zero densities
        if ax is not None:
            ax.plot(xcenters, probdens, **options)
            ax.set_axisbelow(True)
            ax.grid(True)
        return xcenters, probdens

    def prob_dens_plot_CustomBins(self, data, bins, ax=None, xcenters=None, plotZero=True, **options):
        _xcenters,probdens = prob_dens_CustomBin(np.hstack(data).flatten(), bins=bins)
        if xcenters is not None:
            _xcenters = xcenters
        if not plotZero: _xcenters[np.argwhere(probdens==0)] = np.nan # remove zero densities
        if not plotZero: probdens[np.argwhere(probdens==0)] = np.nan # remove zero densities
        if ax is not None:
            ax.plot(_xcenters, probdens, **options)
            ax.set_axisbelow(True)
            ax.grid(True)
        return _xcenters, probdens

    def cumu_dens_plot(self, data, ax=None, likelihood=False, **options):
        """`likelihood`: if True, y-axis is the \"likelihood of occurrence\", otherwise \"number of occurrence\""""
        xdata,cumudens = cumu_dens(data)
        if likelihood: cumudens = cumudens/data.size
        if ax is not None:
            ax.step(xdata, cumudens, **options)
            ax.set_axisbelow(True)
            ax.grid(True)
        return xdata, cumudens

    def raster_plot(self, spike_times, ax=None, colors=None, time_range_ms=["auto","auto"], **options):
        """Neuron index starts from 1."""
        if colors is None: colors = ["k" for _ in range(spike_times.size)]
        elif type(colors)==str: colors = [colors for _ in range(spike_times.size)]
        if time_range_ms[0] == "auto": time_range_ms[0] = float(np.amin([x[0] for x in spike_times if x.size != 0]))
        if time_range_ms[1] == "auto": time_range_ms[1] = float(np.amax([x[-1] for x in spike_times if x.size != 0]))
        spike_times = np.array([x[np.where((time_range_ms[0] < np.array(x)) & (np.array(x) < time_range_ms[1]))] for x in spike_times],dtype=object)
        # [ax.scatter(x/1000, np.full(x.size, i+1), c=colors[i], lw=0, **options) for i,x in enumerate(spike_times)]
        [ax.plot(x/1000, np.full(x.size, i+1), c=colors[i], lw=0, **options) for i,x in enumerate(spike_times)]

    def event_plot(self, spike_times, ax=None, colors=None, time_range_ms=["auto","auto"], **options):
        """Neuron index starts from 1."""
        if time_range_ms[0] == "auto": time_range_ms[0] = 0
        if time_range_ms[1] == "auto": time_range_ms[1] = float(np.amax(np.hstack(spike_times)))
        if colors is None: colors = ["k" for _ in range(len(spike_times))]
        spike_times = [np.array([])]+list(spike_times)
        colors = ["k"]+list(colors)
        ax.eventplot((np.array(spike_times,dtype=object)/1000),colors=colors,**options)

    mycolors = ["k", "r", "b", "g", "m", "c", "darkorange", "violet", "brown", "y"]
    def config_figure(self, figsize=[6.4, 4.8], dpi=100):
        """`figsize`: tuple or list\n`dpi`: int"""
        plt.rcParams["figure.figsize"] = figsize
        plt.rcParams["figure.dpi"] = dpi
    def config_font(self, font=None, size=None):
        """`font`: "avenir" or "charter" or "courier" etc.\n`size`: int or float"""
        if font!=None: plt.rc("font", family=font)
        if size!=None: plt.rc("font", size=size)
    def set_mathtext_style(self, font):
        params = {"mathtext.fontset": "custom"}
        matplotlib.rcParams["mathtext.rm"] = font
        matplotlib.rcParams["mathtext.it"] = font+":italic"
        matplotlib.rcParams["mathtext.bf"] = font+":bold"
        plt.rcParams.update(params)
        # plt.rcParams.update({"mathtext.default":"regular"})
    def set_font_STIX(self, size=None):
        # params = {"text.usetex":False}
        # plt.rcParams.update(params)
        matplotlib.rcParams["mathtext.fontset"] = "stix"
        matplotlib.rcParams["font.family"] = "STIXGeneral"
        if size!=None: plt.rc("font", size=size)
    def set_mathtext_style_STIX(self):
        params = {"mathtext.fontset": "stix"}
        plt.rcParams.update(params)

    def stix_style(self, size=20): self.config_figure([6,5]); self.set_font_STIX(size=size); self.set_mathtext_style_STIX()
    def serif_style(self, size=20): self.config_figure([6,5]); self.config_font(font="Charter",size=size); self.set_mathtext_style("Charter")
    def typewriter_style(self, size=20): self.config_figure([6,5]); self.config_font(font="Courier",size=size); self.set_mathtext_style("Courier")
    def modern_style(self, size=20): self.config_figure([6,5]); self.config_font(font="Avenir",size=size); self.set_mathtext_style("Avenir")
    def default_legend_style(self):
        plt.rcParams["legend.fontsize"] = 16
        plt.rcParams["legend.frameon"] = True
        plt.rcParams["legend.fancybox"] = False
        plt.rcParams["legend.facecolor"] = (1, 1, 1)
        plt.rcParams["legend.edgecolor"] = (0, 0, 0)
        plt.rcParams["legend.framealpha"] = .95
        plt.rcParams["legend.borderpad"] = .4
        plt.rcParams["legend.columnspacing"] = 1.5

    def multiple_formatter(self, denominator=2, number=np.pi, latex="\pi"):
        def gcd(a, b):
            while b: a, b = b, a%b
            return a
        def _multiple_formatter(x, pos):
            den = denominator
            num = np.int(np.rint(den*x/number))
            com = gcd(num,den)
            (num,den) = (int(num/com),int(den/com))
            if den==1:
                if num==0: return r'$0$'
                if num==1: return r'$%s$'%latex
                elif num==-1: return r'$-%s$'%latex
                else: return r'$%s%s$'%(num,latex)
            else:
                if num==1: return r'$\frac{%s}{%s}$'%(latex,den)
                elif num==-1: return r'$\frac{-%s}{%s}$'%(latex,den)
                else: return r'$\frac{%s%s}{%s}$'%(num,latex,den)
        return _multiple_formatter
    class Multiple:
        def __init__(self, denominator=2, number=np.pi, latex="\pi"):
            self.denominator = denominator
            self.number = number
            self.latex = latex

        def locator(self):
            return plt.MultipleLocator(self.number / self.denominator)

        def formatter(self):
            return plt.FuncFormatter(self.multiple_formatter(self.denominator, self.number, self.latex))


class NeuroData:

    def __init__(self, directory:str, **options):
        self.directory = directory
        self.configs = yaml.safe_load(open(os.path.join(directory,"sett.json"),'r'))
        """
        Numerical settings:
        - `num_neuron`: int
        - `stepsize_ms`: float
        - `sampfreq_Hz`: float
        - `duration_ms`: float
        - `num_step`: int

        Inputs to neurons:
        - `noise_intensity`: float
        - `rng_seed`: float
        - `const_current`: float
        - `stimulus_file`: str

        Network:
        - `network_file`: str
        - `weightscale_factor_exc`: float
        - `weightscale_factor_inh`: float

        Other settings:
        - `init_potential`: float
        - `init_recovery` or `init_adaptation`: float
        - `exp_trunc_step_exc`: int
        - `exp_trunc_step_inh`: int
        - `data_series_export`: dict
            - `potential`: bool
            - `recovery` or `adaptation`: bool 
            - `current_synap`: bool
            - `conductance_exc`: bool
            - `conductance_inh`: bool
            - `current_stoch`: bool
        """
        self._compatibility()
        self._suppress_warning = True
        self.network = NeuronalNetwork()
        self._netFound = False
        try: self.network.adjacency_matrix_from_file(self.configs["network_file"]); self._netFound = True
        except FileNotFoundError:
            try: self.network.adjacency_matrix_from_file(os.path.join(directory,self.configs["network_file"])); self._netFound = True
            except FileNotFoundError:
                if not self._suppress_warning: print("Warning: network file not found. \"network\" functions cannot be used. Note: to access the network file, put it into the same the directory as this script.")
        if self._netFound:
            self.network.scale_synaptic_weights(scale=self.configs["weightscale_factor_exc"], neuron_type="exc")
            self.network.scale_synaptic_weights(scale=self.configs["weightscale_factor_inh"], neuron_type="inh")
        else: del self.network

        self.dynamics = NeuronalDynamics(os.path.join(directory),self.configs["stepsize_ms"],self.configs["duration_ms"])
        try: self.configs["data_series_export"]["recovery"]; _ufile = "recv.bin"
        except KeyError: self.configs["data_series_export"]["adaptation"]; _ufile = "adap.bin"
        self.dynamics.time_series_data_from_file(self.configs["num_neuron"],self.directory,"memp.bin",_ufile,"gcde.bin","gcdi.bin","isyn.bin","istc.bin")
        self.dynamics.init_data_analysis()

    def _compatibility(self):
        ### compatibility ###
        try: self.configs["stepsize_ms"]
        except KeyError: self.configs["stepsize_ms"] = self.configs["dt_ms"]
        try: self.configs["num_neuron"]
        except KeyError: self.configs["num_neuron"] = self.configs["neuron_num"]
        try: self.configs["weightscale_factor_exc"]
        except KeyError:
            try: self.configs["weightscale_factor_exc"],self.configs["weightscale_factor_inh"] = self.configs["weightscale_factor"], self.configs["weightscale_factor"]
            except KeyError: self.configs["weightscale_factor_exc"],self.configs["weightscale_factor_inh"] = self.configs["beta"], self.configs["beta"]
        try: self.configs["noise_intensity"]
        except KeyError:
            try: self.configs["noise_intensity"] = self.configs["noiselv"]
            except KeyError: self.configs["noise_intensity"] = self.configs["alpha"]
        if type(self.configs["duration_ms"])==str: self.configs["duration_ms"] = float(self.configs["duration_ms"])
        if type(self.configs["stepsize_ms"])==str: self.configs["stepsize_ms"] = float(self.configs["stepsize_ms"])
        self.configs["sampfreq_Hz"] = 1000/self.configs["stepsize_ms"]
        self.configs["num_step"] = int(self.configs["duration_ms"]/self.configs["stepsize_ms"])
        yaml.dump(self.configs, open(os.path.join(self.directory, "sett.yml"), 'w'), default_flow_style=False)

    def remove_dynamics(self, remove_beg_t_ms:float, remove_end_t_ms:float):
        """remove the first `remove_beg_t_ms` ms and the last `remove_end_t_ms` ms"""
        self.dynamics._reset_all()
        self.dynamics.spike_steps = trim_spike_steps(self.dynamics.spike_steps, remove_beg_t_ms, self.configs["duration_ms"]-remove_end_t_ms, self.configs["stepsize_ms"])-int(remove_beg_t_ms/self.configs["stepsize_ms"])
        self.configs["duration_ms"] -= remove_beg_t_ms+remove_end_t_ms
        self.dynamics._duration_ms -= remove_beg_t_ms+remove_end_t_ms
        self.dynamics._num_step = int(self.configs["duration_ms"]/self.configs["stepsize_ms"])
        self.dynamics.time_series.set_trim_duration(remove_beg_t_ms, remove_end_t_ms, self.configs["stepsize_ms"])

    def retain_dynamics(self, beg_t_ms:float, end_t_ms:float):
        """retain only dynamics from `beg_t_ms` to `end_t_ms`"""
        self.remove_dynamics(beg_t_ms, self.configs["duration_ms"]-end_t_ms)

    def remove_neurons(self, remove_index:list):
        """`remove_index`: a list of indices of neurons to be removed"""
        self.neuron_mask = np.full(self.configs["num_neuron"], True)
        self.neuron_mask[remove_index] = False
        self.configs["num_neuron"] -= len(remove_index)
        self.dynamics._reset_all()
        self.dynamics.spike_steps = self.dynamics.spike_steps[self.neuron_mask]
        if self._netFound:
            self.network.adjacency_matrix = self.network.adjacency_matrix[np.ix_(np.array(self.neuron_mask),np.array(self.neuron_mask))]

    def retain_neurons(self, retain_index:list):
        """`retain_index`: a list of indices of neurons to be retained, other neurons are removed"""
        self.neuron_mask = np.full(self.configs["num_neuron"], False)
        self.neuron_mask[retain_index] = True
        self.configs["num_neuron"] = len(retain_index)
        self.dynamics._reset_all()
        self.dynamics.spike_steps = self.dynamics.spike_steps[self.neuron_mask]
        if self._netFound:
            self.network.adjacency_matrix = self.network.adjacency_matrix[np.ix_(np.array(self.neuron_mask),np.array(self.neuron_mask))]


class FittingTools:

    def __init__(self):
        self.ks_test_results_DF = None
        self.fit_xmin = None
        self.fit_xmax = None
        self.fit_alpha = None
        self.fit_lambda = None

    def load_data(self, data:np.ndarray, sort=True):
        self.data = np.hstack(data).flatten()
        if sort: self.data = np.sort(self.data)

    def draw_data_distribution(self, linbinsize=0.001, logbinsize=0.1, ax=None, discrete_data=True, **options):
        if discrete_data:
            # for discrete data
            _x, _y = np.unique(self.data, return_counts=True)
            _y = _y.astype(float)/_y.sum()
            # bins, x = get_linlogbins((self.data.min()-linbinsize/2,self.data.max()), linbinsize, logbinsize, return_xcenters=True)
            bins, x = get_linbins((self.data.min(),self.data.max()), linbinsize, return_xcenters=True)
            digits = np.digitize(_x, bins) - 1
            y = np.array([_y[digits == i].sum()/len(np.arange(bins[i], bins[i+1], linbinsize)) for i in range(len(bins) - 1)])
            x, y = x[y > 0], y[y > 0]
        else:
            # for continuous data
            if logbinsize is None: bins, xcenters = get_linbins(min_max_val=(np.amin(self.data),np.amax(self.data)), binsize=linbinsize, return_xcenters=True)
            else: bins, xcenters = get_linlogbins(min_max_val=(np.amin(self.data),np.amax(self.data)), linbinsize=linbinsize, logbinsize=logbinsize, return_xcenters=True, print_cutoff=True)
            x, y = qgraph.prob_dens_plot_CustomBins(self.data, bins=bins, ax=None, plotZero=False, xcenters=xcenters, marker="o", ls="-", mfc="none")
        if ax is not None:
            ax.plot(x, y, **options)
            ax.set_axisbelow(True)
            ax.grid(True)
        return x, y

    # def draw_data_distribution(self, linbinsize=0.001, logbinsize=0.1, ax=None, discrete_data=True, **options):
    #     if discrete_data:
    #         # for discrete data
    #         _x, _y = np.unique(self.data, return_counts=True)
    #         _y = _y.astype(float)/_y.sum()
    #         bins, x = get_linlogbins((self.data.min()-linbinsize/2,self.data.max()), linbinsize, logbinsize, return_xcenters=True)
    #         digits = np.digitize(_x, bins) - 1
    #         y = np.array([_y[digits == i].sum()/len(np.arange(bins[i], bins[i+1], linbinsize)) for i in range(len(bins) - 1)])
    #         x, y = x[y > 0], y[y > 0]
    #     else:
    #         # for continuous data
    #         if logbinsize is None: bins, xcenters = get_linbins(min_max_val=(np.amin(self.data),np.amax(self.data)), binsize=linbinsize, return_xcenters=True)
    #         else: bins, xcenters = get_linlogbins(min_max_val=(np.amin(self.data),np.amax(self.data)), linbinsize=linbinsize, logbinsize=logbinsize, return_xcenters=True, print_cutoff=True)
    #         x, y = qgraph.prob_dens_plot_CustomBins(self.data, bins=bins, ax=None, plotZero=False, xcenters=xcenters, marker="o", ls="-", mfc="none")
    #     if ax is not None:
    #         ax.plot(x, y, **options)
    #         ax.set_axisbelow(True)
    #         ax.grid(True)
    #     return x, y

    def perform_exponential_fitting_KS_test_discrete(self, discrete_stepsize:float,
                                xmin_search_range:tuple[float,float], xmax_search_range:tuple[float,float],
                                xmin_search_skip=1, xmax_search_skip=1, verbose=False):
        """
        `xmin_search_range`: tuple, range of min values searched for fitting
        `xmax_search_range`: tuple, range of max values searched for fitting
        `xmin_search_skip`: int, skips in xmin_search_range, default 1 (no skip)
        `xmax_search_skip`: int, skips in xmax_search_range, default 1 (no skip)
        """
        self.data_discrete = np.round(self.data/discrete_stepsize).astype(int)
        self.xmin_search_range, self.xmax_search_range = xmin_search_range, xmax_search_range
        self.xmin_search_skip, self.xmax_search_skip = xmin_search_skip, xmax_search_skip
        self.discrete_KSfit = True
        if xmin_search_range[0] == xmin_search_range[1]: xmin_chosen = np.array([xmin_search_range[0]])
        else: xmin_chosen = self.data[np.argwhere((xmin_search_range[0] <= self.data) & (self.data <= xmin_search_range[1]))][::xmin_search_skip].flatten()
        if xmax_search_range[0] == xmax_search_range[1]: xmax_chosen = np.array([xmax_search_range[0]])
        else: xmax_chosen = self.data[np.argwhere((xmax_search_range[0] <= self.data) & (self.data <= xmax_search_range[1]))][::xmax_search_skip].flatten()
        if verbose:
            print("number of xmin used: {:d}".format(xmin_chosen.shape[0]))
            print("number of xmax used: {:d}".format(xmax_chosen.shape[0]))
        ks_test_results = []

        if xmin_chosen.shape[0] == 1:
            xmin = int(xmin_chosen[0]/discrete_stepsize)
            for xmax in tqdm(np.round(xmax_chosen/discrete_stepsize).astype(int)):
                fit = Fit(self.data_discrete, discrete=True, xmin=xmin, xmax=xmax, verbose=False)
                ks = fit.exponential.KS()
                fit_lambda = fit.exponential.Lambda/discrete_stepsize
                xmin, xmax = fit.exponential.xmin*discrete_stepsize, fit.exponential.xmax*discrete_stepsize
                ks_test_results.append([fit_lambda, xmin, xmax, ks])
        else:
            for xmax in tqdm(np.round(xmax_chosen/discrete_stepsize).astype(int)):
                _ks_test_results = []
                for xmin in np.round(xmin_chosen/discrete_stepsize).astype(int):
                    fit = Fit(self.data_discrete, discrete=True, xmin=xmin, xmax=xmax, verbose=False)
                    ks = fit.exponential.KS()
                    fit_lambda = fit.exponential.Lambda/discrete_stepsize
                    xmin, xmax = fit.exponential.xmin*discrete_stepsize, fit.exponential.xmax*discrete_stepsize
                    _ks_test_results.append([fit_lambda, xmin, xmax, ks])
                ks_test_results.append(_ks_test_results[np.array(_ks_test_results)[-1].argmin()])
        self.ks_test_results_DF = pd.DataFrame(data=ks_test_results, columns=["fit_lambda", "fit_xmin", "fit_xmax", "KS"])
        if verbose: print(self.ks_test_results_DF)
        self.fit_xmin = self.ks_test_results_DF.loc[self.ks_test_results_DF["KS"].argmin()]["fit_xmin"]
        self.fit_xmax = self.ks_test_results_DF.loc[self.ks_test_results_DF["KS"].argmin()]["fit_xmax"]
        self.fit_lambda = self.ks_test_results_DF.loc[self.ks_test_results_DF["KS"].argmin()]["fit_lambda"]
        self.ks = self.ks_test_results_DF.loc[self.ks_test_results_DF["KS"].argmin()]["KS"]
        self.kstest_info = {
            "fit_type": "exponential",
            "KS": self.ks,
            "fit_param1": self.fit_lambda,
            "fit_xmin": self.fit_xmin,
            "fit_xmax": self.fit_xmax,
            "xmin_search_range": self.xmin_search_range,
            "xmax_search_range": self.xmax_search_range,
            "xmin_search_skip": self.xmin_search_skip,
            "xmax_search_skip": self.xmax_search_skip,
            "discrete_KSfit": True
        }

    def perform_exponential_fitting_KS_test(self, xmin_search_range:tuple[float,float], xmax_search_range:tuple[float,float],
                                xmin_search_skip=1, xmax_search_skip=1, discrete_KSfit=True, verbose=False):
        """
        `xmin_search_range`: tuple, range of min values searched for fitting
        `xmax_search_range`: tuple, range of max values searched for fitting
        `xmin_search_skip`: int, skips in xmin_search_range, default 1 (no skip)
        `xmax_search_skip`: int, skips in xmax_search_range, default 1 (no skip)
        """
        self.xmin_search_range, self.xmax_search_range = xmin_search_range, xmax_search_range
        self.xmin_search_skip, self.xmax_search_skip = xmin_search_skip, xmax_search_skip
        self.discrete_KSfit = False
        if xmin_search_range[0] == xmin_search_range[1]: xmin_chosen = np.array([xmin_search_range[0]])
        else: xmin_chosen = self.data[np.argwhere((xmin_search_range[0] <= self.data) & (self.data <= xmin_search_range[1]))][::xmin_search_skip].flatten()

        if xmax_search_range[0] == xmax_search_range[1]: xmax_chosen = np.array([xmax_search_range[0]])
        else: xmax_chosen = self.data[np.argwhere((xmax_search_range[0] <= self.data) & (self.data <= xmax_search_range[1]))][::xmax_search_skip].flatten()
        if verbose:
            print("number of xmin used: {:d}".format(xmin_chosen.shape[0]))
            print("number of xmax used: {:d}".format(xmax_chosen.shape[0]))
        ks_test_results = []

        if xmin_chosen.shape[0] == 1:
            xmin = xmin_chosen[0]
            for xmax in tqdm(xmax_chosen):
                fit = Fit(self.data, discrete=discrete_KSfit, xmin=xmin, xmax=xmax, verbose=False)
                ks = fit.exponential.KS()
                fit_lambda = fit.exponential.Lambda
                xmin, xmax = fit.exponential.xmin, fit.exponential.xmax
                ks_test_results.append([fit_lambda, xmin, xmax, ks])
        else:
            for xmax in tqdm(xmax_chosen):
                _ks_test_results = []
                for xmin in xmin_chosen:
                    fit = Fit(self.data, discrete=discrete_KSfit, xmin=xmin, xmax=xmax, verbose=False)
                    ks = fit.exponential.KS()
                    fit_lambda = fit.exponential.Lambda
                    xmin, xmax = fit.exponential.xmin, fit.exponential.xmax
                    _ks_test_results.append([fit_lambda, xmin, xmax, ks])
                ks_test_results.append(_ks_test_results[np.array(_ks_test_results)[-1].argmin()])
        self.ks_test_results_DF = pd.DataFrame(data=ks_test_results, columns=["fit_lambda", "fit_xmin", "fit_xmax", "KS"])
        if verbose: print(self.ks_test_results_DF)
        self.fit_xmin = self.ks_test_results_DF.loc[self.ks_test_results_DF["KS"].argmin()]["fit_xmin"]
        self.fit_xmax = self.ks_test_results_DF.loc[self.ks_test_results_DF["KS"].argmin()]["fit_xmax"]
        self.fit_lambda = self.ks_test_results_DF.loc[self.ks_test_results_DF["KS"].argmin()]["fit_lambda"]
        self.ks = self.ks_test_results_DF.loc[self.ks_test_results_DF["KS"].argmin()]["KS"]
        self.kstest_info = {
            "fit_type": "exponential",
            "KS": self.ks,
            "fit_param1": self.fit_lambda,
            "fit_xmin": self.fit_xmin,
            "fit_xmax": self.fit_xmax,
            "xmin_search_range": self.xmin_search_range,
            "xmax_search_range": self.xmax_search_range,
            "xmin_search_skip": self.xmin_search_skip,
            "xmax_search_skip": self.xmax_search_skip,
            "discrete_KSfit": False
        }

    def perform_powerlaw_fitting_KS_test_discrete(self, discrete_stepsize:float,
                                        xmin_search_range:tuple[float,float], xmax_search_range:tuple[float,float],
                                        xmax_search_skip=1, verbose=False):
        """
        `xmin_search_range`: tuple, range of min values searched for fitting
        `xmax_search_range`: tuple, range of max values searched for fitting
        `xmax_search_skip`: int, skips in xmax_search_range, default 1 (no skip)
        """
        self.data_discrete = np.round(self.data/discrete_stepsize).astype(int)
        self.xmin_search_range, self.xmax_search_range = xmin_search_range, xmax_search_range
        self.xmax_search_skip, self.discrete_KSfit = xmax_search_skip, True
        if xmax_search_range[0] == xmax_search_range[1]: xmax_chosen = [xmax_search_range[0]]
        else: xmax_chosen = self.data[np.argwhere((xmax_search_range[0] <= self.data) & (self.data <= xmax_search_range[1]))][::xmax_search_skip].flatten()
        if verbose: print("number of xmax used: {:d}".format(xmax_chosen.shape[0]))
        ks_test_results = []

        for xmax in tqdm(np.round(xmax_chosen/discrete_stepsize).astype(int)):
            fit = Fit(self.data_discrete, discrete=True,
                     xmin=(int(xmin_search_range[0]/discrete_stepsize), int(xmin_search_range[1]/discrete_stepsize)),
                     xmax=xmax, verbose=False)
            ks = fit.power_law.KS()
            fit_alpha = fit.power_law.alpha
            xmin, xmax = fit.power_law.xmin*discrete_stepsize, fit.power_law.xmax*discrete_stepsize
            ks_test_results.append([fit_alpha, xmin, xmax, ks])
        self.ks_test_results_DF = pd.DataFrame(data=ks_test_results, columns=["fit_alpha", "fit_xmin", "fit_xmax", "KS"])
        if verbose: print(self.ks_test_results_DF)
        self.fit_xmin = self.ks_test_results_DF.loc[self.ks_test_results_DF["KS"].argmin()]["fit_xmin"]
        self.fit_xmax = self.ks_test_results_DF.loc[self.ks_test_results_DF["KS"].argmin()]["fit_xmax"]
        self.fit_alpha = self.ks_test_results_DF.loc[self.ks_test_results_DF["KS"].argmin()]["fit_alpha"]
        self.ks = self.ks_test_results_DF.loc[self.ks_test_results_DF["KS"].argmin()]["KS"]
        self.kstest_info = {
            "fit_type": "powerlaw",
            "KS": self.ks,
            "fit_param1": self.fit_alpha,
            "fit_xmin": self.fit_xmin,
            "fit_xmax": self.fit_xmax,
            "xmin_search_range": self.xmin_search_range,
            "xmax_search_range": self.xmax_search_range,
            "xmin_search_skip": None,
            "xmax_search_skip": self.xmax_search_skip,
            "discrete_KSfit": True
        }

    def perform_powerlaw_fitting_KS_test(self, xmin_search_range:tuple[float,float], xmax_search_range:tuple[float,float],
                                        xmax_search_skip=1, verbose=False):
        """
        `xmin_search_range`: tuple, range of min values searched for fitting
        `xmax_search_range`: tuple, range of max values searched for fitting
        `xmax_search_skip`: int, skips in xmax_search_range, default 1 (no skip)
        """
        self.xmin_search_range, self.xmax_search_range = xmin_search_range, xmax_search_range
        self.xmax_search_skip, self.discrete_KSfit = xmax_search_skip, False
        if xmax_search_range[0] == xmax_search_range[1]: xmax_chosen = [xmax_search_range[0]]
        else: xmax_chosen = self.data[np.argwhere((xmax_search_range[0] <= self.data) & (self.data <= xmax_search_range[1]))][::xmax_search_skip].flatten()
        if verbose: print("number of xmax used: {:d}".format(xmax_chosen.shape[0]))
        ks_test_results = []

        for xmax in tqdm(xmax_chosen):
            fit = Fit(self.data, discrete=False, xmin=xmin_search_range, xmax=xmax, verbose=False)
            ks = fit.power_law.KS()
            fit_alpha = fit.power_law.alpha
            xmin, xmax = fit.power_law.xmin, fit.power_law.xmax
            ks_test_results.append([fit_alpha, xmin, xmax, ks])
        self.ks_test_results_DF = pd.DataFrame(data=ks_test_results, columns=["fit_alpha", "fit_xmin", "fit_xmax", "KS"])
        if verbose: print(self.ks_test_results_DF)
        self.fit_xmin = self.ks_test_results_DF.loc[self.ks_test_results_DF["KS"].argmin()]["fit_xmin"]
        self.fit_xmax = self.ks_test_results_DF.loc[self.ks_test_results_DF["KS"].argmin()]["fit_xmax"]
        self.fit_alpha = self.ks_test_results_DF.loc[self.ks_test_results_DF["KS"].argmin()]["fit_alpha"]
        self.ks = self.ks_test_results_DF.loc[self.ks_test_results_DF["KS"].argmin()]["KS"]
        self.kstest_info = {
            "fit_type": "powerlaw",
            "KS": self.ks,
            "fit_param1": self.fit_alpha,
            "fit_xmin": self.fit_xmin,
            "fit_xmax": self.fit_xmax,
            "xmin_search_range": self.xmin_search_range,
            "xmax_search_range": self.xmax_search_range,
            "xmin_search_skip": None,
            "xmax_search_skip": self.xmax_search_skip,
            "discrete_KSfit": False
        }

    def save_KS_test_results(self, directory:str, results_filename="KS_test_results.pdDF", info_filename="KS_test_info.json"):
        """Save the K-S test info and K-S test results for different xmax:
        - `ks_test_info`: dict (json)
            - `fit_type`: str, can be `powerlaw`, `exponential`
            - `KS`: float, KS value
            - `fit_param1`: float, alpha if powerlaw, lambda if exponential
            - `fit_xmin`: float, min value for which the fitted function starts
            - `fit_xmax`: float, max value for which the fitted function starts
            - `xmin_search_range`: tuple, range of min values searched for fitting
            - `xmax_search_range`: tuple, range of max values searched for fitting
            - `xmin_search_skip`: int, skips in xmin_search_range, `None` for powerlaw fitting
            - `xmax_search_skip`: int, skips in xmax_search_range
            - `discrete_KSfit`: bool
        - `ks_test_results`: pandas DataFrame
            - columns: `"fit_alpha"` or `"fit_lambda"`, `"fit_xmin"`, `"fit_xmax"`, `"KS"`
            - rows: `fit_xmax` in `xmax_search_range` with `xmax_search_skip`
        """
        if os.path.isfile(os.path.join(directory,"KS_test_results.pdDF")):
            _input = input("KS test results exist, attempting overwrite, proceed? (Y/N): ")
            if _input == "Y" or _input == "y": _write_data = True
            else: _write_data = False
        else: _write_data = True
        if _write_data:
            open(os.path.join(directory,info_filename), "w").write(json.dumps(self.kstest_info, indent=4))
            self.ks_test_results_DF.to_pickle(os.path.join(directory,results_filename))

    def load_KS_test_results(self, directory:str, results_filename="KS_test_results.pdDF", info_filename="KS_test_info.json"):
        """Load the K-S test info and K-S test results for different xmax as a tuple:
        - `ks_test_info`: dict (json)
            - `fit_type`: str, can be `powerlaw`, `exponential`
            - `KS`: float, KS value
            - `fit_param1`: float, alpha if powerlaw, lambda if exponential
            - `fit_xmin`: float, min value for which the fitted function starts
            - `fit_xmax`: float, max value for which the fitted function starts
            - `xmin_search_range`: tuple, range of min values searched for fitting
            - `xmax_search_range`: tuple, range of max values searched for fitting
            - `xmin_search_skip`: int, skips in xmin_search_range, `None` for powerlaw fitting
            - `xmax_search_skip`: int, skips in xmax_search_range
            - `discrete_KSfit`: bool
        - `ks_test_results`: pandas DataFrame
            - columns: `"fit_alpha"` or `"fit_lambda"`, `"fit_xmin"`, `"fit_xmax"`, `"KS"`
            - rows: `fit_xmax` in `xmax_search_range` with `xmax_search_skip`
        """
        self.ks_test_results_DF = pd.read_pickle(os.path.join(directory,results_filename))
        self.ks_test_info = json.load(open(os.path.join(directory,info_filename), "r"))
        self.ks = self.ks_test_info["KS"]
        if self.ks_test_info["fit_type"] == "powerlaw":
            self.fit_alpha = self.ks_test_info["fit_param1"]
        elif self.ks_test_info["fit_type"] == "exponential":
            self.fit_lambda = self.ks_test_info["fit_param1"]
        self.fit_xmin = self.ks_test_info["fit_xmin"]
        self.fit_xmax = self.ks_test_info["fit_xmax"]
        self.xmin_search_range = self.ks_test_info["xmin_search_range"]
        self.xmax_search_range = self.ks_test_info["xmax_search_range"]
        self.xmax_search_skip = self.ks_test_info["xmax_search_skip"]
        self.discrete_KSfit = self.ks_test_info["discrete_KSfit"]
        return self.ks_test_results_DF, self.ks_test_info

    def draw_fitted_powerlaw_distribution(self, plot_x, plot_y, ax=None, **options):
        """`ax`: matplotlib ax to be plotted
        `plot_x`: x-values of the original data plot
        `plot_y`: y-values of the original data plot
        Return the x- and y-values of the fitted plot"""
        if self.fit_alpha is None: raise ValueError("no power-law fitting data")
        x_fit = np.arange(self.fit_xmin, self.fit_xmax, (self.fit_xmax-self.fit_xmin)/100)
        x_gmean = stats.gmean(plot_x[np.argwhere((self.fit_xmin <= plot_x) & (plot_x <= self.fit_xmax))])
        y_gmean = stats.gmean(plot_y[np.argwhere((self.fit_xmin <= plot_x) & (plot_x <= self.fit_xmax))])
        y_fit = np.power(x_fit,-self.fit_alpha)*y_gmean/x_gmean**-self.fit_alpha
        if ax is not None:
            ax.plot(x_fit, y_fit, **options)
            ax.set_axisbelow(True)
            ax.grid(True)
        return x_fit, y_fit

    def draw_fitted_exponential_distribution(self, plot_x, plot_y, ax=None, **options):
        """`ax`: matplotlib ax to be plotted
        `plot_x`: x-values of the original data plot
        `plot_y`: y-values of the original data plot
        Return the x- and y-values of the fitted plot"""
        if self.fit_lambda is None: raise ValueError("no exponential fitting data")
        x_fit = np.arange(self.fit_xmin, self.fit_xmax, (self.fit_xmax-self.fit_xmin)/100)
        x_mean = np.mean(plot_x[np.argwhere((self.fit_xmin <= plot_x) & (plot_x <= self.fit_xmax))])
        y_gmean = stats.gmean(plot_y[np.argwhere((self.fit_xmin <= plot_x) & (plot_x <= self.fit_xmax))])
        y_fit = np.exp(-self.fit_lambda*x_fit)*y_gmean/np.exp(x_mean*-self.fit_lambda)
        if ax is not None:
            ax.plot(x_fit, y_fit, **options)
            ax.set_axisbelow(True)
            ax.grid(True)
        return x_fit, y_fit



qgraph = QuickGraph()
qgraph.stix_style()
# qgraph.modern_style()
# qgraph.serif_style()
# qgraph.typewriter_style()
qgraph.default_legend_style()
qgraph.config_font("avenir")


"""
QUICK COPY
----------
### imports ###
from mylib import NeuroData, qgraph
from matplotlib import pyplot as plt
import numpy as np


### start ###
directory = "/Users/likchun/NeuroProject/raw_data/net_unifindeg_constwij/spontaneous/0.2,0.2,3"
nd = NeuroData(directory)

fig, ax = plt.subplots(figsize=(5,5))
fig, axes = plt.subplots(3, 2, figsize=(10,5), sharex=True, sharey=True)
gridspec_kw={"width_ratios":[.7,1]}


### legend size and location ###
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right outside of the current axis
ax.legend(loc="center left", bbox_to_anchor=(1,0.5))



### figures in grid ###
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig = plt.figure(figsize=(15,15))
gs = fig.add_gridspec(nrows=3, ncols=4,height_ratios=[4,1,1],width_ratios=[1,1,1,1],hspace=.5)

ax1a = fig.add_subplot(gs[:,0])
div1 = make_axes_locatable(ax1a)
ax1b = div1.append_axes("bottom", "40%", pad=0, sharex=ax1a)
ax1c = div1.append_axes("bottom", "40%", pad=0, sharex=ax1a)
ax1d = div1.append_axes("bottom", "60%", pad=.2, sharex=ax1a)
axes1 = [ax1a, ax1b, ax1c, ax1d]

"""