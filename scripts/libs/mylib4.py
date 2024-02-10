"""
MyLib4
------
Tools & templates

Last update: 8 Feb, 2024 (pm)
"""

import os
import csv
import math
import yaml
import itertools
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy import signal
from scipy import ndimage

MAXVAL_uint16_t = 65535
MAXVAL_uint8_t = 255


t_range_ = lambda duration_ms, stepsize_ms, transient_ms=0: np.arange(int(duration_ms/stepsize_ms)+1)[int(transient_ms/stepsize_ms):]*stepsize_ms
"""time scale: milliseconds (ms)"""

def load_spike_steps(filepath:str, delimiter='\t', incspkc=True):
    if incspkc: return np.array([np.array(x,dtype=int)[1:] for x in csv.reader(open(filepath,'r'),delimiter=delimiter)],dtype=object)
    else: return np.array([np.array(x,dtype=int) for x in csv.reader(open(filepath,'r'),delimiter=delimiter)],dtype=object)

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
    numspike, binedge = np.histogram(np.hstack(spike_steps*stepsize_ms),np.linspace(0,binsize_ms*num_bin,num_bin))
    return ((binedge[1:]+binedge[:-1])/2/1000)[:-1], (numspike/(binedge[1:]-binedge[:-1])*1000/spike_steps.size)[:-1] # remove the last bin, which cannot be fitted into the time interval for the given bin size

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

def get_spike_train(spike_steps:np.ndarray, num_steps:int):
    spike_train = np.zeros((spike_steps.size,num_steps+1),dtype=int) # +1 to include the initial time step t0
    for i,s in enumerate(spike_train): s[spike_steps[i]] = 1
    return spike_train[:,1:]

def trim_spike_steps(spike_steps:np.ndarray, start_t_ms:float, end_t_ms:float, stepsize_ms:float):
    """Return spike steps"""
    return np.array([np.array(ss[np.where((start_t_ms/stepsize_ms < np.array(ss)) & (np.array(ss) < end_t_ms/stepsize_ms))], dtype=int) for ss in spike_steps], dtype=object)



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
    density, binedge = np.histogram(data, bins=bins, density=True)
    return (binedge[1:]+binedge[:-1])/2, density

def prob_dens_CustomBin(data:np.ndarray, bins:np.ndarray):
    """Return a tuple:\n
    - mid-point of bins
    - probability density"""
    density, binedge = np.histogram(data, bins=bins, density=True)
    return (binedge[1:]+binedge[:-1])/2, density

def cumu_dens(data:np.ndarray):
    """Return a tuple:\n
    - data in discrete step
    - cumulative probability density"""
    data_sorted = np.sort(data)
    return np.concatenate([data_sorted,data_sorted[[-1]]]), np.arange(data_sorted.size+1)

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
        """Multiply all synaptic weights by a common factor: `scale`."""
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
        self._avg_spike_rate_binned = None
        self._avg_spike_rate_gauskern = None
        self.time_series = None

    def _reset_all(self):
        self._spike_times = None
        self._spike_train = None
        self._spike_count = None
        self._mean_spike_rate = None
        self._interspike_intervals = None
        self._avg_spike_rate_binned = None
        self._avg_spike_rate_gauskern = None
        # self.time_series = None

    @property
    def spike_steps(self): return self._spike_steps

    @spike_steps.setter
    def spike_steps(self, _spike_steps): self._spike_steps = _spike_steps

    @property
    def spike_times(self):
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
        if self._mean_spike_rate is None: self._mean_spike_rate = get_mean_spike_rate(self._spike_steps,self._duration_ms)
        return self._mean_spike_rate

    @property
    def interspike_intervals(self):
        if self._interspike_intervals is None: self._interspike_intervals = get_interspike_intervals(self._spike_steps,self._stepsize_ms)
        return self._interspike_intervals

    def average_spike_rate_time_histogram(self, binsize_ms:float):
        return get_average_spike_rate_time_histogram(self._spike_steps,self._stepsize_ms,self._duration_ms,binsize_ms)

    def average_spike_rate_time_curve(self, kernelstd_ms:float):
        return get_average_spike_rate_time_curve(self._spike_steps,self._stepsize_ms,self._duration_ms,kernelstd_ms)

    class _time_series_handler:

        def __init__(self, num_neuron:int,
            potential_filepath="memp.bin", recovery_filepath="recv.bin",
            conductance_exc_filepath="gcde.bin", conductance_inh_filepath="gcdi.bin",
            synap_current_filepath="isyn.bin", stoch_current_filepath="stoc.bin"):
            self._num_neuron = num_neuron
            self._num_trim_step = 0
            self._v_filepath = potential_filepath
            self._u_filepath = recovery_filepath
            self._synapi_filepath = synap_current_filepath
            self._ge_filepath = conductance_exc_filepath
            self._gi_filepath = conductance_inh_filepath
            self._stochi_filepath = stoch_current_filepath

        @property
        def membrane_potential(self):
            try: return load_time_series(self._v_filepath, num_neuron=self._num_neuron)[:,self._num_trim_step:]
            except FileNotFoundError: print("warning: no membrane potential series file \"{}\"".format(self._v_filepath))

        @property
        def recovery_variable(self):
            try: return load_time_series(self._u_filepath, num_neuron=self._num_neuron)[:,self._num_trim_step:]
            except FileNotFoundError: print("warning: no recovery variable series file \"{}\"".format(self._u_filepath))

        @property
        def presynaptic_current(self):
            try: return load_time_series(self._synapi_filepath, num_neuron=self._num_neuron)[:,self._num_trim_step:]
            except FileNotFoundError: print("warning: no presynaptic current series file \"{}\"".format(self._synapi_filepath))

        @property
        def conductance_exc(self):
            try: return load_time_series(self._ge_filepath, num_neuron=self._num_neuron)[:,self._num_trim_step:]
            except FileNotFoundError: print("warning: no presynaptic EXC conductance series file \"{}\"".format(self._ge_filepath))

        @property
        def conductance_inh(self):
            try: return load_time_series(self._gi_filepath, num_neuron=self._num_neuron)[:,self._num_trim_step:]
            except FileNotFoundError: print("warning: no presynaptic INH conductance series file \"{}\"".format(self._gi_filepath))

        @property
        def stochastic_current(self):
            try: return load_time_series(self._stochi_filepath, num_neuron=self._num_neuron)[:,self._num_trim_step:]
            except FileNotFoundError: print("warning: no stochastic current series file \"{}\"".format(self._stochi_filepath))

        def set_trim_transient_duration(self, transient_beg_t_ms:float, stepsize_ms:float):
            self._num_trim_step = int(transient_beg_t_ms/stepsize_ms)

    def time_series_data_from_file(self, num_neuron:int,
            potential_filepath="memp.bin", recovery_filepath="recv.bin",
            conductance_exc_filepath="gcde.bin", conductance_inh_filepath="gcdi.bin",
            synap_current_filepath="curr.bin", stoch_current_filepath="stoc.bin"):
        self.time_series = self._time_series_handler(num_neuron, potential_filepath, recovery_filepath,
            conductance_exc_filepath, conductance_inh_filepath, synap_current_filepath, stoch_current_filepath)


class QuickGraph:

    def bar_chart_INT(self, dataINT, ax=None, **options):
        label,count = np.unique(dataINT,return_counts=True)
        if ax is not None:
            ax.bar(label,count,align="center",**options)
            ax.set_axisbelow(True)
            ax.grid(True)
        return label,count

    def prob_dens_plot(self, data, binsize:float, ax=None, plotZero=True, min_val="auto", max_val="auto", **options):
        xbin,probdens = prob_dens(np.hstack(data).flatten(), binsize, min_val, max_val)
        if ax is not None:
            if not plotZero: xbin[np.argwhere(probdens==0)] = np.nan # remove zero densities
            if not plotZero: probdens[np.argwhere(probdens==0)] = np.nan # remove zero densities
            ax.plot(xbin, probdens, **options)
            ax.set_axisbelow(True)
            ax.grid(True)
        return xbin, probdens

    def cumu_dens_plot(self, data, ax=None, likelihood=False, **options):
        """`likelihood`: if True, y-axis is the \"likelihood of occurrence\", otherwise \"number of occurrence\""""
        xdata,cumudens = cumu_dens(data)
        if likelihood: cumudens = cumudens/data.size
        if ax is not None:
            ax.step(xdata, cumudens, **options)
            ax.set_axisbelow(True)
            ax.grid(True)
        return xdata, cumudens

    def raster_plot(self, spike_times, ax=None, colors=None, time_range=["auto","auto"], **options):
        """Units of `beg_t` and `end_t` are millisecond (ms). Neuron index starts from 1."""
        if colors is None: colors = ["k" for _ in range(spike_times.size)]
        elif type(colors)==str: colors = [colors for _ in range(spike_times.size)]
        if time_range[0] == "auto": time_range[0] = float(np.amin([x[0] for x in spike_times if x.size != 0]))
        if time_range[1] == "auto": time_range[1] = float(np.amax([x[-1] for x in spike_times if x.size != 0]))
        spike_times = np.array([x[np.where((time_range[0] < np.array(x)) & (np.array(x) < time_range[1]))] for x in spike_times],dtype=object)
        # [ax.scatter(x/1000, np.full(x.size, i+1), c=colors[i], lw=0, **options) for i,x in enumerate(spike_times)]
        [ax.plot(x/1000, np.full(x.size, i+1), c=colors[i], lw=0, **options) for i,x in enumerate(spike_times)]

    def event_plot(self, spike_times, ax=None, colors=None, time_range=["auto","auto"], **options):
        """Units of `beg_t` and `end_t` are millisecond (ms). Neuron index starts from 1."""
        if time_range[0] == "auto": time_range[0] = 0
        if time_range[1] == "auto": time_range[1] = float(np.amax(np.hstack(spike_times)))
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
            while b:
                a, b = b, a%b
            return a
        def _multiple_formatter(x, pos):
            den = denominator
            num = np.int(np.rint(den*x/number))
            com = gcd(num,den)
            (num,den) = (int(num/com),int(den/com))
            if den==1:
                if num==0:
                    return r'$0$'
                if num==1:
                    return r'$%s$'%latex
                elif num==-1:
                    return r'$-%s$'%latex
                else:
                    return r'$%s%s$'%(num,latex)
            else:
                if num==1:
                    return r'$\frac{%s}{%s}$'%(latex,den)
                elif num==-1:
                    return r'$\frac{-%s}{%s}$'%(latex,den)
                else:
                    return r'$\frac{%s%s}{%s}$'%(num,latex,den)
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
        All configs:
        - `network_file`: str
        - `stimulus_file`: str
        - `noiselv`: float
        - `rng_seed`: float
        - `num_neuron`: int
        - `stepsize_ms`: float
        - `sampfreq_Hz`: float
        - `duration_ms`: float
        - `num_step`: int
        - `const_current`: float
        - `weightscale_factor`: float
        - `exp_trunc_step_exc`: int
        - `exp_trunc_step_inh`: int
        - `init_potential`: float
        - `init_recovery`: float
        - `data_series_export`: dict
            - `current`: bool
            - `potential`: bool
            - `recovery`: bool
        """
        ### compatibility ###
        try: self.configs["stepsize_ms"]
        except KeyError: self.configs["stepsize_ms"] = self.configs["dt_ms"]
        try: self.configs["num_neuron"]
        except KeyError: self.configs["num_neuron"] = self.configs["neuron_num"]
        try: self.configs["weightscale_factor"]
        except KeyError: self.configs["weightscale_factor"] = self.configs["beta"]
        try: self.configs["noiselv"]
        except KeyError: self.configs["noiselv"] = self.configs["alpha"]

        if type(self.configs["duration_ms"])==str: self.configs["duration_ms"] = float(self.configs["duration_ms"])
        if type(self.configs["stepsize_ms"])==str: self.configs["stepsize_ms"] = float(self.configs["stepsize_ms"])
        self.configs["sampfreq_Hz"] = 1000/self.configs["stepsize_ms"]
        self.configs["num_step"] = int(self.configs["duration_ms"]/self.configs["stepsize_ms"])
        yaml.dump(self.configs, open(os.path.join(directory, "sett.yml"), 'w'), default_flow_style=False)

        self.network = NeuronalNetwork()
        self._netFound = False
        try: self.network.adjacency_matrix_from_file(self.configs["network_file"]); self._netFound = True
        except FileNotFoundError:
            try: self.network.adjacency_matrix_from_file(os.path.join(directory,self.configs["network_file"])); self._netFound = True
            except FileNotFoundError: print("Warning: network file not found. \"network\" functions cannot be used.")
        if self._netFound: self.network.scale_synaptic_weights(scale=self.configs["weightscale_factor"], neuron_type="all")
        else: del self.network

        self.dynamics = NeuronalDynamics(os.path.join(directory),self.configs["stepsize_ms"],self.configs["duration_ms"])
        self.dynamics.time_series_data_from_file(self.configs["num_neuron"],
            os.path.join(directory,"memp.bin"),os.path.join(directory,"recv.bin"),
            os.path.join(directory,"gcde.bin"),os.path.join(directory,"gcdi.bin"),
            os.path.join(directory,"isyn.bin"),os.path.join(directory,"stoc.bin"))

    def trim_transient_dynamics(self, transient_beg_t_ms:float):
        self.dynamics._reset_all()
        self.dynamics.spike_steps = trim_spike_steps(self.dynamics.spike_steps,transient_beg_t_ms,self.configs["duration_ms"],self.configs["stepsize_ms"])-int(transient_beg_t_ms/self.configs["stepsize_ms"])
        self.configs["duration_ms"] -= transient_beg_t_ms
        self.dynamics._duration_ms -= transient_beg_t_ms
        self.dynamics._num_step = int(self.configs["duration_ms"]/self.configs["stepsize_ms"])
        self.dynamics.time_series.set_trim_transient_duration(transient_beg_t_ms,self.configs["stepsize_ms"])

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



qgraph = QuickGraph()
qgraph.stix_style()
# qgraph.modern_style()
# qgraph.serif_style()
# qgraph.typewriter_style()
qgraph.default_legend_style()


"""gridspec_kw={"width_ratios":[.7,1]}"""