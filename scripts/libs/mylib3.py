"""
MyLib3
------
Contains useful tools & shortcuts

Last update: 6 December, 2023 (Morning)
"""


import os
import csv
import math
import random
import itertools
import collections
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn import mixture
import pandas
import scipy
import yaml
from scipy.ndimage import gaussian_filter1d

MAXVAL_uint16_t = 65535
MAXVAL_uint8_t = 255


ts = lambda duration_ms, stepsize_ms, transient_ms=0: np.arange(int(duration_ms/stepsize_ms))[int(transient_ms/stepsize_ms):]*stepsize_ms # ms
"""Unit: ms"""


### IO tools ###

def load_array(filename, delimiter=' ', dtype=float):
    """Load a one line array from a file."""
    return np.array(list(csv.reader(open(filename, 'r', newline=''), delimiter=delimiter)), dtype=dtype).flatten()

def load_array2D(filename, delimiter=' ', dtype=float):
    """Load a ragged array of arrays with variable length from a file."""
    return np.array([np.array(x, dtype=dtype) for x in csv.reader(open(filename, 'r', newline=''), delimiter=delimiter)], dtype=object)

def convert_symmetric_matrix(lower_left_matrix):
    N = lower_left_matrix.shape[0]+1
    sym_matrix = np.zeros((N, N))
    for i in range(N-1):
        for j in range(0, i):
            sym_matrix[i+1][j] = lower_left_matrix[i][j]
    for i in range(N-1):
        for j in range(i, N):
            sym_matrix[i][j] = sym_matrix[j][i]
    return sym_matrix

def load_spike_times_don(filename, step_size):
    return [np.array(x)*step_size for x in [np.fromstring(line.rstrip(), dtype=int, sep=" ") for line in open(filename, "r").readlines()]]

def load_spike_steps(filename: str, stepsize: float, delimiter='\t', format_inc_spkc=True):
    """Load spike time-stamps for ALL neurons from "spks.txt" files. Return spike times of neurons instead of steps. Set stepsize=1 to return time steps.
    If the first element in each line is the spike count of the neuron, enable `format_inc_spkc`."""
    if format_inc_spkc: return np.array([np.delete(np.array(list(filter(None, n)), dtype=float)*stepsize, [0], 0) for n in list(csv.reader(open(filename, 'r'), delimiter=delimiter))], dtype=object)
    else: return np.array([np.array(list(filter(None, n)), dtype=float)*stepsize for n in list(csv.reader(open(filename, 'r'), delimiter=delimiter))], dtype=object)

def load_spike_times(filename: str, delimiter='\t', format_inc_spkc=False):
    """Load spike time-stamps for ALL neurons from "spkt.txt" files. Return spike time-stamps of neurons.
    If the first element in each line is the spike count of the neuron, enable `format_inc_spkc`."""
    if format_inc_spkc: return np.array([np.delete(np.array(list(filter(None, n)), dtype=float), [0], 0) for n in list(csv.reader(open(filename, 'r'), delimiter=delimiter))], dtype=object)
    else: return np.array([np.array(list(filter(None, n)), dtype=float) for n in list(csv.reader(open(filename, 'r'), delimiter=delimiter))], dtype=object)

def dump_spike_steps(network_spike_times, filename: str, stepsize: float, delimiter='\t', format_inc_spkc=True):
    """Dump spike time-steps for ALL neurons into "spks.txt" file. `network_spike_times` is an array containing N arrays of spike time-stamps."""
    N_spike_steps = np.array([np.array(spkt/stepsize, dtype=int) for spkt in network_spike_times], dtype=object) # Convert spike time-stamps into spike time-steps
    with open(filename, 'w') as f:
        if format_inc_spkc: [[f.write("{}".format(len(spks))), [f.write("{}{}".format(delimiter, s)) for s in spks], f.write("\n")] for spks in N_spike_steps]
        else: [[f.write("{}".format(spks[0])), [f.write("{}{}".format(delimiter, s)) for s in spks[1:]], f.write("\n")] for spks in N_spike_steps]

def dump_spike_times(network_spike_times, filename: str, delimiter='\t', format_inc_spkc=False, decimal_place=2):
    """Dump spike time-stamps for ALL neurons into "spkt.txt" file. `network_spike_times` is an array containing N arrays of spike time-stamps."""
    with open(filename, 'w') as f:
        if format_inc_spkc: [[f.write("{}".format(len(spkt))), [f.write("{}{}".format(delimiter, round(t, decimal_place))) for t in spkt], f.write("\n")] for spkt in network_spike_times]
        else: [[[f.write("{}".format(round(t, decimal_place))) for t in spkt[:1]], [f.write("{}{}".format(delimiter, round(t, decimal_place))) for t in spkt[1:]], f.write("\n")] for spkt in network_spike_times]

def load_time_series(data_file_path: str, num_neuron: int, dtype=np.float32):
    """The data file should be a 32-bit float binary file. Use `dtype` to change the data type."""
    time_series = np.fromfile(data_file_path, dtype=dtype)
    return time_series.reshape((int(time_series.shape[0]/num_neuron), num_neuron)).T

def load_stimulus(filename, returnInfo=False):
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
    content = np.array(list(csv.reader(open(filename,'r',newline=''), delimiter='\t')), dtype=object)
    stim_info = content[0]
    for i in [0,1,2,3,5,6,7]: stim_info[i] = float(stim_info[i])
    if returnInfo: return np.array(content[1],dtype=int), np.array(content[2],dtype=float), stim_info
    else: return np.array(content[1],dtype=int), np.array(content[2],dtype=float)


### spike data tools ###

def sort_(array, by, ascending_order=False):
    """Sort the array by the input argument `by`, in descending order."""
    if len(array) != len(by): raise IndexError("\"array\" must have same length as \"by\"")
    sorting = pandas.DataFrame(np.array([array, by]).T, columns=["array", "by"])
    if not ascending_order: return sorting.sort_values(by="by")["array"]
    else: return np.flip(sorting.sort_values(by="by")["array"])

def sort_spike_times(network_spike_times, by, ascending_order=False):
    """Sort the N arrays of spike trains/spike time-stamps by the input argument `by`, in descending order."""
    if len(network_spike_times) != len(by): raise IndexError("\"network_spike_times\" must have same length as \"by\"")
    sorting = pandas.DataFrame(np.array([network_spike_times, by]).T, columns=["spike_times", "by"])
    if not ascending_order: return sorting.sort_values(by="by")["spike_times"]
    else: return np.flip(sorting.sort_values(by="by")["spike_times"])

def trim_spike_times(network_spike_times, start_t: float, end_t: float):
    """Trim the spike time-stamps to a given range. `start_t` and `end_t` are of unit milliseconds (ms)."""
    # return np.array([spkt[np.where(np.logical_and(spkt >= start_t, spkt < end_t))] for spkt in network_spike_times], dtype=object)
    return np.array([spkt[np.where((start_t < np.array(spkt)) & (np.array(spkt) < end_t))] for spkt in network_spike_times], dtype=object)

def get_spike_count(network_spike_times):
    """Return spike counts from network spike time-stamps."""
    return np.array([len(x) for x in network_spike_times])

def get_mean_firing_rate(spike_count, duration_ms: float, return_unit="Hz"):
    """Return mean firing rates (Hz, by default) from spike counts. `duration` is of unit milliseconds (ms). For `spike_count` see `get_spike_count()`."""
    if return_unit == "Hz": return np.array(spike_count) / duration_ms * 1000
    elif return_unit == "kHz": return np.array(spike_count) / duration_ms

def get_timedep_popul_firing_rate_binned(network_spike_times, num_bins, return_unit="s,Hz"):
    """Return (1) mid-point of bins, (2) network average firing rate from network spike time-stamps."""
    spike_num, binedges = np.histogram(np.hstack(network_spike_times), num_bins)
    if return_unit == "s,Hz": netfr = spike_num/(binedges[1:]-binedges[:-1])*1000/len(network_spike_times)
    elif return_unit == "s,kHz": netfr = spike_num/(binedges[1:]-binedges[:-1])/len(network_spike_times)
    else: raise ValueError("return_unit cannot be {}".format(return_unit))
    return (binedges[1:]+binedges[:-1])/2/1000, netfr

def get_timedep_popul_firing_rate_gauskern(spike_train,stepsize_ms,kernel_bandwidth_ms=1):
    """Return time series of population firing rate (unit:Hz)"""
    return gaussian_filter1d(spike_train.mean(axis=0),kernel_bandwidth_ms/stepsize_ms)[:-1]/(stepsize_ms/1000)

def get_interspike_intervals(network_spike_times, return_unit="s"):
    """Return ISIs (s, by default) from N lists of spike time-stamps for N neurons.
    `network_spike_times` must be an array or a list."""
    if return_unit == "s": return np.array([np.diff(n_times)/1000 for n_times in network_spike_times], dtype=object)
    elif return_unit == "ms": return np.array([np.diff(n_times) for n_times in network_spike_times], dtype=object)

def spiketrain_from_spikestep(spike_timesteps, num_of_timesteps: int):
    """The spike times file should contains only the time steps (but not time-stamps) at which the neuron spikes."""
    spike_train = np.zeros(num_of_timesteps)
    spike_train[spike_timesteps] = 1
    return spike_train


### essentials ###

def band_filter_butter(signal, cutoff, fs, filter_order=2, btype="bandpass"):
    b, a = scipy.signal.butter(N=filter_order, Wn=cutoff, btype=btype, fs=fs)
    return scipy.signal.lfilter(b, a, signal)

def power_spectral_density(signal, sampling_freq_Hz, duration_s=None, normalizedByTotalPower=False, onlyPositiveFreq=True):
    """Return the power spectral densities[1] and the sampling frequencies[0] of a signal.\n
    `normalizedByTotalPower`, default: False. If True, the power spectrum is normalized by 
    dividing all densities with the total area under curve, such that the graph may be 
    interpreted as a probability density function of the frequencies."""
    if normalizedByTotalPower: duration_s = 1
    else:
        if duration_s is None: raise ValueError("the argument \"duration_s\" must be provided")
    power_spectrum = np.abs(np.fft.fft(signal))**2/duration_s
    freqs = np.fft.fftfreq(signal.size, 1/sampling_freq_Hz)
    idx = np.argsort(freqs)
    freqs = freqs[idx]
    power_spectrum = power_spectrum[idx]
    if onlyPositiveFreq:
        posfreqs_idx = np.argwhere(freqs>0)[0][0]
        freqs, power_spectrum = freqs[posfreqs_idx:], power_spectrum[posfreqs_idx:]
    if normalizedByTotalPower: normalization_fractor = 1./np.sum(power_spectrum)
    else: normalization_fractor = 1
    return freqs, power_spectrum*normalization_fractor

def response_amplitude(spike_train,stepsize_ms):
    populfr = get_timedep_popul_firing_rate_gauskern(spike_train,stepsize_ms)
    return np.quantile(populfr,.99)-np.quantile(populfr,.01)

def delta_measure(resp_amp,resp_amp_0): return np.log(resp_amp/resp_amp_0)


### figure tools ###

def fig_template_basic(figsize=(7,7)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True)
    ax.set_axisbelow(True)
    return fig, ax

def fig_template_dens_dist(xlabel='', figsize=(9,7)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set(xlabel=xlabel, ylabel='Probability density')
    ax.grid(True)
    ax.set_axisbelow(True)
    return fig, ax

def dens_dist(data: any, binsize: float, min_val="auto", max_val="auto"):
    """Return [0] x-location and [1] height of bins."""
    start_from_min = False
    start_from_max = False
    if min_val == 'auto': min_val = np.amin(data)
    else: start_from_min = True
    if max_val == 'auto': max_val = np.amax(data)
    else: start_from_max = True
    bins_amt = math.ceil((max_val - min_val) / binsize)
    if bins_amt < 0: raise ValueError("variable \"bins_amt\" is negative, possible erroneous argument \"min_val\" or \"max_val\"")
    if start_from_max:
        bins = np.linspace(max_val-bins_amt*binsize, max_val, bins_amt)
    elif start_from_max and start_from_min:
        bins = np.linspace(min_val, max_val, bins_amt)
    else:
        bins = np.linspace(min_val, min_val+bins_amt*binsize, bins_amt)
    density, binedge = np.histogram(data, bins=bins, density=True)
    return (binedge[1:]+binedge[:-1])/2, density

def cumu_dist(data: any, binsize: float, min_val="auto", max_val="auto"):
    """Return [0] x-location and [1] height of bins."""
    start_from_min = False
    start_from_max = False
    if min_val == 'auto': min_val = np.amin(data)
    else: start_from_min = True
    if max_val == 'auto': max_val = np.amax(data)
    else: start_from_max = True
    bins_amt = math.ceil((max_val - min_val) / binsize)
    if bins_amt < 0: raise ValueError("variable \"bins_amt\" is negative, possible erroneous argument \"min_val\" or \"max_val\"")
    if start_from_max:
        bins = np.linspace(max_val-bins_amt*binsize, max_val, bins_amt)
    elif start_from_max and start_from_min:
        bins = np.linspace(min_val, max_val, bins_amt)
    else:
        bins = np.linspace(min_val, min_val+bins_amt*binsize, bins_amt)
    n, binedge, patches = plt.hist(data, bins_amt, density=True, histtype="step", cumulative=True)
    return (binedge[1:]+binedge[:-1])/2, n

def dens_dist_xlogscale(data: any, binsize: float):
    """Return [0] x-location and [1] height of bins."""
    data = np.array(data)
    data = data[data != 0]
    max_elem, min_elem = np.amax(data), np.amin(data)
    bins_amt = math.ceil((math.log10(max_elem) - math.log10(min_elem)) / binsize)
    bins = np.logspace(math.log10(min_elem), math.log10(min_elem)+bins_amt*binsize, bins_amt)
    count, binedge = np.histogram(data, bins=bins)
    # prob_density = np.divide(count, np.diff(binedge))
    normalization = np.dot((binedge[1:]+binedge[:-1])/2, count)
    prob_density = count/normalization
    return (binedge[1:] + binedge[:-1]) / 2, prob_density


### other data tools ###

def pd_dataframe(keys: list, data: list):
    return pandas.DataFrame(np.array(data, dtype=object).T, columns=keys)

def flatten_list(list_of_list: list[list]):
    return [x for sublist in list_of_list for x in sublist]

def group_tuple_list(data: list[tuple], key: any):
    """Group a list of tuples by one of the tuple elements."""
    ddict = collections.defaultdict(list)
    for x in data: ddict[x[key]].append(x)
    return ddict

def number_density(data: list[float], binwidth: float, beg: float, end: float):
    return np.histogram(data, bins=np.arange(beg, end+binwidth, binwidth))[0]

def is_within_region(point: tuple, region: list[tuple]):
    """Return True if the given point is the within the specified region else False.

    PARAMETERS:
    `point`: tuple, (x, y, z)
    `region`: list of tuple, specify the boundaries at each axis

    `len(region)` = `len(point)` = dimension."""
    if len(point) != len(region): raise IndexError('dimensions of `point` and `region` do not match')
    return np.all([region[n][0] <= point[n] <= region[n][1] for n in range(len(region))])

def filter_by_coordinates_3D(points_in_3D: list[tuple], region: list[tuple]):
    """Filter data points in 3D space by coordinates."""
    # return data cooradinates in each axis between the bounds
    return np.array([point for point in points_in_3D if is_within_region(point, region)])

def wrap_within_period(x, period=2*np.pi): return (x + period/2) % (period) - period/2

def breakline_at_discontinuities(y,lower_discont,upper_discont,threshold=.998):
    y = y.copy()
    y[np.where(y <= threshold*lower_discont)[0]] = np.nan
    y[np.where(y >= threshold*upper_discont)[0]] = np.nan
    return y

def fill_lower_trimatrix(trimatrix_flatten):
    size = int(np.sqrt(len(trimatrix_flatten)*2))+1
    mask = np.tri(size,dtype=bool,k=-1) # or np.arange(n)[:,None] > np.arange(n)
    out = np.zeros((size,size),dtype=float)
    out[mask] = trimatrix_flatten
    return out

def root_mean_square_error(a,b): return np.sqrt(np.mean(np.power(b-a,2)))

def gaussian_fn(x, A, x0, sigma): return 0 + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def fit_gaussian(x, y, fn="pdf", init_param=[0,1], iterations=2000):
    """Fit the data with a Gaussian distribution.\n
    Parameters:
    - `x`, `y`: 1D array, data points to be fitted
    - `fn`: str, {"pdf", "cdf"}, density function used
    - `init_param`: [*mean*, *SD*], initial mean & S.D. to start with

    Return:
    - (mu, sigma), (x_fit, y_fit)
    """
    x_fit = np.linspace(x[0], x[-1], int((x[-1]-x[0])/0.1))
    if fn == "pdf": _fn = scipy.stats.norm.pdf
    elif fn == "cdf": _fn = scipy.stats.norm.cdf
    else: raise ValueError("invalid argument \"fn\"")
    mu, sigma = scipy.optimize.curve_fit(_fn, x, y, p0=init_param, maxfev=iterations)[0]
    return (mu, sigma), (x_fit, _fn(x_fit, mu, sigma))

def fit_gaussian_mixture(data: np.ndarray, component: int, fn="pdf"):
    """Perform a maximum likelihood estimation of a Gaussian mixture model,
    for K Gaussian mixtures.
    See: https://towardsdatascience.com/gaussian-mixture-models-explained-6986aaf5a95\n
    Return [ [ (mean, S.D., mixing probability), (x_fit, y_fit) ] for K components ]"""
    if fn == "pdf": _fn = scipy.stats.norm.pdf
    elif fn == "cdf": _fn = scipy.stats.norm.cdf
    gmm = mixture.GaussianMixture(n_components=component).fit(data.reshape(-1, 1))
    x_fit = data.copy().ravel()
    x_fit.sort()
    y_fits = [weight*_fn(x_fit, mean, np.sqrt(covar)).ravel() for weight, mean, covar in zip(gmm.weights_, gmm.means_, gmm.covariances_)]
    return [((mu, np.sqrt(covar), weight), (x_fit, y_fit)) for mu, covar, weight, y_fit in zip(gmm.means_, gmm.covariances_, gmm.weights_, y_fits)]



class NNetwork:

    def adjacency_matrix_from_file(self, filename: str) -> None:
        """
        Read an adjacency matrix from a file, which stores
        only nonzero elements in each row, with format:
            j i w_ij, separated by ` ` (whitespace),
        "j" is the pre-synaptic neuron index,
        "i" is the post-synaptic neuron index,
        "w_ji" is the synaptic weight of the link directing
        from j to i. Our neuron index runs from 1 to N.
        """
        content = list(csv.reader(open(filename, 'r', newline=''), delimiter=' '))
        self.size = int(content[0][0])                                     # the first row is the network size/number of neurons
        self.adjacency_matrix = np.zeros((self.size, self.size))
        for x in content[1:]:                                              # the remaining rows are the links with
            #                         "j"          "i"           "wij"     # non-zero synaptic weights
            self.adjacency_matrix[int(x[1])-1][int(x[0])-1] = float(x[2])   # "-1" as our index runs from 1 to N
        self.link_ij = np.vstack(np.nonzero(self.adjacency_matrix))

    def adjacency_matrix_from(self, adjacency_matrix: any) -> None:
        """Import our adjacency matrix directly from
        a 2D list or numpy array."""
        self.adjacency_matrix = np.array(adjacency_matrix)
        self.size = self.adjacency_matrix.shape[0]
        self.link_ij = np.vstack(np.nonzero(self.adjacency_matrix))

    def adjacency_matrix_to_file(self, filename: str) -> None:
        """Write the adjacency matrix into a file.
        See `adjacency_matrix_from_file` for format."""
        delimiter = ' '
        with open(filename, 'w') as f:
            f.write(str(self.size))
            for j, i in itertools.product(range(self.size), range(self.size)):
                if self.adjacency_matrix[i][j] != 0:
                    f.write('\n{:d}{}{:d}{}{:.10f}'.format(j+1, delimiter, \
                        i+1, delimiter, self.adjacency_matrix[i][j]))

    def scale_synaptic_weights(self, scale: float, neuron_type="all"):
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
        return np.array(['uncl' if np.all(col == 0) \
            else ('exc' if np.all(col >= 0) \
                else ('inh' if np.all(col <= 0) \
                    else 'none')) \
                        for col in self.adjacency_matrix.T])

    @property
    def num_of_links(self):
        """Find the number of links with non-zero synaptic weight."""
        return len(np.hstack(self.adjacency_matrix)[np.hstack(self.adjacency_matrix) != 0])

    @property
    def num_of_exc_links(self):
        """Find the number of links with positive synaptic weight."""
        return len(np.hstack(self.adjacency_matrix)[np.hstack(self.adjacency_matrix) > 0])

    @property
    def num_of_inh_links(self):
        """Find the number of links with negative synaptic weight."""
        return len(np.hstack(self.adjacency_matrix)[np.hstack(self.adjacency_matrix) < 0])

    @property
    def connection_prob(self):
        """Find the connection probability defined by [number of links/total number of possible links]."""
        return self.num_of_links/self.size**2

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
    def mean_in_weight(self):
        """Find the mean synaptic weight of incoming links."""
        return np.array([np.sum(row)/len(row[row != 0]) if len(row[row != 0]) > 0 \
            else None for row in self.adjacency_matrix])

    @property
    def mean_in_weight_inh(self):
        """Find the mean synaptic weight of inhibitory incoming links."""
        return np.array([np.sum(row.clip(max=0))/len(row[row < 0]) if len(row[row < 0]) > 0 \
            else None for row in self.adjacency_matrix])

    @property
    def mean_in_weight_exc(self):
        """Find the mean synaptic weight of excitatory incoming links."""
        return np.array([np.sum(row.clip(min=0))/len(row[row > 0]) if len(row[row > 0]) > 0 \
            else None for row in self.adjacency_matrix])

    @property
    def mean_out_weight(self):
        """Find the mean synaptic weight of outgoing links."""
        return np.array([np.sum(col)/len(col[col != 0]) if len(col[col != 0]) > 0 \
            else None for col in self.adjacency_matrix.T])

class Dynamics:

    def spike_data_from_file(self, filename, stepsize_ms, duration_ms, start_t=0):
        self.spike_steps = load_spike_steps(filename, 1)
        self.spike_times = load_spike_steps(filename, stepsize_ms)
        # remove transient dynamics
        self.spike_times = trim_spike_times(self.spike_times,start_t=start_t,end_t=duration_ms)
        self.spike_train = np.array([spiketrain_from_spikestep(np.array(x/stepsize_ms,dtype=int),int(duration_ms/stepsize_ms)+1) for x in self.spike_times])[:,int(start_t/stepsize_ms):]
        # self.spike_train = np.array([spiketrain_from_spikestep(x,int(duration_ms/stepsize_ms)+1) for x in np.array([np.array(spkt/stepsize_ms,dtype=int) for spkt in self.spike_times])])
        # self.spike_train = np.array([spiketrain_from_spikestep(x.astype(int),int(duration_ms/stepsize_ms)+1) for x in load_spike_steps(filename,1)])
        self.spike_count = get_spike_count(self.spike_times)
        self.mean_firing_rate = get_mean_firing_rate(self.spike_count, duration_ms-start_t)
        self.interspike_intervals = get_interspike_intervals(self.spike_times)
        self.timedep_popul_firing_rate = get_timedep_popul_firing_rate_gauskern(self.spike_train,stepsize_ms)
        self.timedep_popul_firing_rate_binned = get_timedep_popul_firing_rate_binned(self.spike_times,int((duration_ms-start_t)/5.))[1]

    def timeseries_data_from_file(self, num_neuron, potential_filename="memp.bin",
            recovery_filename="recv.bin", current_filename="curr.bin", conductance_exc_filename="gcde.bin",
            conductance_inh_filename="gcdi.bin", stoch_current_filename="stoc.bin"):
        self.__num_neuron = num_neuron
        self.__v_filename = potential_filename
        self.__u_filename = recovery_filename
        self.__i_filename = current_filename
        self.__ge_filename = conductance_exc_filename
        self.__gi_filename = conductance_inh_filename
        self.__stochi_filename = stoch_current_filename

    @property
    def membrane_potential_series(self):
        try: return load_time_series(self.__v_filename, num_neuron=self.__num_neuron)
        except FileNotFoundError: print("warning: no membrane potential series file \"{}\"".format(self.__v_filename))

    @property
    def recovery_variable_series(self):
        try: return load_time_series(self.__u_filename, num_neuron=self.__num_neuron)
        except FileNotFoundError: print("warning: no recovery variable series file \"{}\"".format(self.__u_filename))

    @property
    def presynaptic_current_series(self):
        try: return load_time_series(self.__i_filename, num_neuron=self.__num_neuron)
        except FileNotFoundError: print("warning: no presynaptic current series file \"{}\"".format(self.__i_filename))

    @property
    def conductance_exc_series(self):
        try: return load_time_series(self.__ge_filename, num_neuron=self.__num_neuron)
        except FileNotFoundError: print("warning: no presynaptic EXC conductance series file \"{}\"".format(self.__ge_filename))

    @property
    def conductance_inh_series(self):
        try: return load_time_series(self.__gi_filename, num_neuron=self.__num_neuron)
        except FileNotFoundError: print("warning: no presynaptic INH conductance series file \"{}\"".format(self.__gi_filename))

    @property
    def stochastic_current_series(self):
        try: return load_time_series(self.__stochi_filename, num_neuron=self.__num_neuron)
        except FileNotFoundError: print("warning: no stochastic current series file \"{}\"".format(self.__stochi_filename))

class Graphing:

    def line_plot(self, x, y, discrete=False, ax=None, **options):
        """
        **options:
            `style`: str, e.g.: `"--"`, `"-^"`, ..., default `"-"`
            `c`: str, plot color, default `"k"`
            `a`: float, plot opaqueness, default `1`
            `ms`: float, marker size, default `10`
            `lw`: float, marker size, default `2`
            `label`: str, default `""`(hide)
            `title`: str, default `""`(hide)
        """
        style, c, a, ms, lw = "-", "k", 1, 10, 2
        label, title = "", ""
        for key, value in options.items():
            if key == "style": style = value
            elif key == "c": c = value
            elif key == "a": a = value
            elif key == "ms": ms = value
            elif key == "lw": lw = value
            elif key == "label": label = value
            elif key == "title": title = value
        if ax == None:
            fig, ax = plt.subplots(figsize=(6,5))
            ax.set_title(title, y=1.02, loc='left')
        else: ax = ax
        if not discrete: ax.plot(x, y, style, color=c, ms=ms, lw=lw, alpha=a, label=label, zorder=5)
        else: ax.step(x, y, style, color=c, ms=ms, lw=lw, alpha=a, label=label, zorder=5)
        ax.set_axisbelow(True)
        ax.grid(True)
        return x, y

    def scatter_plot(self, x, y, ax=None, **options):
        """
        **options:
            `style`: str, e.g.: `"--"`, `"-^"`, ..., default `"x"`
            `c`: str, plot color, default `"k"`
            `a`: float, plot opaqueness, default `1`
            `s`: float, marker size, default `1`
            `label`: str, default `""`(hide)
            `title`: str, default `""`(hide)
        """
        style, c, a, s = "x", "k", 1, 1
        label, title = "", ""
        for key, value in options.items():
            if key == "style": style = value
            elif key == "c": c = value
            elif key == "a": a = value
            elif key == "s": s = value
            elif key == "label": label = value
            elif key == "title": title = value
        if ax == None:
            fig, ax = plt.subplots(figsize=(6,5))
            ax.set_title(title, y=1.02, loc='left')
        else: ax = ax
        try: ax.scatter(x, y, marker=style, s=s, color=c, alpha=a, label=label, zorder=5)
        except ValueError: ax.scatter(x, y, marker=style, s=s, c=c, alpha=a, label=label, zorder=5)
        ax.set_axisbelow(True)
        ax.grid(True)
        return x, y

    def bar_chart_INT(self, data_ints, ax=None, **options):
        """The data are preferably integers. y-axis gives
        the frequency of occurrence. Size of bin is 1.

        **options:
            `c`: str, plot color, default `"k"`
            `a`: float, plot opaqueness, default `1`
            `label`: str, default `""`(hide)
            `title`: str, default `""`(hide)
        """
        c, a, label, title = "k", 1, "", ""
        for key, value in options.items():
            if key == "c": c = value
            elif key == "a": a = value
            elif key == "label": label = value
            elif key == "title": title = value
        if ax == None:
            fig, ax = plt.subplots()
            ax.set_title(title, y=1.02, loc='left')
            ax.set(ylabel="Frequency")
        else: ax = ax
        labels, counts = np.unique(data_ints, return_counts=True)
        if ax != "noplot":
            ax.bar(labels, counts, align="center", color=c, alpha=a, label=label, zorder=5)
            ax.set_axisbelow(True)
            ax.grid(True)
        return labels, counts

    def distribution_density_plot(self, data, binsize: float, xlogscale=False, max_y_is_1=False, ax=None, **options):
        """Draw a probability distribution.\b
        Caution: there may be fewer data in the last bin (or first bin if `max_val` is used),
        consider removing it or set a bin size such that the data range is evenly distributed
        with the bins.

        `data` is a one-dimenional array containing numerical values
        `xlogscale`: use log scale to bin the data
        `max_y_is_1`: the maximum y-value is set to 1

        **options:
            `min_val`: minimum value to start from
            `max_val`: maximum value to end with
            other options: (see function `line_plot()`)
        """
        min_val, max_val = "auto", "auto"
        # if len(~np.isfinite(data)) > 0:
        #     print("warning: invalid values such as inf, nan are detected and removed")
        #     data = data[np.isfinite(data)]
        for key, value in options.items():
            if key == "": _ = value
            elif key == "min_val": min_val = value
            elif key == "max_val": min_val = value
        if xlogscale: x, y = dens_dist_xlogscale(np.hstack(data).flatten(), binsize)
        else: x, y = dens_dist(np.hstack(data).flatten(), binsize, min_val, max_val)
        if max_y_is_1 == True: y /= np.amax(y)
        if ax == None:
            fig, ax = plt.subplots()
            ax.set(ylabel="Probability density")
        if ax != "noplot":
            self.line_plot(x, y, ax=ax, **options)
            ax.set_axisbelow(True)
            ax.grid(True)
        return x, y

    def cumulative_distribution_plot(self, data, likelihood=False, ax=None, **options):
        """Draw a cumulative probability distribution.

        `data` is a one-dimenional array containing numerical values
        `likelihood`: y-axis is "likelihood of occurrence", otherwise "count" 

        **options:
            other options: (see function `line_plot()`)
        """
        x = np.sort(np.hstack(data).flatten())
        x = np.concatenate([x, x[[-1]]])
        if likelihood: y = np.arange(x.size)/x.size
        else: y = np.arange(x.size)
        if ax == None:
            fig, ax = plt.subplots()
            if likelihood: ax.set(ylabel="Likelihood of occurrence")
            else: ax.set(ylabel="Count")
        self.line_plot(x, y, discrete=True, ax=ax, **options)
        ax.set_axisbelow(True)
        ax.grid(True)
        return x, y

    def raster_plot_network(self, spike_times, ax, start_t="auto", end_t="auto", **options):
        """Draw raster plot of neurons in a network.
        `start_t` and `end_t` should be in millisecond (ms).
        Neuron index starts from 1.

        **options:
        - `sort_by_spike_count`: bool, sort the neurons in raster plot by their spike counts,
        the most spiked neuron is at the top position, default `False`(do not sort)
        - `separate_neuron_type`: list, neuron type of neurons, separate neurons by their types
        (orange: EXC, teal: INH),
        default `False`(do not sort)
        - `color`: str or list, dots color, default `'k'`(black)
        - `alpha`: float, dots transparency, default `1`(opaque)
        """
        """`save_name`: str, save file name, default `''`(do not save)"""
        sort_by_spike_count = False
        separate_neuron_type = []
        colors, alpha, zorder = "k", 1, 3
        for key, value in options.items():
            if key == "sort_by_spike_count": sort_by_spike_count = value
            elif key == "separate_neuron_type": separate_neuron_type = value
            elif key == 'color': colors = value
            elif key == 'alpha': alpha = value
            # elif key == 'zorder': zorder = value
        if start_t == 'auto': start_t = float(np.amin([x[0] for x in spike_times if len(x) != 0]))
        if end_t == 'auto': end_t = float(np.amax([x[-1] for x in spike_times if len(x) != 0]))

        if len(separate_neuron_type) != 0:
            neuron_type = [0 if t == "exc" else 1 for t in separate_neuron_type]
            type_color = ["b" if t == "inh" else "r" for t in separate_neuron_type]
            # type_color = ["teal" if t == "inh" else "darkorange" for t in separate_neuron_type]
        else:
            type_color = np.zeros(len(spike_times))
            neuron_type = np.zeros(len(spike_times))
        if sort_by_spike_count: spike_count = get_spike_count(spike_times)
        else: spike_count = np.zeros(len(spike_times))

        keys = ["spike_times", "spike_count", "neuron_type", "type_color"]
        data = [[x for x in spike_times], spike_count, neuron_type, type_color]
        df = pd_dataframe(keys, data)
        if not sort_by_spike_count and len(separate_neuron_type) == 0: pass
        else:
            df.sort_values(by=["neuron_type", "spike_count"], ascending=[False, True], inplace=True)
            if len(separate_neuron_type) != 0: colors = list(df["type_color"])
        for i, n_t in enumerate(df["spike_times"]):
            n_t = np.array(n_t)[np.where((start_t < np.array(n_t)) & (np.array(n_t) < end_t))]
            if type(colors) == list: ax.plot(n_t/1000., np.full(len(n_t), i+1), c=colors[i], alpha=alpha, marker="o", ms=.5, lw=0, zorder=0)
            else: ax.plot(n_t/1000., np.full(len(n_t), i+1), c=colors, alpha=alpha, marker="o", ms=.5, lw=0, zorder=0)
        return ax

    def event_plot_neurons(self, spike_times, ax, start_t="auto", end_t="auto", colors="k") -> None:
        """Draw a single neuron raster plot, where each vertical
        line represents a spike at the corresponding time.
        `start_t` and `end_t` should be in millisecond (ms)."""
        if start_t == "auto": start_t = 0
        if end_t == "auto": end_t = float(np.amax(np.hstack(spike_times)))
        spike_times = [np.array([])]+list(spike_times)
        if type(colors)==list or type(colors)==np.ndarray: colors = ["k"]+list(colors)
        ax.eventplot((np.array(spike_times,dtype=object)/1000), colors=colors)
        return ax

    def timedep_popul_firing_rate_binned(self, spike_times, resolution_ms: float, start_t=0, end_t="auto",
                                        color="k", fr_unit="Hz", ax=None, raster_plot=None, **options):
        """Draw the time-dependent population firing rate, which is
        the number of spikes of all neurons in time interval Δt
        divided by Δt and the number of neurons, i.e.,

        netFR(t) = [ Σ_(neuron i=1->N) spike_count(t+Δt) ] / [ Δt * N ]

        \b
        `start_t` / `end_t`: float, should be in millisecond (ms).

        `raster_plot`: bool or dict, default `None`
        - when value == True or type == dict: show raster plot
        - dict {
            `sort_by_spike_count`: bool, sort the neurons in raster plot by their spike counts,
                the most spiked neuron is at the top position, default `False`(do not sort)
            `alpha`: float, dots transparency, default `1`(opaque) }
        """
        sort_by_spike_count, alpha = False, 1
        if raster_plot is not None:
            if type(raster_plot) == dict:
                for key, value in raster_plot.items():
                    if key == 'sort_by_spike_count': sort_by_spike_count = value
                    elif key == 'alpha': alpha = value
        if end_t == "auto": end_t = float(np.amax(np.hstack(spike_times)))

        num_bins = int((end_t-start_t)/resolution_ms)
        bins = np.linspace(start_t, end_t, num=num_bins)
        occurence, binedge = np.histogram(np.hstack(spike_times), bins=bins, density=False)
        x, y = (binedge[1:]+binedge[:-1])/2/1000, occurence/(resolution_ms/1000)/len(spike_times)
        if fr_unit == "kHz": y /= 1000

        if ax == None:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.set(xlabel="Time (s)", ylabel="Average population\nfiring rate (Hz)")
            ax.set(xlim=(start_t/1000, end_t/1000), ylim=(0, 1.2*np.amax(y)))
            if raster_plot is None: ax.grid(True)
            ax.text(1, 1.01, "bin size: {} ms".format(resolution_ms),
                    horizontalalignment="right", verticalalignment="bottom",
                    transform=ax.transAxes, fontdict=dict(size=12))
        else: ax = ax
        if raster_plot is not None:
            axTwin = ax.twinx()
            axTwin.set_yticklabels([])
            axTwin.set_ylim(0, len(spike_times))
            ax.set_zorder(axTwin.get_zorder() + 1)
            ax.set_frame_on(False)
            ax.plot(x, y, c="r", zorder=5)
            self.raster_plot_network(spike_times, start_t, end_t, alpha=alpha,
                                     sort_by_spike_count=sort_by_spike_count, ax=axTwin)
        else: ax.plot(x, y, c=color, **options); ax.grid(True)
        return ax

    def timedep_popul_firing_rate_gauskern(self, spike_train, duration_ms, stepsize_ms, resolution_ms: float,
                                          start_t=0, end_t="auto", color="k", ax=None, **options):
        """Firing rate unit: Hz
        Return tuple [0]: time (x), [1]: popul FR (y), [2]: matplotlib axes"""
        x = ts(duration_ms,stepsize_ms)[int(start_t/stepsize_ms):]/1000 # s
        y = get_timedep_popul_firing_rate_gauskern(spike_train,stepsize_ms)
        if ax == None:
            fig, ax = plt.subplots(figsize=(10,4))
            ax.set(xlabel="Time (s)", ylabel="Population\nfiring rate (Hz)")
            ax.set(xlim=(start_t/1000, end_t/1000), ylim=(0, 1.2*np.amax(y)))
            ax.text(1, 1.01, "bin size: {} ms".format(resolution_ms),
                    horizontalalignment="right", verticalalignment="bottom",
                    transform=ax.transAxes, fontdict=dict(size=12))
        else: ax = ax
        ax.plot(x, y, c=color, **options); ax.grid(True)
        return x, y, ax

    def power_spectrum(self, signal, stepsize_ms, duration_ms=None, normalizedByTotalPower=False, freq_range="auto", ax=None, **options):
        """Return the power spectral densities[1] and frequencies[0] of a signal.\n
        `normalizedByTotalPower`, default: False. If True, the power spectrum is normalized by 
        dividing all densities with the total area under curve, such that the graph may be 
        interpreted as a probability density function of the frequencies."""
        if normalizedByTotalPower: duration_ms = 1000
        else:
            if duration_ms is None: raise ValueError("the argument \"duration_ms\" must be provided")
        x, y = power_spectral_density(signal, 1000/stepsize_ms, duration_ms/1000, normalizedByTotalPower)
        if freq_range=="auto": freq_range = 1000/stepsize_ms
        if ax == None:
            fig, ax = plt.subplots(figsize=(10,4))
            if normalizedByTotalPower: ax.set(xlabel="frequency (Hz)", ylabel="normalized spectral power density")
            else: ax.set(xlabel="frequency (Hz)", ylabel="spectral power density")
            ax.set(xlim=(0,freq_range))
        else: ax = ax
        ax.plot(x,y,**options)
        ax.grid(True)
        return x, y, ax

    def _power_spectrum(self, signal, sampling_freq, axes):
        """`sampling_freq`=`int(1000/stepsize_ms)`"""
        fourier_real_part = np.real(np.fft.fft(signal))
        fourier_imag_part = np.imag(np.fft.fft(signal))
        power_spectrum = np.abs(np.fft.fft(signal))**2
        freqs = np.fft.fftfreq(signal.size, 1/sampling_freq)

        idx = np.argsort(freqs)
        # sp_sum = np.sum(power_spectrum[idx])
        graphing.line_plot(freqs[idx], fourier_real_part[idx], c="r", lw=.7, label="real part", ax=axes[0])
        graphing.line_plot(freqs[idx], fourier_imag_part[idx], c="c", lw=.7, label="imaginary part", ax=axes[0])
        graphing.line_plot(freqs[idx], power_spectrum[idx], lw=1, label="power spectrum", ax=axes[1])
        lg1 = axes[0].legend(fontsize=14, ncol=1, facecolor=(1,1,1), edgecolor=(.5,.5,.5), framealpha=.9, loc="upper right")
        lg1.get_frame().set_boxstyle("round", pad=.05, rounding_size=.5)
        lg2 = axes[1].legend(fontsize=14, facecolor=(1,1,1), edgecolor=(.5,.5,.5), framealpha=.9)
        lg2.get_frame().set_boxstyle("round", pad=.05, rounding_size=.5)
        axes[0].set(ylabel="magnitude", xlabel="frequency")
        axes[0].set_title("Fourier transform", loc="left", pad=25)
        axes[1].set(ylabel="power (mV^2/Hz)", xlabel="frequency (Hz)")

    def config_figure(self, figsize=[6.4, 4.8], dpi=100):
        """`figsize`: tuple or list\n`dpi`: int"""
        plt.rcParams["figure.figsize"] = figsize
        plt.rcParams["figure.dpi"] = dpi

    def config_font(self, font=None, size=None):
        """`font`: "avenir" or "charter" or "courier" etc.\n`size`: int or float"""
        if font!=None: plt.rc("font", family=font)
        if size!=None: plt.rc("font", size=size)

    def set_regular_mathtext_style(self):
        plt.rcParams.update({"mathtext.default":"regular"})

    mycolors = ["k", "r", "b", "g", "m", "darkorange", "c", "violet", "y"]

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

    def serif_style(self,size=20): self.config_figure([6,5]); self.config_font(font="Charter",size=size); self.set_regular_mathtext_style()
    def typewriter_style(self,size=20): self.config_figure([6,5]); self.config_font(font="Courier",size=size); self.set_regular_mathtext_style()
    def modern_style(self,size=20): self.config_figure([6,5]); self.config_font(font="Avenir",size=size); self.set_regular_mathtext_style()
    def default_legend_style(self):
        plt.rcParams["legend.fontsize"] = 16
        plt.rcParams["legend.frameon"] = True
        plt.rcParams["legend.fancybox"] = True
        plt.rcParams["legend.facecolor"] = (1, 1, 1)
        plt.rcParams["legend.edgecolor"] = (0, 0, 0)
        plt.rcParams["legend.framealpha"] = .9
        plt.rcParams["legend.borderpad"] = .4
        plt.rcParams["legend.columnspacing"] = 1.5


class SimulationData:

    def __init__(self, data_path: str, transient_time_ms=0.0, **options) -> None:
        self.data_path = data_path
        self.settings = yaml.safe_load(open(os.path.join(data_path, "sett.json"), 'r'))
        """
        All settings:
        - `network_file`: str
        - `stimulus_file`: str
        - `noiselv`: float
        - `rng_seed`: float
        - `num_neuron`: int
        - `stepsize_ms`: float
        - `sampling_freq_Hz`: float
        - `duration_ms`: float
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
        self.settings["sampling_freq_Hz"] = 1000/self.settings["stepsize_ms"]
        spks_fname, spks_dt_ms, spks_duration_ms = "spks.txt", self.settings["stepsize_ms"], self.settings["duration_ms"]
        for key, value in options.items():
            if key == "": _ = value
            elif key == "spks_filename": spks_fname = value
            elif key == "spks_stepsize_ms": spks_dt_ms = value
            elif key == "spks_duration_ms": spks_duration_ms = value
            else: print("argument: \"{}\" does not exist".format(key))
        yaml.dump(self.settings, open(os.path.join(data_path, "sett.yml"), 'w'), default_flow_style=False)
        self.network = NNetwork()
        self._networkFound = False
        try: self.network.adjacency_matrix_from_file(self.settings["network_file"]); self._networkFound = True
        except FileNotFoundError:
            try: self.network.adjacency_matrix_from_file(os.path.join(data_path, self.settings["network_file"])); self._networkFound = True
            except FileNotFoundError: print("Warning: network file not found. \"network\" functions cannot be used.")
        if self._networkFound: self.network.scale_synaptic_weights(scale=self.settings["weightscale_factor"], neuron_type="all")
        self.dynamics = Dynamics()
        self.transient_time = transient_time_ms
        self.dynamics.spike_data_from_file(os.path.join(data_path, spks_fname), spks_dt_ms, spks_duration_ms, start_t=transient_time_ms)
        self.dynamics.timeseries_data_from_file(self.settings["num_neuron"],
                os.path.join(self.data_path, "memp.bin"),
                os.path.join(self.data_path, "recv.bin"),
                os.path.join(self.data_path, "isyn.bin"),
                os.path.join(self.data_path, "gcde.bin"),
                os.path.join(self.data_path, "gcdi.bin"),
                os.path.join(self.data_path, "stoc.bin"))
        self.__extra_keys, self.__extra_data = [], []

    def add_cols_dataframe(self, keys: list, data: list):
        self.__extra_keys, self.__extra_data = keys, [arr.tolist() if type(arr) != list else arr for arr in data]

    @property
    def dataframe(self):
        """default content:
        - [0]`neuron_index`: int
        - [1]`neuron_type`: {"inh", "exc", "uncl"}
        - [2]`spike_count`: int
        - [3]`mean_firing_rate_Hz`: float
        - [4]`interspike_intervals_ms`: array of floats
        - [5]`spike_timestamps_in_ms`: array of floats
        """
        keys = ["neuron_index", "neuron_type", "spike_count", "mean_firing_rate_Hz",
                "interspike_intervals_ms", "spike_timestamps_in_ms"] + self.__extra_keys
        data = [np.arange(1, self.settings["num_neuron"]+1),
                self.network.neuron_type,
                self.dynamics.spike_count,
                self.dynamics.mean_firing_rate,
                self.dynamics.interspike_intervals,
                self.dynamics.spike_times] + self.__extra_data
        return pandas.DataFrame(np.array(data, dtype=object).T, columns=keys)


class Compressor:

    def load_encoded(self, filename):
        return np.load(open(filename,"rb")), np.load(open("{}_a".format(filename),"rb"))

    def save_encoded(self, filename, encoded, a):
        np.save(open(filename,"wb"),encoded), np.save(open("{}_a".format(filename),"wb"),a)

    def _encode_uint16_t(self, signal, upper_cutoff=30, lower_cutoff=-130, const_a=400):
        signal[signal > upper_cutoff] = upper_cutoff
        signal[signal < lower_cutoff] = lower_cutoff
        return np.array((upper_cutoff-signal)*const_a,dtype=int)

    def _decode_uint16_t(self, encoded, upper_cutoff=30, const_a=400):
        return upper_cutoff-np.array(encoded,dtype=float)/const_a

    def encode_uintX_t(self, signal, X=8):
        """
        Encode floating point signal into uintX_t, `X=8`:uint8_t, `X=16`:uint16_t\n
        Decode signal using `decode_uintX_t`\n
        Return `(encoded_signal, [upper_cutoff,lower_cutoff,map_const,X])`
        """
        upper_cutoff = np.amax(signal)
        lower_cutoff = np.amin(signal)
        if X==16: map_const = float(MAXVAL_uint16_t)/(upper_cutoff-lower_cutoff)
        elif X==8: map_const = float(MAXVAL_uint8_t)/(upper_cutoff-lower_cutoff)
        signal[signal > upper_cutoff] = upper_cutoff
        signal[signal < lower_cutoff] = lower_cutoff
        return (np.array((upper_cutoff-signal)*map_const,dtype="B"),[upper_cutoff,lower_cutoff,map_const,X])

    def decode_uintX_t(self, encoded, a):
        """Decode signal from `encode_uintX_t`"""
        return a[0]-np.array(encoded,dtype=float)/a[2]

class AnalysisTools:

    coeff_variation = lambda self, data_array: np.std(data_array)/np.mean(data_array)

    compressor = Compressor()

    # network_ISICV = lambda self, spike_times: self.coeff_variation(np.diff(np.sort(np.hstack(spike_times))))

    # def spectral_power(self): ...

    # def spectral_entropy(self): ...

    # def detect_bursts(self): ...


class AnalyzedData:

    def draw_phase_diagram_A(self, ax, directory="/Users/likchun/NeuroProject/raw_data/networkB[EI4]_voltage/"):
        w_inh = [1e-8,.2,.4,.6]
        w_exc = [0,.02,.04,.06,.08,.1,.12,.14,.16,.18,.2,.3,.35,.4,.42,.45,.5,.6]
        alpha = 5
        plv_thresh = .25

        crit_wexcL,crit_wexcR = [],[]
        for w_i in w_inh:
            _prev = []
            for w_e in w_exc:
                filepath = directory+"a{2}/matrixB_wi{0}-we{1}_a{2}".format(w_i,w_e,alpha)
                gplv = np.load(open(os.path.join(filepath,"plv_data"),"rb"),allow_pickle=True)[0]
                try:
                    if gplv > plv_thresh and _prev[-1][1] < plv_thresh:
                        crit_wexcL.append(w_e+(_prev[-1][0]-w_e)/(_prev[-1][1]-gplv)*(plv_thresh-gplv))
                except IndexError: pass
                try:
                    if gplv < plv_thresh and _prev[-1][1] > plv_thresh: 
                        crit_wexcR.append(w_e+(_prev[-1][0]-w_e)/(_prev[-1][1]-gplv)*(plv_thresh-gplv))
                except IndexError: pass
                _prev.append((w_e,gplv))
        graphing.line_plot(crit_wexcL,w_inh,style=".-",c="k",ax=ax)
        graphing.line_plot(crit_wexcR,w_inh,style=".-",c="k",ax=ax)

        ax.set(xlim=(0,.7),ylim=(0,.6))
        ax.set(xticks=[0,.2,.4,.6],yticks=[0,.2,.4,.6])
        ax.set(xlabel="$g_{{exc}}$",ylabel="$g_{{inh}}$")
        ax.text(.06,.5,"I",fontdict=dict(font="charter",fontsize=35),ha="center",va="center")
        ax.text(.27,.3,"II",fontdict=dict(font="charter",fontsize=35),ha="center",va="center")
        ax.text(.58,.15,"III",fontdict=dict(font="charter",fontsize=35),ha="center",va="center")


graphing = Graphing()
analtool = AnalysisTools()
analdata = AnalyzedData()


graphing.modern_style()
graphing.default_legend_style()



### useful figure templates ###
"""
gridspec_kw={"height_ratios":[2,1,1]}

lg.get_frame().set_boxstyle("round", pad=.1, rounding_size=.5)

ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi/2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi/12))
ax.xaxis.set_major_formatter(plt.FuncFormatter(graphing.multiple_formatter()))

ax.ticklabel_format(axis="y", style="scientific", scilimits=(-3,3), useOffset=False, useLocale=False, useMathText=True)

ax.yaxis.label.set_color("r")
ax.tick_params(axis="y", colors="r")
"""