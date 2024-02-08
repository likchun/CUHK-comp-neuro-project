"""
AvalancheLib
------------
From Don

Last update: 29 November, 2023
"""

import numpy as np
import powerlaw


class AvalancheTool():

    def get_adaptive_timebin_width(self, mean_interspike_intervals_in_stepsize):
        return max(mean_interspike_intervals_in_stepsize/2, 1) # in unit of simulation time step dt

    def fit_powerlaw_expoenent(self, data, xmin=1, xmax=100):
        return powerlaw.Fit(data, discrete=True, xmin=xmin, xmax=xmax).power_law.alpha

    def get_avalanche_sizes_and_durations(self, spike_steps, total_timesteps:int, timebin_width:int):
        """Compute the sizes (number of spikes in an avalanche) and durations (number of time bins in an avalanche) of avalanches in a spike train.
        
        spike_steps: spike_steps[i][j] is the step of the j-th spike of the i-th neuron.
        total_timesteps: total number of time steps in the simulation.
        timebin_width: width of the time bin (in unit of simulation time step) used to compute the avalanches.
        """
        total_timesteps = int(total_timesteps)
        steps_all = np.hstack(spike_steps)
        bins = _get_lin_bins(vlim=(0, total_timesteps), bin_width=timebin_width)
        counts = np.histogram(steps_all, bins)[0]
        clusters = [cluster[1:] for cluster in np.split(counts, np.nonzero(counts == 0)[0]) if cluster.sum()]
        # Find the location of avalanches (work in progress)
        # indices = bins[np.array([(indices[1], indices[-1] + 1) for indices in np.split(np.arange(len(counts)), np.nonzero(counts == 0)[0]) if len(indices) > 1])]
        # plt.scatter(steps_all, neurons_all, s=0.01)
        # plt.hlines(np.full(len(indices), 500), indices[:, 0], indices[:, 1], lw=1000, colors="g", alpha=.4, zorder=-1)
        # plt.show()
        sizes = np.array([cluster.sum() for cluster in clusters])
        durations = np.array([len(cluster) for cluster in clusters])
        return sizes, durations

    def get_avalanche_areas(self, spike_steps, num_time_step:int, time_bin_width:int):
        """Compute the areas (number of distinct neuron fired during an avalanche) of avalanches in a spike train. This takes longer to compute than sizes and durations.

        spike_steps: spike_steps[i][j] is the step of the j-th spike of the i-th neuron.
        total_timesteps: total number of time steps in the simulation.
        timebin_width: width of the time bin (in unit of simulation time step) used to compute the avalanches.
        """
        total_timesteps = int(total_timesteps)
        neurons_list = [np.full(len(steps), i) for i, steps in enumerate(spike_steps)]
        steps_all = np.hstack(spike_steps)
        neurons_all = np.hstack(neurons_list)
        bins = _get_lin_bins(vlim=(0, num_time_step), bin_width=time_bin_width)
        counts = np.histogram(steps_all, bins)[0]
        zero_center_indices = np.nonzero(counts==0)[0]
        starts, ends = bins[zero_center_indices + 1], np.append(bins[zero_center_indices[1:]], bins[-1])
        starts, ends = starts[starts != ends], ends[starts != ends]
        areas = np.array([area for area in [len(np.unique(neurons_all[(steps_all >= i) & (steps_all < j)])) for i, j in zip(starts, ends)] if area])
        return areas

    def get_histogram_hybrid_bin(self, values:list, lin_bin_width:float=1, log_bin_width:float=0.1):
        """Compute the histogram using hybrid bin, that is linear binning for small values and logarithmic binning for large values.

        values: list of values for computing the histogram.
        lin_bin_width: width of the linear bin in linear scale.
        log_bin_width: width of the log bin in log scale.
        """
        x, y = np.unique(values, return_counts=True)
        y = y.astype(float)/y.sum()
        bins, bin_x = _get_hybrid_bins(np.array(values), lin_bin_width=lin_bin_width, log_bin_width=log_bin_width, return_centers=True)
        digits = np.digitize(x, bins) - 1
        bin_y = np.array([y[digits == i].sum()/(np.floor(bins[i+1]) - np.ceil(bins[i]) + 1) for i in range(len(bins) - 1)])
        return (bin_x, bin_y), (x, y)


def _get_lin_bins(values=None, num_bin=None, bin_width=None, return_centers=False, vlim=None):
    if values is not None and vlim is None:
        vmin, vmax = values.min(), values.max()
    elif vlim is not None and values is None:
        vmin, vmax = vlim
    else:
        print("specify either values or vlim only")
        exit(123)
    if num_bin is not None and bin_width is None:
        bin_width = (vmax - vmin) / num_bin
        bin_min, bin_max = vmin - bin_width/2, vmax + bin_width/2
        bins = np.linspace(bin_min, bin_max, num_bin+2)
    elif bin_width is not None and num_bin is None:
        bin_min, bin_max = vmin - bin_width/2, vmax + bin_width/2
        bins = np.arange(bin_min, bin_max + bin_width, bin_width)
    else:
        print("specify either num_bin or bin_width only")
        exit(123)
    if return_centers:
        centers =  (bins[:-1] + bins[1:])/2
        return (bins, centers)
    return bins

def _get_log_bins(values=None, num_bin=None, bin_width=None, return_centers=False, vlim=None):
    if values is not None and vlim is None:
        vmin, vmax = np.log10(values.min()), np.log10(values.max())
    elif vlim is not None and values is None:
        vmin, vmax = vlim
    else:
        print("specify either values or vlim only")
        exit(123)
    if num_bin is not None and bin_width is None:
        bin_width = (vmax - vmin) / num_bin
    elif bin_width is not None and num_bin is None:
        num_bin = np.floor((vmax - vmin) / bin_width).astype(int)
    else:
        print("specify either num_bin or bin_width only")
        exit(123)
    bin_min, bin_max = vmin - bin_width/2, vmax + bin_width/2
    bins = 10**np.arange(bin_min, bin_max+bin_width, bin_width)
    if return_centers:
        centers = np.sqrt(bins[:-1] * bins[1:])
        return (bins, centers)
    return bins

def _get_hybrid_bins(values=None, num_lin_bin=None, lin_bin_width=None, num_log_bin=None, log_bin_width=None, return_centers=False, vlim=None, k=None):
    lin_bins, lin_centers = _get_lin_bins(values, num_bin=num_lin_bin, bin_width=lin_bin_width, vlim=vlim, return_centers=True)
    log_bins, log_centers = _get_log_bins(values, num_bin=num_log_bin, bin_width=log_bin_width, vlim=np.log10(vlim) if vlim is not None else None, return_centers=True)
    if k is None:
        if log_bin_width is None:
            log_bin_width = log_bins[1] - log_bins[0]
        k = int(np.ceil(1/(1 - 10**(-log_bin_width))))
    trunc_lin_bins = lin_bins[lin_bins < k]
    trunc_log_bins = log_bins[log_bins >= k]
    if len(trunc_log_bins) == 0:
        if return_centers:
            return (lin_bins, lin_centers)
        return lin_bins
    if len(trunc_lin_bins) == 0:
        if return_centers:
            return (log_bins, log_centers)
        return log_bins
    bins = np.append(trunc_lin_bins, trunc_log_bins)
    if return_centers:
        centers = np.hstack([(trunc_lin_bins[:-1] + trunc_lin_bins[1:])/2, (trunc_lin_bins[-1] + trunc_log_bins[0])/2, np.sqrt(trunc_log_bins[:-1] * trunc_log_bins[1:])])
        return (bins, centers)
    return bins
