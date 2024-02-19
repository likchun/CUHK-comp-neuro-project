"""
BurstDetectLib
--------------
From Don

Last update: 14 Feburary, 2024
"""

import numpy as np
from scipy.signal import find_peaks
from statsmodels.nonparametric.smoothers_lowess import lowess
import pickle
import os


def split_by_diff(differences, threshold, min_size):
    # split a series at where the differences exceeds the threshold
    # 1. find out where the series exceeds the threshold, call those gap_ids
    # 2. construct a range from gap_ids[i]+1 to gap_ids[i+1]+1
    # 3. keep only those ranges that has size >= min_size
    # 
    # suppose min_size is 3
    # aaaaAaaaAAaa
    #     ^   ^^
    # aaaa aaa

    # add 0 to front, len(differences) to back, to include burst in both ends
    gap_ids = np.hstack([-1, np.nonzero(np.array(differences) > threshold)[0], len(differences)])
    
    # gap_ids[i] +1, (+1: exclude the starting gap)
    # gap_ids[i+1] -1 +1 +1, (-1: just before the gap), (+1: range end exclusive), (+1: isi -> spikes (|-|-|-|, - -> |))
    groups = [np.arange(gap_ids[i] + 1, gap_ids[i+1] + 1) for i in np.arange(len(gap_ids) - 1) if gap_ids[i+1] - gap_ids[i] >= min_size]
    return groups if groups else [[]]

def local_argmin_between_peaks(histogram, bin_centers, between_peak):
    peak_ids = find_peaks(histogram, distance=2)[0]
    left_peak_ids = peak_ids[bin_centers[peak_ids] < between_peak]
    right_peak_ids = peak_ids[bin_centers[peak_ids] >= between_peak]
    if len(left_peak_ids) == 0 or len(right_peak_ids) == 0:
        return 0
    left_peak_id = left_peak_ids[-1]
    right_peak_id = right_peak_ids[0]
    return left_peak_id + np.argmin(histogram[left_peak_id:right_peak_id])

def find_bursts(logisis, bins, max_cutoff, min_void, min_burst_size):
    # output: (groups, flag, (intra_burst_peak, inter_burst_peak, void, logisi_threshold))

    # if there is no spike at all
    if len(logisis) == 0:
        return ([[]], -1, (-1, -1, -1, -1))

    centers = (bins[:-1] + bins[1:])/2
    histogram = np.histogram(logisis, bins)[0].astype(float)
    histogram = lowess(histogram, centers, frac=0.15, return_sorted=False, it=0)

    # find peaks, separated by at least 2 bins
    peak_ids = find_peaks(histogram, distance=2)[0]
    peak_histogram = histogram[peak_ids]
    peak_isis = centers[peak_ids]

    # if there is no peak at all, this is almost impossible
    if len(peak_ids) == 0:
        return ([[]], -2, (-1, -1, -1, -1))

    # find all peaks with isi <= max_cutoff
    cutoff_mask = peak_isis <= max_cutoff
    cutoff_peak_ids = peak_ids[cutoff_mask]
    
    # if there is no peak at isi <= max cutoff
    if len(cutoff_peak_ids) == 0:
        return ([[]], -3, (-1, -1, -1, -1))
    
    # intra_burst_peak is the tallest peak with isi <= max_cutoff
    cutoff_peak_histogram = peak_histogram[cutoff_mask]
    intra_burst_peak_id = cutoff_peak_ids[np.argmax(cutoff_peak_histogram)]
    intra_burst_peak = centers[intra_burst_peak_id]

    # if there is no peak to the right of the intra_burst_peak
    peak_ids = peak_ids[peak_ids > intra_burst_peak_id]
    if len(peak_ids) == 0:
        return ([[]], -4, (intra_burst_peak, -1, -1, -1))
    
    # start with the tallest peak
    peak_ids = peak_ids[np.argsort(histogram[peak_ids])[::-1]]

    cutoff = -1
    best_void = -1
    best_cutoff = -1
    inter_burst_peak_id = -1
    for peak_id in peak_ids:
        # find void between the intra_burst_peak and each peak on the right
        minimum_id = intra_burst_peak_id + np.argmin(histogram[intra_burst_peak_id:peak_id])
        gmin = histogram[minimum_id]
        g1 = histogram[intra_burst_peak_id]
        g2 = histogram[peak_id]

        void = 1 - gmin / np.sqrt(g1*g2)

        if void > best_void:
            best_void = void
            best_cutoff = centers[minimum_id]
            inter_burst_peak_id = peak_id

        # found the tallest peak with void >= min_void since we start from the tallest peak
        # this is also automatically the peak with highest void so far 
        if void >= min_void:
            cutoff = centers[minimum_id]
            break
    
    inter_burst_peak = centers[inter_burst_peak_id]

    # if no peak has void >= min_void is found
    if cutoff == -1:
        return ([[]], -5, (intra_burst_peak, inter_burst_peak, best_void, best_cutoff))

    groups = [[]]
    if cutoff <= max_cutoff:
        flag = 0
        logisi_threshold = cutoff
        groups = split_by_diff(logisis, logisi_threshold, min_burst_size)
    else:
        flag = 1 if histogram[intra_burst_peak_id] < histogram.max() and histogram[inter_burst_peak_id] < histogram.max() else 2
        if min_burst_size == 2:
            logisi_threshold = cutoff
            groups = split_by_diff(logisis, logisi_threshold, min_burst_size)
            groups = [group for group in groups if np.any(logisis[group[:-1]] <= max_cutoff)]
        else:
            # find the core group
            logisi_threshold = max_cutoff
            groups_core = split_by_diff(logisis, logisi_threshold, min_burst_size)
            
            if len(groups_core[0]) > 0:
                # extend on both sides using a more lenient isi threshold
                logisi_threshold_extend = cutoff
                groups_extend = split_by_diff(logisis, logisi_threshold_extend, min_burst_size)
                
                # the core group should always be contained in one of the extend group
                # keep the extend group if the extend group has at least one intersection with a core groups
                groups = []
                for group_extend in groups_extend:
                    for group_core in groups_core:
                        # only need to check one of the spikes to check intersection
                        if group_core[0] in group_extend:
                            groups.append(group_extend)
                            break

    return (groups, flag, (intra_burst_peak, inter_burst_peak, best_void, cutoff))

def get_burst_ranges_list(dt, steps_list, max_cutoff, bin_width, min_void, min_burst_size, fix_max_cutoff=False, save_dir="./"):
    """
    - burst_ranges_list is a list of list of range
      - 1st dim: different neurons
      - 2nd dim: different bursts
      - 3rd dim: the index of steps of different spikes of a burst that is each element points to a step of a burst of a neuron 
    - burst_mask is a bool array of whether a neuron is bursting"""
    logisis_list = [np.log10(np.diff(steps.astype(float)) * dt) for steps in steps_list]
    logisis_all = np.hstack(logisis_list)
    if len(logisis_all) and not fix_max_cutoff:
        vmin, vmax = logisis_all.min(), logisis_all.max()
        bin_min, bin_max = vmin - bin_width/2, vmax + bin_width/2
        bins = np.arange(bin_min, bin_max + bin_width, bin_width)
        centers =  (bins[:-1] + bins[1:])/2
        histogram_all = np.histogram(logisis_all, bins, density=True)[0].astype(float)
        histogram_all = lowess(histogram_all, centers, frac=0.15, return_sorted=False, it=0)
        local_argmin = local_argmin_between_peaks(histogram_all, centers, max_cutoff)
        max_cutoff = centers[local_argmin] if local_argmin != 0 else max_cutoff
    else:
        bins, centers, histogram_all = [], [], []
    burst_ranges_list, flags, other_outputs = zip(*[find_bursts(logisis=logisis, bins=bins, max_cutoff=max_cutoff, min_void=min_void, min_burst_size=min_burst_size) for logisis in logisis_list])
    burst_ranges_list = [[burst_range for burst_range in burst_ranges if len(burst_range)] for burst_ranges in burst_ranges_list]
    # print(f"bursting fraction: {np.sum([len(burst_ranges) > 0 for burst_ranges in burst_ranges_list])} ({np.sum([len(burst_ranges) > 0 for burst_ranges in burst_ranges_list]) / num_neuron * 100:.3g}%)")
    # print(", ".join([f"flag {i}: {(flags == i).sum()}" for i in np.unique(flags)]))
    # intra_burst_peaks, inter_burst_peaks, voids, isi_thresholds = zip(*other_outputs)
    # np.save(save_dir + "flags.npy", flags)
    # np.save(save_dir + "intra_burst_peaks.npy", intra_burst_peaks)
    # np.save(save_dir + "inter_burst_peaks.npy", inter_burst_peaks)
    # np.save(save_dir + "voids.npy", voids)
    # np.save(save_dir + "isi_thresholds.npy", isi_thresholds)
    return burst_ranges_list


class BurstDetect:

    def __init__(self, spike_steps:np.ndarray, stepsize_ms:float):
        self._spike_steps = spike_steps
        self._stepsize_ms = stepsize_ms
        self.burst_data = get_burst_ranges_list(dt=self._stepsize_ms, steps_list=self._spike_steps, max_cutoff=1, bin_width=0.1, min_void=0.7, min_burst_size=2)
        self._burst_neuron_mask = np.hstack([len(burst_ranges) > 0 for burst_ranges in self.burst_data])

    def save_burst_data(self, save_dir=None):
        if save_dir is not None:
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            else: raise FileExistsError
            with open(os.path.join(save_dir,"burst_data.pkl"), "wb") as f:
                pickle.dump(self.burst_data, f, pickle.HIGHEST_PROTOCOL)
            np.save(os.path.join(save_dir,"burst_mask.npy"), self.burst_mask)

    @property
    def bursting_fraction(self):
        return self._burst_neuron_mask.mean()

    @property
    def burst_size(self):
        return np.array([np.array([len(burst) for burst in neuron]) for neuron in self.burst_ranges_list], dtype=object)

