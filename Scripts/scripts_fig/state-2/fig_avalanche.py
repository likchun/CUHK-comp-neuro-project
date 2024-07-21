from preamble import *
import powerlaw


def get_histogram_hybrid_bin(values:list, lin_bin_width:float=1, log_bin_width:float=0.1, _values=None):

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

    """Compute the histogram using hybrid bin, that is linear binning for small values and logarithmic binning for large values.

    values: list of values for computing the histogram.
    lin_bin_width: width of the linear bin in linear scale.
    log_bin_width: width of the log bin in log scale.
    """
    x, y = np.unique(values, return_counts=True)
    y = y.astype(float)/y.sum()
    bins, bin_x = _get_hybrid_bins(np.array(values), lin_bin_width=lin_bin_width, log_bin_width=log_bin_width, return_centers=True)
    if _values==None:
        digits = np.digitize(x, bins) - 1
        bin_y = np.array([y[digits == i].sum()/(np.floor(bins[i+1]) - np.ceil(bins[i]) + 1) for i in range(len(bins) - 1)])
        return (bin_x, bin_y), (x, y)
    else:
        bin_y = [np.array(_values)[np.argwhere((np.array(values) > low) & (np.array(values) <= high))].mean() for low, high in zip(bins[:-1], bins[1:])]
        return bin_x, bin_y


fig0, ax0 = plt.subplots(figsize=[6,5])
ax0.set(xlabel="IEI ($\Delta t$)", ylabel="$p$(IEI)")
ax0.set(xscale="log", yscale="log")

fig, [ax1,ax2] = plt.subplots(1, 2, figsize=[9,4.5])
figb, ax3 = plt.subplots(figsize=[5,4.5])
ax1.set(xlabel="$s$", ylabel="$P(s)$")
ax2.set(xlabel="$T$ $(\Delta t)$", ylabel="$P(T)$")
ax3.set(xlabel="$T$ $(\Delta t)$", ylabel=r"$\langle s|T \rangle$")


""""""
t_interval = (0,20000)
param = (0.2, 0.2, 3, 1)
Delta_t = 15
xmin_s = 3
xmax_s = 100
xmin_T = 2
xmax_T = 30
marker = "^"
color = "r"

nd = NeuroData(os.path.join(data_path.SD_netA_mapout_gEgI, "{},{},{},{}".format(*param)))
nd.retain_dynamics(*t_interval)
qgraph.prob_dens_plot(nd.dynamics.interevent_intervals_insteps, 1, c=color, ax=ax0)

# test avalanche detection
steps = np.hstack(nd.dynamics.spike_steps)-int(t_interval[0]/nd.configs["stepsize_ms"])
time_bins = np.arange(0, int((t_interval[1]-t_interval[0])/nd.configs["stepsize_ms"]), Delta_t)
counts = np.histogram(steps, time_bins)[0]
print(f"{np.nonzero(counts)[0]}")
clusters = [cluster[1:] for cluster in np.split(counts, np.nonzero(counts==0)[0]) if cluster.sum()]
sizes = np.array([cluster.sum() for cluster in clusters])
print(f"{sizes}")
durations = np.array([len(cluster) for cluster in clusters])
print(f"{durations}")

fit_sizes_exp = powerlaw.Fit(sizes, discrete=True, xmin=xmin_s, xmax=xmax_s).power_law.alpha
fit_duration_exp = powerlaw.Fit(durations, discrete=True, xmin=xmin_T, xmax=xmax_T).power_law.alpha
print("Fitted exponent for avalanche size: {}".format(fit_sizes_exp))
print("Fitted exponent for avalanche duration: {}".format(fit_duration_exp))

(bin_x, bin_y), (x, y) = get_histogram_hybrid_bin(sizes)
ax1.plot(bin_x, bin_y, c=color, marker=marker, mfc="none", ms=7, lw=0)
cen_x, cen_y = np.sqrt(bin_x[np.argmin(abs(bin_x-xmin_s))]*bin_x[np.argmin(abs(bin_x-xmax_s))]), np.sqrt(bin_y[np.argmin(abs(bin_x-xmin_s))]*bin_y[np.argmin(abs(bin_x-xmax_s))])
fit_x = np.logspace(np.log10(xmin_s), np.log10(xmax_s))
fit_y = fit_x**(-fit_sizes_exp) * cen_y / cen_x**(-fit_sizes_exp)

(bin_x, bin_y), (x, y) = get_histogram_hybrid_bin(durations)
ax2.plot(bin_x, bin_y, c=color, marker=marker, mfc="none", ms=7, lw=0)
cen_x, cen_y = np.sqrt(bin_x[np.argmin(abs(bin_x-xmin_T))]*bin_x[np.argmin(abs(bin_x-xmax_T))]), np.sqrt(bin_y[np.argmin(abs(bin_x-xmin_T))]*bin_y[np.argmin(abs(bin_x-xmax_T))])
fit_x = np.logspace(np.log10(xmin_T), np.log10(xmax_T))
fit_y = fit_x**(-fit_duration_exp) * cen_y / cen_x**(-fit_duration_exp)

Tsort_ind = np.argsort(durations)
values, indices, counts = np.unique(durations[Tsort_ind], return_index=True, return_counts=True)
subarrays = np.split(sizes[Tsort_ind], indices)
bin_x, bin_y = get_histogram_hybrid_bin(values, _values=[a.mean() for a in subarrays[1:]])
ax3.plot(bin_x, bin_y, c=color, marker=marker, mfc="none", ms=7, lw=0)


""""""
t_interval = (0,20000)
param = (0.1, 0.6, 3, 1)
Delta_t = 115#0.115
xmin_s = 3
xmax_s = 100
xmin_T = 2
xmax_T = 30
marker = "o"
color = "b"

nd = NeuroData(os.path.join(data_path.SD_netA_mapout_gEgI, "{},{},{},{}".format(*param)))
nd.retain_dynamics(*t_interval)
qgraph.prob_dens_plot(nd.dynamics.interevent_intervals_insteps, 1, c=color, ax=ax0)

steps = np.hstack(nd.dynamics.spike_steps)-int(t_interval[0]/nd.configs["stepsize_ms"])
time_bins = np.arange(0, int((t_interval[1]-t_interval[0])/nd.configs["stepsize_ms"]), Delta_t)
counts = np.histogram(steps, time_bins)[0]
clusters = [cluster[1:] for cluster in np.split(counts, np.nonzero(counts==0)[0]) if cluster.sum()]
sizes = np.array([cluster.sum() for cluster in clusters])
durations = np.array([len(cluster) for cluster in clusters])

fit_sizes_exp = powerlaw.Fit(sizes, discrete=True, xmin=xmin_s, xmax=xmax_s).power_law.alpha
fit_duration_exp = powerlaw.Fit(durations, discrete=True, xmin=xmin_T, xmax=xmax_T).power_law.alpha
print("Fitted exponent for avalanche size: {}".format(fit_sizes_exp))
print("Fitted exponent for avalanche duration: {}".format(fit_duration_exp))

(bin_x, bin_y), (x, y) = get_histogram_hybrid_bin(sizes)
ax1.plot(bin_x, bin_y, c=color, marker=marker, mfc="none", ms=7, lw=0)
cen_x, cen_y = np.sqrt(bin_x[np.argmin(abs(bin_x-xmin_s))]*bin_x[np.argmin(abs(bin_x-xmax_s))]), np.sqrt(bin_y[np.argmin(abs(bin_x-xmin_s))]*bin_y[np.argmin(abs(bin_x-xmax_s))])
fit_x = np.logspace(np.log10(xmin_s), np.log10(xmax_s))
fit_y = fit_x**(-fit_sizes_exp) * cen_y / cen_x**(-fit_sizes_exp)

(bin_x, bin_y), (x, y) = get_histogram_hybrid_bin(durations)
ax2.plot(bin_x, bin_y, c=color, marker=marker, mfc="none", ms=7, lw=0)
cen_x, cen_y = np.sqrt(bin_x[np.argmin(abs(bin_x-xmin_T))]*bin_x[np.argmin(abs(bin_x-xmax_T))]), np.sqrt(bin_y[np.argmin(abs(bin_x-xmin_T))]*bin_y[np.argmin(abs(bin_x-xmax_T))])
fit_x = np.logspace(np.log10(xmin_T), np.log10(xmax_T))
fit_y = fit_x**(-fit_duration_exp) * cen_y / cen_x**(-fit_duration_exp)

Tsort_ind = np.argsort(durations)
values, indices, counts = np.unique(durations[Tsort_ind], return_index=True, return_counts=True)
subarrays = np.split(sizes[Tsort_ind], indices)
bin_x, bin_y = get_histogram_hybrid_bin(values, _values=[a.mean() for a in subarrays[1:]])
ax3.plot(bin_x, bin_y, c=color, marker=marker, mfc="none", ms=7, lw=0)


""""""
t_interval = (0,20000)
param = (0.3, 8, 3, 1)
Delta_t = 40
xmin_s = 3
xmax_s = 100
xmin_T = 2
xmax_T = 30
marker = "s"
color = "g"

nd = NeuroData(os.path.join(data_path.SD_netA_mapout_gEgI, "{},{},{},{}".format(*param)))
nd.retain_dynamics(*t_interval)
qgraph.prob_dens_plot(nd.dynamics.interevent_intervals_insteps, 1, c=color, ax=ax0)

steps = np.hstack(nd.dynamics.spike_steps)-int(t_interval[0]/nd.configs["stepsize_ms"])
time_bins = np.arange(0, int((t_interval[1]-t_interval[0])/nd.configs["stepsize_ms"]), Delta_t)
counts = np.histogram(steps, time_bins)[0]
clusters = [cluster[1:] for cluster in np.split(counts, np.nonzero(counts==0)[0]) if cluster.sum()]
sizes = np.array([cluster.sum() for cluster in clusters])
durations = np.array([len(cluster) for cluster in clusters])

fit_sizes_exp = powerlaw.Fit(sizes, discrete=True, xmin=xmin_s, xmax=xmax_s).power_law.alpha
fit_duration_exp = powerlaw.Fit(durations, discrete=True, xmin=xmin_T, xmax=xmax_T).power_law.alpha
print("Fitted exponent for avalanche size: {}".format(fit_sizes_exp))
print("Fitted exponent for avalanche duration: {}".format(fit_duration_exp))

(bin_x, bin_y), (x, y) = get_histogram_hybrid_bin(sizes)
ax1.plot(bin_x, bin_y, c=color, marker=marker, mfc="none", ms=7, lw=0)
cen_x, cen_y = np.sqrt(bin_x[np.argmin(abs(bin_x-xmin_s))]*bin_x[np.argmin(abs(bin_x-xmax_s))]), np.sqrt(bin_y[np.argmin(abs(bin_x-xmin_s))]*bin_y[np.argmin(abs(bin_x-xmax_s))])
fit_x = np.logspace(np.log10(xmin_s), np.log10(xmax_s))
fit_y = fit_x**(-fit_sizes_exp) * cen_y / cen_x**(-fit_sizes_exp)

(bin_x, bin_y), (x, y) = get_histogram_hybrid_bin(durations)
ax2.plot(bin_x, bin_y, c=color, marker=marker, mfc="none", ms=7, lw=0)
cen_x, cen_y = np.sqrt(bin_x[np.argmin(abs(bin_x-xmin_T))]*bin_x[np.argmin(abs(bin_x-xmax_T))]), np.sqrt(bin_y[np.argmin(abs(bin_x-xmin_T))]*bin_y[np.argmin(abs(bin_x-xmax_T))])
fit_x = np.logspace(np.log10(xmin_T), np.log10(xmax_T))
fit_y = fit_x**(-fit_duration_exp) * cen_y / cen_x**(-fit_duration_exp)

Tsort_ind = np.argsort(durations)
values, indices, counts = np.unique(durations[Tsort_ind], return_index=True, return_counts=True)
subarrays = np.split(sizes[Tsort_ind], indices)
bin_x, bin_y = get_histogram_hybrid_bin(values, _values=[a.mean() for a in subarrays[1:]])
ax3.plot(bin_x, bin_y, c=color, marker=marker, mfc="none", ms=7, lw=0)


""""""

[ax.set(xscale="log", yscale="log") for ax in [ax1,ax2,ax3]]
ax1.set(ylim=(1e-8,1e0), xticks=[1,10,100,1000,10000], yticks=[1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8])
ax2.set(ylim=(1e-7,1e0), xticks=[1,10,100,1000], yticks=[1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7])
ax3.set(ylim=(None,1e4), xlim=(None,1e3))

ax1.text(.85,.87, "(a)", transform=ax1.transAxes)
ax2.text(.85,.87, "(b)", transform=ax2.transAxes)
ax3.text(.05,.87, "(c)", transform=ax3.transAxes)

ax1.legend(["$(0.2,0.2)$","$(0.1,0.6)$","$(0.3,8)$"], title="$(g_E,g_I)$", fontsize=myFontSize2-2, loc="lower left")
ax3.legend(["$(0.2,0.2)$","$(0.1,0.6)$","$(0.3,8)$"], title="$(g_E,g_I)$", fontsize=myFontSize2-1, loc="lower right")

fig0.tight_layout()
fig.tight_layout()
figb.tight_layout()
# plt.show()
fig.savefig("fig_avalanchePsPT.pdf", dpi=400)
fig.savefig("fig_avalanchePsPT.png", dpi=400)
figb.savefig("fig_avalanchegamma.pdf", dpi=400)
figb.savefig("fig_avalanchegamma.png", dpi=400)