from preamble import *
from mylib import err_prop_2arr_muldiv, get_spike_count, power_spectral_density_normalized
from scipy import signal, stats, ndimage


gIs = [0,.2,.4,.6,1,2,4,6,8]
gE = 0.2

results = np.load(os.path.join(data_path.DA_state2_netburst, "data_netburstInh,gE02.npy"), allow_pickle=True)

mean_fr = np.array(results[0], dtype=float)
num_netburst = np.array(results[1], dtype=int)
freq_nb = np.array(results[2], dtype=float)
mean_INBI_ms = np.array(results[3], dtype=float)
std_INBI_ms = np.array(results[4], dtype=float)
mean_K_nb = np.array(results[5], dtype=float)
std_K_nb = np.array(results[6], dtype=float)
mean_Nactive_nb_0 = np.array(results[7], dtype=float)
std_Nactive_nb_0 = np.array(results[8], dtype=float)
mean_Nactive_nb_1 = np.array(results[9], dtype=float)
std_Nactive_nb_1 = np.array(results[10], dtype=float)
mean_Nactive_nb_2 = np.array(results[11], dtype=float)
std_Nactive_nb_2 = np.array(results[12], dtype=float)
mean_T_nb_03 = np.array(results[13], dtype=float)
std_T_nb_03 = np.array(results[14], dtype=float)
mean_T_nb_04 = np.array(results[15], dtype=float)
std_T_nb_04 = np.array(results[16], dtype=float)
mean_T_nb_05 = np.array(results[17], dtype=float)
std_T_nb_05 = np.array(results[18], dtype=float)


fig, axes = plt.subplots(2, 2, figsize=[9,9])

axes[0][0].plot(gIs, freq_nb, "^-", c="navy", mfc="navy", ms=myMarkerSize0)
axes[0][0].set(ylabel="$f_{nb}$ or $f_{osc}$ (Hz)", ylim=(0,16), yticks=[0,2,4,6,8,10,12,14,16])
axes[0][0].set(xlabel="$g_I$", xlim=(8*-0.03, 8*1.03), xticks=[0,1,2,4,6,8])

axes[1][0].errorbar(gIs, mean_Nactive_nb_0/1000, std_Nactive_nb_0/1000, fmt="o-", c="navy", ms=myMarkerSize0, capsize=4, label="0")
axes[1][0].set(ylabel="participation frac.", ylim=(0,1111/1000), yticks=[0,.2,.4,.6,.8,1])
axes[1][0].set(xlabel="$g_I$", xlim=(8*-0.03, 8*1.03), xticks=[0,1,2,4,6,8])

axes[0][1].errorbar(gIs, mean_INBI_ms, std_INBI_ms, fmt="o-", c="navy", ms=myMarkerSize0, capsize=4, label="interval")
axes[0][1].plot(gIs, mean_T_nb_04, "x:", c="navy", ms=myMarkerSize0, label="$w_{nb}$")
axes[0][1].set(xlabel="$g_I$", ylabel="INBI (ms)")
axes[0][1].set(xlim=(8*-0.03, 8*1.03), xticks=[0,1,2,4,6,8], ylim=(0,300), yticks=[0,50,100,150,200,250,300])
axes[0][1].legend()


binsize_ms = 5

params = [(gE,gI,3,1) for gI in gIs]
nds = [NeuroData(os.path.join(data_path.SD_netA_mapout_gEgI, "{},{},{},{}".format(*par))) for par in params]
[nd.remove_dynamics(500,0) for nd in nds]

foscs = []
for i,nd in enumerate(nds):
    r_t = nd.dynamics.average_firing_rate_time_histogram(5, time_scale="s")[1]
    freq,spow = power_spectral_density_normalized(r_t, 1000/5)
    spow_smooth = ndimage.gaussian_filter1d(spow, 5)
    foscs.append(freq[np.argmax(spow_smooth)])
axes[0][0].plot(gIs, foscs, "s--", c="navy", mfc="none", ms=myMarkerSize0)
axes[0][0].legend(["$f_{nb}$", "$f_{osc}$"], loc="lower right")


gIs = [.2,.4,.6,2,8]
params = [(gE,gI,3,1) for gI in gIs]
bin_l_ms = 35 # should be integral multiple of binsize_ms
bin_r_ms = 65 # should be integral multiple of binsize_ms

nds = [NeuroData(os.path.join(data_path.SD_netA_mapout_gEgI, "{},{},{},{}".format(*par))) for par in params]
[nd.remove_dynamics(500,0) for nd in nds]

num_spk_dist_e, num_spk_dist_i = [], []
neuronfr_populevent = []
for i,nd in enumerate(nds):
    avgfr = nd.dynamics.average_firing_rate_time_histogram(binsize_ms)
    pinds, props = signal.find_peaks(avgfr[1], prominence=10)#, distance=100/binsize_ms)
    vinds = np.array((pinds[1:]+pinds[:-1])/2).astype(int)
    pinds = pinds[1:-1] # exclude the first and the last peaks
    pinds_instepsize = pinds * binsize_ms/nd.configs["stepsize_ms"]
    vinds_instepsize = vinds * binsize_ms/nd.configs["stepsize_ms"]

    splitinds = vinds_instepsize
    splited_spikes = [[
        spike_steps[np.argwhere((splitinds[i] < spike_steps) & (spike_steps <= splitinds[i+1]))]
            for spike_steps in nd.dynamics.spike_steps]
                for i in range(len(splitinds)-1)]

    num_spike_netburst_e = np.array([get_spike_count(ss)[200:1000] for ss in splited_spikes]).flatten()
    num_spike_netburst_i = np.array([get_spike_count(ss)[:200] for ss in splited_spikes]).flatten()

    x_e, y_e = qgraph.bar_chart_INT(num_spike_netburst_e, ax=None)
    x_i, y_i = qgraph.bar_chart_INT(num_spike_netburst_i, ax=None)
    y_e = y_e/np.sum(y_e)
    y_i = y_i/np.sum(y_i)

    num_spk_dist_e.append((x_e, y_e))
    num_spk_dist_i.append((x_i, y_i))
num_spk_dist_e = np.array(num_spk_dist_e, dtype=object)
num_spk_dist_i = np.array(num_spk_dist_i, dtype=object)

fmts = ["kD-", "rs-", "mo-", "bp-", "g^-"]

[axes[1][1].plot(*xy, fmt, ms=6, lw=1) for xy,fmt in zip(num_spk_dist_e, fmts)]
axes[1][1].set(xlim=(18*-.02,18), xticks=[0,3,6,9,12,15,18],
               ylim=(.5*-.03,.5), yticks=[0,.1,.2,.3,.4,.5])
axes[1][1].legend(["0.2","0.4","0.6","2.0","8.0"], title="$g_I$")
axes[1][1].set(xlabel="spike count", ylabel="$P$(spike count)")

axes[0][0].text(-.25,1.1, "(a)", transform=axes[0][0].transAxes)
axes[0][1].text(-.25,1.1, "(b)", transform=axes[0][1].transAxes)
fig.tight_layout()
axes[1][0].text(-.25,1.1, "(c)", transform=axes[1][0].transAxes)
axes[1][1].text(-.25,1.1, "(d)", transform=axes[1][1].transAxes)

# plt.show()
fig.savefig("fig_netburstInh.pdf", dpi=400)
fig.savefig("fig_netburstInh.png", dpi=400)