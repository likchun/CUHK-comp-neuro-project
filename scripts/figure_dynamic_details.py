from libs.mylib3 import SimulationData, graphing, load_stimulus
from libs.plvlib import PLVtool
from libs.avalanchelib import AvalancheTool
from matplotlib import pyplot as plt
import numpy as np
import os
plvtool = PLVtool()
avaltool = AvalancheTool()


### Setting & Parameters ###

config_a = (0.05,0.2,3)
config_b = (0,5,0.2)

# set the fitting ranges for avalanche distributions
xmin_s, xmax_s = 2, 50
xmin_T, xmax_T = 2, 10

# set the time window for bumped-up version of the raster plot
bumpup_raster_cen_t = 2500 # ms
bumpup_raster_len_t = 500 # ms

# set transient dynamic time
transient_time_ms = 1000

# paths to data files
if config_b[0]==0: data_directory = "/Users/likchun/NeuroProject/raw_data/net_unifindeg_constwij/spontaneous/{0},{1},{2}".format(*config_a)
else: data_directory = "/Users/likchun/NeuroProject/raw_data/net_unifindeg_constwij/stimulus_evoked/stim_p1b[A{0},f{1},m{2}]/{3},{4},{5}".format(*config_b,*config_a)
stimuli_directory = "/Users/likchun/NeuroProject/CUHK-comp-neuro-project/stimuli"

#############################

sd = SimulationData(data_directory,transient_time_ms)
gplv,pwplv,_ = plvtool.load_plv_data(os.path.join(sd.data_path,"plv_data"))
isi = np.hstack(sd.dynamics.interspike_intervals)
fr = sd.dynamics.mean_firing_rate
# exc_idx,inh_idx = np.argwhere(sd.network.neuron_type=="exc"),np.argwhere(sd.network.neuron_type=="inh")
# isi_e,isi_i = np.hstack(sd.dynamics.interspike_intervals[exc_idx]),np.hstack(sd.dynamics.interspike_intervals[inh_idx])
# fr_e,fr_i = sd.dynamics.mean_firing_rate[exc_idx],sd.dynamics.mean_firing_rate[inh_idx]
print("mean ISI: {:.5f} ms, {:.3f} time step".format(np.hstack(sd.dynamics.interspike_intervals).mean()*1000,np.hstack(sd.dynamics.interspike_intervals).mean()/sd.settings["stepsize_ms"]))

if sd.settings["stimulus_file"] != "none":
    stim_nidx,stim_series,stim_info = load_stimulus(os.path.join(stimuli_directory,sd.settings["stimulus_file"]),returnInfo=True)
    isSimulated = np.full(sd.settings["num_neuron"],False)
    isSimulated[stim_nidx] = True

# timebin_width = 1 # in unit of simulation time step dt
timebin_width = max(np.hstack(sd.dynamics.interspike_intervals).mean()/sd.settings["stepsize_ms"]/2, 1) # in unit of simulation time step dt
print("Avalanche time bin width used: {:.3f} (in unit of dt={}ms)".format(timebin_width,sd.settings["stepsize_ms"]))
avalanche_sizes, avalanche_durations = avaltool.get_avalanche_sizes_and_durations(sd.dynamics.spike_steps,sd.settings["duration_ms"]/sd.settings["stepsize_ms"],timebin_width)
print("Number of avalanche detected: {}".format(avalanche_sizes.shape[0]))
fit_avalsizes_exp = avaltool.fit_powerlaw_expoenent(avalanche_sizes,xmin_s,xmax_s)
print("Avalanche size fitted exponent: {:.5f}".format(fit_avalsizes_exp))
fit_avaldurations_exp = avaltool.fit_powerlaw_expoenent(avalanche_durations,xmin_T,xmax_T)
print("Avalanche duration fitted exponent: {:.5f}".format(fit_avaldurations_exp))

#############################

raster_colors = np.full(sd.settings["num_neuron"],"black")
if sd.settings["stimulus_file"] != "none":
    raster_colors[np.argwhere(isSimulated==True)] = "blue"
    my_sorting_ind = np.lexsort((sd.dynamics.mean_firing_rate,isSimulated,sd.network.neuron_type))
else: my_sorting_ind = np.lexsort((sd.dynamics.mean_firing_rate,sd.network.neuron_type))
spike_times = sd.dynamics.spike_times[my_sorting_ind][::-1]
raster_colors = list(raster_colors[my_sorting_ind][::-1])

#############################

fig, axes = plt.subplots(3,3,figsize=(13,8),gridspec_kw={"width_ratios":[1.5,1,1]})
# ei_color = dict(e=(.5,.1,0),i=(0,.1,.5))

graphing.raster_plot_network(spike_times,alpha=.8,color=raster_colors,ax=axes[0][0])
axes[0][0].set(xlim=(2,3),ylim=(0,sd.settings["num_neuron"]))
axes[0][0].set(xlabel="time (s)",ylabel="all neurons",yticks=[1,200,1000],xticks=[2,2.5,3])
# graphing.raster_plot_network(spike_times[::20]*1000,alpha=.8,color=raster_colors[::20],ax=axes[0][1])
graphing.event_plot_neurons(spike_times[::40]*1000,ax=axes[0][1])
axes[0][1].set(xlim=(bumpup_raster_cen_t-bumpup_raster_len_t,bumpup_raster_cen_t+bumpup_raster_len_t),ylim=(.5,25.5))
axes[0][1].set(xlabel="time (ms)",ylabel="25 neurons",yticks=[1,25])

graphing.timedep_popul_firing_rate_binned(sd.dynamics.spike_times,5.,ax=axes[1][0])
graphing.timedep_popul_firing_rate_gauskern(sd.dynamics.spike_train,sd.settings["duration_ms"],sd.settings["stepsize_ms"],1.,start_t=1000,color="b",ax=axes[1][0],linestyle="--",lw=.5)
if sd.settings["stimulus_file"] != "none":
    ax10_twinx = axes[1][0].twinx()
    ax10_twinx.yaxis.label.set_color("r")
    ax10_twinx.tick_params(axis="y", colors="r")
    ax10_twinx.plot(np.arange(len(stim_series))*sd.settings["stepsize_ms"]/1000,stim_series,":",c="r")
axes[1][0].set(xlim=(2,3),ylim=(None,None))
axes[1][0].set(xlabel="time (s)",ylabel="avg FR (Hz)",xticks=[2,2.5,3])

graphing.distribution_density_plot(fr,(np.amax(fr)-np.amin(fr))/11,ax=axes[1][1])
# graphing.distribution_density_plot(fr,(np.amax(fr)-np.amin(fr))/60,ax=axes[1][1])
axes[1][1].set(xlim=(0,None))
axes[1][1].set(xlabel="FR (Hz)",ylabel="P(FR)")

graphing.distribution_density_plot(np.log10(isi),.02,ax=axes[1][2])
axes[1][2].set(xlabel="$log_{{10}}$[ISI (s)]",ylabel="P($log_{{10}}$[ISI])")

graphing.distribution_density_plot(pwplv,.005,ax=axes[0][2])
axes[0][2].set(xlim=(0,1),ylim=(None,None))
axes[0][2].set(xlabel="PLV",ylabel="P(PLV)")

(bin_x, bin_y), (x, y) = avaltool.get_histogram_hybrid_bin(avalanche_sizes)
axes[2][1].plot(x,y,c="b",marker="o",ms=1,lw=0,alpha=.5)
axes[2][1].plot(bin_x,bin_y,c="k",marker="o",mfc="none",ms=10,lw=0)
# axes[2][1].plot(x,y/x**fit_avalsizes_exp,c="b",marker="o",ms=1,lw=0,alpha=.5) # compensated plot
# axes[2][1].plot(bin_x,bin_y/bin_x**fit_avalsizes_exp,c="k",marker="o",mfc="none",ms=10,lw=0) # compensated plot
cen_x, cen_y = np.sqrt(bin_x[np.argmin(abs(bin_x-xmin_s))]*bin_x[np.argmin(abs(bin_x-xmax_s))]), np.sqrt(bin_y[np.argmin(abs(bin_x-xmin_s))]*bin_y[np.argmin(abs(bin_x-xmax_s))])
fit_x = np.logspace(np.log10(xmin_s), np.log10(xmax_s))
fit_y = fit_x**(-fit_avalsizes_exp) * cen_y / cen_x**(-fit_avalsizes_exp)
# axes[2][1].plot(fit_x, fit_y, c="r", lw=2) # fitting line
# axes[2][1].plot(fit_x, fit_y/fit_x**fit_avalsizes_exp, c="r", lw=1) # fitting line # compensated plot
axes[2][1].set(xscale="log",yscale="log")
axes[2][1].set(xlabel="s",ylabel="P(s)")


(bin_x, bin_y), (x, y) = avaltool.get_histogram_hybrid_bin(avalanche_durations)
axes[2][2].plot(x,y,c="b",marker="o",ms=1,lw=0,alpha=.5)
axes[2][2].plot(bin_x,bin_y,c="k",marker="o",mfc="none",ms=10,lw=0)
cen_x, cen_y = np.sqrt(bin_x[np.argmin(abs(bin_x-xmin_T))]*bin_x[np.argmin(abs(bin_x-xmax_T))]), np.sqrt(bin_y[np.argmin(abs(bin_x-xmin_T))]*bin_y[np.argmin(abs(bin_x-xmax_T))])
fit_x = np.logspace(np.log10(xmin_T), np.log10(xmax_T))
fit_y = fit_x**(-fit_avaldurations_exp) * cen_y / cen_x**(-fit_avaldurations_exp)
# axes[2][2].plot(fit_x, fit_y, c="r", lw=2) # fitting line
axes[2][2].set(xscale="log",yscale="log")
axes[2][2].set(xlabel="T",ylabel="P(T)")

sp_x,sp_y,_ = graphing.power_spectrum(sd.dynamics.timedep_popul_firing_rate,sd.settings["stepsize_ms"],duration_ms=sd.settings["duration_ms"],normalizedByTotalPower=True,ax=axes[2][0],c="k")
axes[2][0].set(xlim=(0,30),ylabel=(None,None))
axes[2][0].set(xlabel="f (Hz)",ylabel="P(f)")
axes[2][0].ticklabel_format(axis="y", style="scientific", scilimits=(-3,3), useOffset=False, useLocale=False, useMathText=True)
if sd.settings["stimulus_file"] != "none":
    axes[2][0].plot([stim_info[5],stim_info[5]],[0,np.max(sp_y)],":",c="r")

plt.tight_layout()
plt.show()
# fig.savefig("FIG ({0},{1},{2}) [{3},{4},{5}].png".format(*config_a,*config_b),dpi=300)