from mylib3 import *
from plvlib import *
plv = PhaseLockingValue()
graphing.modern_style()
graphing.default_legend_style()


w_inh = .2
w_exc = [.04,.08,.2,.6]
alpha = 5

transient_time_ms = 300


directory0 = "/Users/likchun/NeuroProject/raw_data/networkB[EI4]_voltage/"
sds = [SimulationData(directory0+"a{2}/matrixB_wi{0}-we{1}_a{2}".format(w_inh,w_e,alpha),transient_time_ms=transient_time_ms) for w_e in w_exc]

fig, axes = plt.subplots(4,3,figsize=(12,10),sharex="col",sharey=False)
### raster plots ###
[graphing.raster_plot_network(sd.dynamics.spike_times[::20],ax=axes[i][0]) for i,sd in enumerate(sds)]
### ISI distributions ###
graphing.distribution_density_plot(sds[0].dynamics.interspike_intervals[200:],c="k",binsize=.05,xlogscale=True,ax=axes[0][1])
graphing.distribution_density_plot(sds[1].dynamics.interspike_intervals[200:],c="k",binsize=.02,xlogscale=True,ax=axes[1][1])
graphing.distribution_density_plot(sds[2].dynamics.interspike_intervals[200:],c="k",binsize=.02,xlogscale=True,ax=axes[2][1])
graphing.distribution_density_plot(sds[3].dynamics.interspike_intervals[200:],c="k",binsize=.02,xlogscale=True,ax=axes[3][1])
### timedep popul FR ###
[graphing.timedep_popul_firing_rate_gauskern(sd.dynamics.spike_train,sd.settings["duration_ms"],sd.settings["dt_ms"],start_t=transient_time_ms,ax=axes[i][2]) for i,sd in enumerate(sds)]

axes[-1][0].set(xlabel="time (s)")
axes[-1][1].set(xlabel="ISI (s)")
axes[-1][2].set(xlabel="time (s)")
[axes[i][0].set(ylabel="50 neurons",yticks=[]) for i in range(len(sds))]
[axes[i][1].set(ylabel="prob. density") for i in range(len(sds))]
[axes[i][2].set(ylabel="popul. FR") for i in range(len(sds))]

[axes[i][0].set(xlim=(.5,1.),ylim=(0,50)) for i in range(len(sds))]
# [axes[i][1].set(xlim=(6e-4,1)) for i in range(len(sds))]
[axes[i][1].set(xscale="log",yscale="log") for i in range(len(sds))]
axes[0][0].set_xticks([.5,.75,1.])
axes[0][2].set_xticks([.5,.75,1.])
axes[0][2].set(ylim=(0,20),xlim=(.5,1.))
axes[1][2].set(ylim=(0,100),xlim=(.5,1.))
axes[2][2].set(ylim=(0,800),xlim=(.5,1.))
axes[3][2].set(ylim=(0,300),xlim=(.5,1.))
# axes[0].plot([.750,.880],[270,270],"k.-")
# axes[0].text(.815,271,"0.13 s",fontsize=10,horizontalalignment="center",verticalalignment="bottom")
# axes[1].text(.13,1.5e6,"0.13 s",fontsize=10,horizontalalignment="center",verticalalignment="bottom")

plt.tight_layout()
# plt.show()
fig.savefig("fig_dynamics_details",dpi=400)

# fig.subplots_adjust(wspace=.3,hspace=.4)
# fig.savefig("fig_dynamics_details",bbox_inches="tight",pad_inches=.5,dpi=500)
