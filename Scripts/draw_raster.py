import data_path
from figure_style import *
from mylib import NeuroData, qgraph
from matplotlib import pyplot as plt
import numpy as np
import os


nd = NeuroData(os.path.join(data_path.spont_activ_netA, "0.2,0.2,3,1"))

# nd.remove_dynamics(500,0)
# nd.apply_neuron_mask([])
skip = 1

fig, axes = plt.subplots(2, 1, figsize=(12,6), gridspec_kw={"height_ratios":[1,.6]}, sharex=True)
qgraph.raster_plot(nd.dynamics.spike_times[::1], ax=axes[0], mec="none", ms=3, marker=".", time_scale="ms", colors=np.array(["navy" for _ in range(200)]+["orangered" for _ in range(800)])[::skip])
axes[1].plot(*nd.dynamics.average_firing_rate_time_histogram(5, time_scale="ms"), "k-")
axes[0].set_xlim(0, nd.configs["duration_ms"])
axes[0].set_ylim(.5, int(nd.configs["num_neuron"]/skip)+.5)
axes[1].set(xlabel="time $t$ (ms)", ylabel="$r(t)$ (Hz)")
axes[0].set(ylabel="neuron", yticks=[1,200,1000])
axes[0].tick_params(axis="x", which="both",length=0)

print("Coherence: C={}".format(nd.dynamics.analysis.coherence_parameter(32, True)))
print("Average firing rate: {} Hz".format(nd.dynamics.mean_firing_rate.mean()))

fig.tight_layout()
fig.subplots_adjust(hspace=.15)
plt.show()