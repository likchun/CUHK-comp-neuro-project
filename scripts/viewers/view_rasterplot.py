from libs.mylib3 import SimulationData, graphing
from matplotlib import pyplot as plt
import sys
try: data_directory = str(sys.argv[1])
except IndexError: print("missing required argument: \"data_directory\"\n> for example: /Users/data/output"); exit(1)

################################################################################

# data_directory = "/Users/likchun/NeuroProject/..."

################################################################################


sd = SimulationData(data_directory,transient_time_ms=000)
fig, [ax1,ax2] = plt.subplots(2,1,figsize=(12,7),sharex=True,sharey=True)
graphing.raster_plot_network(sd.dynamics.spike_times,ax=ax1)
graphing.raster_plot_network(sd.dynamics.spike_times,sort_by_spike_count=True,separate_neuron_type=sd.network.neuron_type,ax=ax2)
ax1.set(xlim=(0,sd.settings["duration_ms"]/1000),ylim=(1,sd.settings["neuron_num"]))
ax1.set(ylabel="neuron index")
ax2.set(xlabel="time (s)",ylabel="neuron index")
plt.tight_layout()
plt.show()

# fig.savefig("raster_plot",dpi=300)