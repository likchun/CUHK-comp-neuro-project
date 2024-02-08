import os
import sys
this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(this_dir,".."))
from libs.mylib4 import NeuroData, qgraph
from matplotlib import pyplot as plt
try: data_directory = str(sys.argv[1])
except IndexError: print("missing required argument: \"data_directory\"\n> for example: /Users/data/output"); exit(1)

################################################################################

directory = "/Users/likchun/NeuroProject/..."

################################################################################


nd = NeuroData(directory)
fig, ax = plt.subplots(figsize=(10,5))
qgraph.raster_plot(nd.dynamics.spike_times,ax=ax)
ax.set(xlim=(0,nd.configs["duration_ms"]/1000),ylim=(1,nd.configs["num_neuron"]))
ax.set(ylabel="neuron",xlabel="time (s)")
plt.tight_layout()
plt.show()

# fig.savefig("raster_plot",dpi=300)