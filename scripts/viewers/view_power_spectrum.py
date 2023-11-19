from libs.mylib3 import power_spectral_density, SimulationData, graphing
from matplotlib import pyplot as plt
import numpy as np
import sys
try: data_directory = str(sys.argv[1])
except IndexError: print("missing required argument: \"data_directory\"\n> for example: /Users/data/output"); exit(1)

################################################################################

# data_directory = "/Users/likchun/NeuroProject/..."

################################################################################


transient_time_ms = 500
resolution_ms = 1

sd = SimulationData(data_directory,transient_time_ms=transient_time_ms)
freq,power = power_spectral_density(sd.dynamics.timedep_popul_firing_rate,1/.05,normalizedByTotalPower=True)

# intrinsic_freq = freq[np.argmax(power)]*1000 # unit: Hz
# print("intrinsic frequency: {} Hz".format(intrinsic_freq))

fig,[ax1,ax2,ax3] = plt.subplots(1,3,figsize=(14,6))
graphing.line_plot(freq*1000,power/1000,ax=ax1)
graphing.timedep_popul_firing_rate_gauskern(sd.dynamics.spike_train,sd.settings["duration_ms"]-transient_time_ms,sd.settings["dt_ms"],resolution_ms=resolution_ms,ax=ax2)
graphing.raster_plot_network(sd.dynamics.spike_times,ax=ax3)
ax1.set(xlim=(0,35),xlabel="frequency (Hz)",ylabel="normalized spectral power density")
ax2.set(xlim=(1,2),ylim=(0,None),xlabel="time (s)",ylabel="popul avg FR (Hz)")
ax3.set(xlim=(1,2),ylim=(0,1000),xlabel="time (s)",ylabel="neuron index")
plt.tight_layout()
plt.show()
