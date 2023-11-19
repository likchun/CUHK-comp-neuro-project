from libs.mylib3 import SimulationData
from libs.avalanchelib import AvalancheTool
from matplotlib import pyplot as plt
import numpy as np
import powerlaw
import sys
avaltool = AvalancheTool()
try: data_directory = str(sys.argv[1])
except IndexError: print("missing required argument: \"data_directory\"\n> for example: /Users/data/output"); exit(1)

################################################################################

# data_directory = "/Users/likchun/NeuroProject/..."

################################################################################

# time_bin_width = 1 # in unit of simulation time step dt
xmin_s, xmax_s = 1, 100
xmin_T, xmax_T = 1, 10


# find avalanche sizes and duration #
sd = SimulationData(data_directory)
time_bin_width = max(np.hstack(sd.dynamics.interspike_intervals).mean()/sd.settings["dt_ms"]/2, 1) # in unit of simulation time step dt
sizes, durations = avaltool.get_avalanche_sizes_and_durations(steps_list=sd.dynamics.spike_steps, num_time_step=sd.settings["duration_ms"]/sd.settings["dt_ms"], time_bin_width=time_bin_width)
# areas = get_avalanche_areas(steps_list=sd.dynamics.spike_steps, num_time_step=sd.settings["duration_ms"]/sd.settings["dt_ms"], time_bin_width=time_bin_width)
fit_sizes_exp = powerlaw.Fit(sizes, discrete=True, xmin=xmin_s, xmax=xmax_s).power_law.alpha
fit_duration_exp = powerlaw.Fit(durations, discrete=True, xmin=xmin_T, xmax=xmax_T).power_law.alpha
print("Fitted exponent for avalanche size: {}".format(fit_sizes_exp))
print("Fitted exponent for avalanche duration: {}".format(fit_duration_exp))


# figures #
fig, [ax1,ax2] = plt.subplots(1,2,figsize=(10,5))

(bin_x, bin_y), (x, y) = avaltool.get_histogram_hybrid_bin(sizes)
ax1.plot(x, y, c="b", marker="o", ms=1, lw=0)
ax1.plot(bin_x, bin_y, c="k", marker="o", mfc="none", ms=10, lw=0)
cen_x, cen_y = np.sqrt(bin_x[np.argmin(abs(bin_x-xmin_s))]*bin_x[np.argmin(abs(bin_x-xmax_s))]), np.sqrt(bin_y[np.argmin(abs(bin_x-xmin_s))]*bin_y[np.argmin(abs(bin_x-xmax_s))])
fit_x = np.logspace(np.log10(xmin_s), np.log10(xmax_s))
fit_y = fit_x**(-fit_sizes_exp) * cen_y / cen_x**(-fit_sizes_exp)
ax1.plot(fit_x, fit_y, c="r", lw=1)
ax1.set(xscale="log",yscale="log")
ax1.set(xlabel="number of spikes, s", ylabel="probability density")

(bin_x, bin_y), (x, y) = avaltool.get_histogram_hybrid_bin(durations)
ax2.plot(x, y, c="b", marker="o", ms=1, lw=0)
ax2.plot(bin_x, bin_y, c="k", marker="o", mfc="none", ms=10, lw=0)
cen_x, cen_y = np.sqrt(bin_x[np.argmin(abs(bin_x-xmin_T))]*bin_x[np.argmin(abs(bin_x-xmin_T))]), np.sqrt(bin_y[np.argmin(abs(bin_x-xmin_T))]*bin_y[np.argmin(abs(bin_x-xmax_T))])
fit_x = np.logspace(np.log10(xmin_T), np.log10(xmax_T))
fit_y = fit_x**(-fit_duration_exp) * cen_y / cen_x**(-fit_duration_exp)
ax2.plot(fit_x, fit_y, c="r", lw=1)
ax2.set(xscale="log",yscale="log")
ax2.set(xlabel="number of bins, T")#, ylabel="probability density")

# x, y = get_histogram_hybrid_bin(areas)
# ax3.plot(x, y, c="k", mec="k", marker="o", lw=0, ms=5, mfc="w")
# ax3.set(xscale="log",yscale="log")
# ax3.set(xlabel="number of neurons, a", ylabel="probability density")

plt.tight_layout()
plt.show()
