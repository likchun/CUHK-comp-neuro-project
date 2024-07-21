from preamble import *
from mylib import power_spectral_density_normalized
from scipy import ndimage


nd = NeuroData(os.path.join(data_path.SD_netA_mapout_gEgI, "{},{},{},{}".format(0.3,0.6,3,1)))
nd.remove_dynamics(500,0)

r_t = nd.dynamics.average_firing_rate_time_histogram(5, time_scale="s")[1]
# r_t = ndimage.gaussian_filter1d(r_t, 2)

freq,spow = power_spectral_density_normalized(r_t, 1000/5)
spow_smooth = ndimage.gaussian_filter1d(spow, 5)

fosc = freq[np.argmax(spow_smooth)]
print("network oscillation frequency: f_osc = {:.3f} Hz".format(fosc))

fig, ax = plt.subplots(figsize=[7,5])
ax.plot(freq, spow, "k-", lw=1)
ax.plot(freq, spow_smooth, "r-", lw=1.5)
ax.plot([fosc,fosc], [0,np.amax(spow)], "c-", lw=1)
ax.set(xlim=(0,40))
# ax.legend()
ax.set(xlabel="$f$ (Hz)", ylabel="$S_{rr}$")
fig.tight_layout()

plt.show()