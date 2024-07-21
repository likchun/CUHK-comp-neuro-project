from preamble import *


nd = NeuroData(os.path.join(data_path.SD_isolated_const, "4.0"))
v_e = nd.dynamics.time_series.membrane_potential[0]
u_e = nd.dynamics.time_series.recovery_variable[0]
v_i = nd.dynamics.time_series.membrane_potential[1]
u_i = nd.dynamics.time_series.recovery_variable[1]

fig, [ax1,ax2] = plt.subplots(2, 1, figsize=[9,5], sharex=True, sharey=True)
axtw1 = ax1.twinx()
ax1.plot(np.arange(len(v_e))*nd.configs["stepsize_ms"], v_e, "k-",  lw=1)
axtw1.plot(np.arange(len(u_e))*nd.configs["stepsize_ms"], u_e, "g--",  lw=1)
ax1.set(xlim=(200,1000))
ax1.set(ylim=(-90,30), yticks=[-90,-65,-30,0,30])
axtw1.set(ylim=(-16,0), yticks=[-16,-12,-8,-4,0])
ax1.set(ylabel="$v$ (mV)")
axtw1.set(ylabel="$u$")

axtw2 = ax2.twinx()
ax2.plot(np.arange(len(v_i))*nd.configs["stepsize_ms"], v_i, "k-",  lw=1)
axtw2.plot(np.arange(len(u_i))*nd.configs["stepsize_ms"], u_i, "g--",  lw=1)
axtw2.set(ylim=(-16,0), yticks=[-16,-12,-8,-4,0])
ax2.set(xlabel="$t$ (ms)", ylabel="$v$ (mV)")
axtw2.set(ylabel="$u$")

fig.tight_layout()
# plt.show()
fig.savefig("fig_vuseries.pdf")
fig.savefig("fig_vuseries.png")
