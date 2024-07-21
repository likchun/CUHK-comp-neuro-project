from preamble import *


I_exts = [
    0.0, 3.5, 3.6,
    3.7, 3.71, 3.72, 3.73, 3.74, 3.75, 3.76, 3.77,
    3.771, 3.772, 3.773, 3.774, 3.775,
    3.78, 3.79,
    3.8, 3.81, 3.82, 3.83, 3.84, 3.85, 3.86,
    3.861, 3.862, 3.863, 3.864, 3.865,
    3.87, 3.88, 3.89,
    3.9, 3.91, 3.92, 3.93,
    3.94, 3.95, 3.96, 3.97, 3.98, 3.99,
    4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9,
    5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9,
    6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9,
    7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9,
    8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9,
    9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9,
    10.0,
]

fr_es, fr_is = [],[]
for Iext in I_exts:
    nd = NeuroData(os.path.join(data_path.SD_isolated_const, str(Iext)))
    nd.remove_dynamics(500,0)
    if len(nd.dynamics.interspike_intervals[0])!=0:
        fr_e = 1/np.mean(nd.dynamics.interspike_intervals[0])
    else: fr_e = 0
    if len(nd.dynamics.interspike_intervals[1])!=0:
        fr_i = 1/np.mean(nd.dynamics.interspike_intervals[1])
    else: fr_i = 0
    fr_es.append(fr_e)
    fr_is.append(fr_i)

fig, ax = plt.subplots(figsize=[6,5])
ax.plot(I_exts[:15], fr_es[:15], "k-",  lw=1)
ax.plot(I_exts[15:], fr_es[15:], "k-",  lw=1)
ax.plot(I_exts[:25], fr_is[:25], "k--", lw=1)
ax.plot(I_exts[25:], fr_is[25:], "k--", lw=1)
ax.set(xlabel="$I_{ext}$", ylabel="firing rate (Hz)")
ax.set(xlim=(0,10), xticks=[0,1,2,3,4,5,6,7,8,9,10])
ax.set(ylim=(-5,140), yticks=[0,20,40,60,80,100,120,140])

ax_in = ax.inset_axes([.11, .54, .3, .4], transform=ax.transAxes)
ax_in.plot(I_exts[:15], fr_es[:15], "k-",  lw=1)
ax_in.plot(I_exts[15:], fr_es[15:], "k-",  lw=1)
ax_in.plot(I_exts[:25], fr_is[:25], "k--", lw=1)
ax_in.plot(I_exts[25:], fr_is[25:], "k--", lw=1)
ax_in.set(xlim=(3.7,3.9))
ax_in.set(ylim=(-1,25), yticks=[0,5,10,15,20,25])

fig.tight_layout()
# plt.show()
fig.savefig("fig_frIext.pdf")
fig.savefig("fig_frIext.png")