from preamble import *
from mylib import load_spike_steps


gs = [0,.01,.02,.03,.04,.05,.055,.06,.07,.08,.09,.1,.11,.12,.13,.14,.15,.16,.17,.18,.19,.2]
alpha = 3


spike_steps_EE = [load_spike_steps(os.path.join(data_path.SD_state1_pXY, "pEE/pEE_{}_{}".format(g,alpha))) for g in gs]
spike_steps_IE = [load_spike_steps(os.path.join(data_path.SD_state1_pXY, "pIE/pIE_{}_{}".format(g,alpha))) for g in gs]
Ks_EE = [np.hstack(ss).shape[0] for ss in spike_steps_EE]
Ks_IE = [np.hstack(ss).shape[0] for ss in spike_steps_IE]
ps_EE = [(Ks_EE[i]-Ks_EE[0])/10000 for i in range(len(Ks_EE))]
ps_IE = [(Ks_IE[i]-Ks_IE[0])/10000 for i in range(len(Ks_IE))]

x = np.interp(0.15, ps_EE, gs)
y = np.interp(x, gs, ps_IE)

fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(8,4.5))
ax1.plot(gs, ps_EE, "k^-",  ms=5, label="$\\rho_{{EE}}$")
ax1.plot(gs, ps_IE, "ko--", ms=5, label="$\\rho_{{IE}}$")
ax1.set(xlabel="$g_E$", ylabel="$\\rho_{XY}$", xlim=(0,gs[-1]), ylim=(0,1.25), xticks=[0,.05,.1,.15,.2], xticklabels=["0","0.05","0.1","0.15","0.2"])
# ax1.grid(True)
ax1.legend()


gs = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3]
alpha = 3

spike_steps_EI_A = [load_spike_steps(os.path.join(data_path.SD_state1_pXY, "pEI_A/pEI_{}_{}".format(g,alpha))) for g in gs]
spike_steps_EI_B = [load_spike_steps(os.path.join(data_path.SD_state1_pXY, "pEI_B/pEI_{}_{}".format(g,alpha))) for g in gs]
Ks_EI_A = [np.hstack(ss).shape[0] for ss in spike_steps_EI_A]
Ks_EI_B = [np.hstack(ss).shape[0] for ss in spike_steps_EI_B]
ps_EI_A = [(Ks_EI_A[i]-Ks_EI_A[0])/100000 for i in range(len(Ks_EI_A))]
ps_EI_B = [(Ks_EI_B[i]-Ks_EI_B[0])/100000 for i in range(len(Ks_EI_B))]
ps_EI = (np.array(ps_EI_A)+np.array(ps_EI_B))/2

spike_steps_II_A = [load_spike_steps(os.path.join(data_path.SD_state1_pXY, "pII_A/pII_{}_{}".format(g,alpha))) for g in gs]
spike_steps_II_B = [load_spike_steps(os.path.join(data_path.SD_state1_pXY, "pII_B/pII_{}_{}".format(g,alpha))) for g in gs]
Ks_II_A = [np.hstack(ss).shape[0] for ss in spike_steps_II_A]
Ks_II_B = [np.hstack(ss).shape[0] for ss in spike_steps_II_B]
ps_II_A = [(Ks_II_A[i]-Ks_II_A[0])/100000 for i in range(len(Ks_II_A))]
ps_II_B = [(Ks_II_B[i]-Ks_II_B[0])/100000 for i in range(len(Ks_II_B))]
ps_II = (np.array(ps_II_A)+np.array(ps_II_B))/2

ax2.plot(gs, ps_EI, "k^-",  ms=5, label="$\\rho_{{EI}}$")
ax2.plot(gs, ps_II, "ko--", ms=5, label="$\\rho_{{II}}$")
ax2.set(xlabel="$g_I$", xlim=(0,1), ylim=(-.008,0), xticks=[0,.25,.5,.75,1], xticklabels=["0","0.25","0.5","0.75","1"])
# ax2.grid(True)
ax2.legend()

ax1.text(0,1.1, "(a)", transform=ax1.transAxes)
ax2.text(0,1.1, "(b)", transform=ax2.transAxes)

fig.tight_layout()
fig.subplots_adjust(wspace=.4)
# plt.show()
fig.savefig("fig_rhogEI.pdf", dpi=400)
fig.savefig("fig_rhogEI.png", dpi=400)