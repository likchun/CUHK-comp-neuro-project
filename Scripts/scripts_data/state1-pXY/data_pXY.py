from mylib import load_spike_steps
from preamble import *


alpha = 3

# import data, exc
gEs = [0,.01,.02,.03,.04,.05,.055,.06,.07,.08,.09,.1,.11,.12,.13,.14,.15,.16,.17,.18,.19,.2]

spike_steps_EE = [load_spike_steps(os.path.join(data_path.SD_state1_pXY, "pEE/pEE_{}_{}".format(g,alpha))) for g in gEs]
spike_steps_IE = [load_spike_steps(os.path.join(data_path.SD_state1_pXY, "pIE/pIE_{}_{}".format(g,alpha))) for g in gEs]

Ks_EE = [np.hstack(ss).shape[0] for ss in spike_steps_EE]
Ks_IE = [np.hstack(ss).shape[0] for ss in spike_steps_IE]

ps_EE = [(Ks_EE[i]-Ks_EE[0])/10000 for i in range(len(Ks_EE))]
ps_IE = [(Ks_IE[i]-Ks_IE[0])/10000 for i in range(len(Ks_IE))]


# import data, inh
gIs = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3]

spike_steps_EI_A = [load_spike_steps(os.path.join(data_path.SD_state1_pXY, "pEI_A/pEI_{}_{}".format(g,alpha))) for g in gIs]
spike_steps_EI_B = [load_spike_steps(os.path.join(data_path.SD_state1_pXY, "pEI_B/pEI_{}_{}".format(g,alpha))) for g in gIs]
Ks_EI_A = [np.hstack(ss).shape[0] for ss in spike_steps_EI_A]
Ks_EI_B = [np.hstack(ss).shape[0] for ss in spike_steps_EI_B]
ps_EI_A = [(Ks_EI_A[i]-Ks_EI_A[0])/100000 for i in range(len(Ks_EI_A))]
ps_EI_B = [(Ks_EI_B[i]-Ks_EI_B[0])/100000 for i in range(len(Ks_EI_B))]
ps_EI = (np.array(ps_EI_A)+np.array(ps_EI_B))/2

spike_steps_II_A = [load_spike_steps(os.path.join(data_path.SD_state1_pXY, "pII_A/pII_{}_{}".format(g,alpha))) for g in gIs]
spike_steps_II_B = [load_spike_steps(os.path.join(data_path.SD_state1_pXY, "pII_B/pII_{}_{}".format(g,alpha))) for g in gIs]
Ks_II_A = [np.hstack(ss).shape[0] for ss in spike_steps_II_A]
Ks_II_B = [np.hstack(ss).shape[0] for ss in spike_steps_II_B]
ps_II_A = [(Ks_II_A[i]-Ks_II_A[0])/100000 for i in range(len(Ks_II_A))]
ps_II_B = [(Ks_II_B[i]-Ks_II_B[0])/100000 for i in range(len(Ks_II_B))]
ps_II = (np.array(ps_II_A)+np.array(ps_II_B))/2


np.save("gE,pEE,pIE.npy", [gEs,ps_EE,ps_IE])
np.save("gI,pEI,pII.npy", [gIs,ps_EI,ps_II])