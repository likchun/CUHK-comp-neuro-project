from preamble import *
from import_EntropyFinder import EntropyFinder


params = [
    (0.2, 0,   3, 1),
    (0.2, 0.1, 3, 1),
    (0.2, 0.2, 3, 1),
    (0.2, 0.3, 3, 1),
    (0.2, 0.4, 3, 1),
    (0.2, 0.6, 3, 1),
    (0.2, 0.8, 3, 1),
    (0.2, 1,   3, 1),
    (0.2, 1.5, 3, 1),
    (0.2, 2,   3, 1),
    (0.2, 2.5, 3, 1),
    (0.2, 3,   3, 1),
    (0.2, 3.5, 3, 1),
    (0.2, 4,   3, 1),
]

Nstim = [0, 20, 40, 60, 80, 100]
trial = list(range(0,100))

results = []
for i,par in enumerate(params):
    enF = EntropyFinder()
    enF.load_data(par, data_path.SD_resptostim_step, Nstim, trial, t_range=[0,75])
    mi = enF.mutual_information
    Hr = enF.H_response
    Hc = enF.H_condresp
    a = enF.activity
    R = enF.response
    Pr = enF.P_response
    Pc = enF.P_condresp
    results.append((mi,Hr,Hc,a,R,Pr,Pc,enF.xbins))

    print("-------------------------")
    print("Params: ({},{},{},{})".format(*par))
    print("Result:")
    print("H(R): {:4f}".format(Hr))
    print("H(R|S): {:4f}".format(Hc))
    print("H(R)-H(R|S): {:4f}".format(mi))

results = np.array(results, dtype=object)
np.save(os.path.join(data_path.this_dir, "data_entropygI,state2.npy"), results)