from preamble import *
from mylib import load_spike_steps, get_spike_count


class TheofrS1:

    def __init__(self):
        self.kE, self.kI = 8, 2
        pEs = np.load(os.path.join(data_path.DA_state1_pXY, "gE,pEE,pIE.npy"), "r")
        pIs = np.load(os.path.join(data_path.DA_state1_pXY, "gI,pEI,pII.npy"), "r")
        self.gEs, self.pEEs, self.pIEs = pEs[0], pEs[1], pEs[2]
        self.gIs, self.pEIs, self.pIIs = pIs[0], pIs[1], pIs[2]
        self.spike_rate_0 = [self._get_noise_driven_spike_rate_exc(), self._get_noise_driven_spike_rate_inh()] # [EXC, INH], in Hz

    def _get_noise_driven_spike_rate_exc(self):
        """unit: Hz"""
        return np.mean([x for x in get_spike_count(load_spike_steps(os.path.join(data_path.SD_isolated_stoch_e, "3.0","spks.txt")))])/100

    def _get_noise_driven_spike_rate_inh(self):
        """unit: Hz"""
        return np.mean([x for x in get_spike_count(load_spike_steps(os.path.join(data_path.SD_isolated_stoch_i, "3.0","spks.txt")))])/100

    def get_pEE(self, gE:float): return np.interp(gE, self.gEs, self.pEEs)

    def get_pIE(self, gE:float): return np.interp(gE, self.gEs, self.pIEs)

    def get_pEI(self, gI:float): return np.interp(gI, self.gIs, self.pEIs)

    def get_pII(self, gI:float): return np.interp(gI, self.gIs, self.pIIs)

    def get_determinant(self, gE:float, gI:float):
        kE, kI = self.kE, self.kI
        pEE, pIE = self.get_pEE(gE), self.get_pIE(gE)
        pEI, pII = self.get_pEI(gI), self.get_pII(gI)
        return kE*kI*pEE*pII - kE*kI*pIE*pEI - kE*pEE - kI*pII + 1

    def _convergence_test_geometric_series(self, square_matrix_A, verbose=True):
        if verbose:
            print("testing convergence of A:")
            print("A = "); print(square_matrix_A)
        eigvals = np.linalg.eigvals(square_matrix_A)
        eigvals_modul = np.absolute(eigvals)
        spectral_radius = np.max(eigvals_modul)
        if verbose:
            print(" eigenvalues: {:.5} and {:.5}".format(*eigvals))
            print(" moduli of eigenvalues: {:.5} and {:.5}".format(*eigvals_modul))
            print(" spectral radius: {:.5}".format(spectral_radius))
            print(" test result:")
        if spectral_radius < 1:
            isConvergent = True
            if verbose: print(" > converge")
        else:
            isConvergent = False
            if verbose: print(" > does not converge")
        return isConvergent

    def _sum_of_power_series(self, square_matrix_A, verbose=True):
        isConverge = self._convergence_test_geometric_series(square_matrix_A, verbose)
        if not isConverge:
            if verbose: print("the series does not converge")
            return None
        converged_sum = np.linalg.inv(np.eye(2)-square_matrix_A)
        return converged_sum

    def get_predicted_spike_rate(self, gE:float, gI:float, verbose=False):
        P = np.array([[self.get_pEE(gE), self.get_pEI(gI)], [self.get_pIE(gE), self.get_pII(gI)]])
        K = np.diag(np.array([self.kE, self.kI]))
        A = np.matmul(P, K)
        A_geo_sum = self._sum_of_power_series(A, verbose)
        if A_geo_sum is None: return [np.nan,np.nan] # power series of A does not converge
        spike_rate = np.matmul(A_geo_sum, self.spike_rate_0) # [EXC, INH]
        if verbose: print("geometric series sum of A:"); print(A_geo_sum)
        if verbose: print("predicted neuronal firing rates (Hz):\n EXC: {:.5}\n INH: {:.5}".format(*spike_rate))
        return spike_rate

theofrS1 = TheofrS1()