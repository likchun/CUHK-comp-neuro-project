"""
PLVlib2
-------

Last update: 7 December 2023 (nn)
"""

import numpy as np
from scipy.signal import butter, lfilter, hilbert
from scipy.ndimage import gaussian_filter1d
from numba import cuda

MAXVAL_uint16_t = 65535
MAXVAL_uint8_t = 255


def fill_lowerleft_triangular_matrix(LLtrimatrix_flatten:np.ndarray):
    """Return a matrix whose lower-triangle is filled with elements in `LLtrimatrix_flatten`."""
    size = int(np.sqrt(len(LLtrimatrix_flatten)*2))+1
    mask = np.tri(size,dtype=bool,k=-1) # or np.arange(n)[:,None] > np.arange(n)
    LLmatrix = np.zeros((size,size),dtype=float)
    LLmatrix[mask] = LLtrimatrix_flatten
    return LLmatrix


class PLVtool:

    def __init__(self) -> None:
        """Functions to call in order:\n
        1. `import_data`\n
        2. `process_signals`\n
        3. `extract_phases`\n
        4. `compute_plv` or `compute_plv_CUDA` (if CUDA GPU is available)\n"""

    def import_data(self, spike_steps, stepsize_ms:float, duration_ms:float):
        """Import the time steps at which each neuron spike."""
        self.spike_steps = [np.array(s,dtype=int) for s in spike_steps]
        self._num_steps = int(duration_ms/stepsize_ms)
        self._num_neuron = len(spike_steps)
        self._samp_freq = 1000/stepsize_ms # sampling frequency in Hz
        print("> spike data imported")

    def report_warning_W1(self, q=10):
        """Report the total number and indices of neurons which have spike counts less than `q`."""
        spike_count = np.array([x.size for x in self.spike_steps])
        spklessthanQ_ind = np.argwhere(spike_count<q).flatten()
        spklessthanQ_num = spklessthanQ_ind.size
        if spklessthanQ_num > 0:
            print("! WARNING: {} neurons (out of {}) have less than {} spikes. The PLV calculation does not work well for spike trains with too few spikes. Continuing may result in larger PLV than expected. Please consider using longer spike trains.".format(spklessthanQ_num,self._num_neuron,q))
            print("! The indices of neurons which have less than {} spikes are returned by this function.".format(q))
            return spklessthanQ_ind
        else: print("> All neurons have spike count greater than or equal to {}.".format(q))

    def process_signals(self, lowpass_freqs, decimation=0):
        """Process the imported signals.\n
        Parameters:\n
        - `lowpass_freqs`: frequency cutoffs of the low-pass Butterworth filter, array of floats
        - `decimation`: reduce the time resolution of the processed signals by this input value, int\n
        Return the processed signals."""
        self._lowpass_freqs = lowpass_freqs
        self._decimation = decimation
        if len(lowpass_freqs) != self._num_neuron: raise ValueError("size of \"lowpass_freqs\" does not match the number of neurons")
        A,B = 200,1
        self.signals = A*self._get_spike_train() + B*self._triangular_spike_interpolation(self.spike_steps, 0, self._num_steps)
        ### Butterworth low-pass filter ###
        butter_coeffs = [butter(btype="lowpass", N=4, Wn=lowpass_freqs[i], fs=self._samp_freq) for i in range(self._num_neuron)]
        self.signals = np.array([lfilter(*butter_coeffs[i], s) for i,s in enumerate(self.signals)])
        ### Gaussian kernel filter ###
        # self.signals = [gaussian_filter1d(s,500,truncate=2) for i,s in enumerate(self.signals)]
        self.signals -= np.median(self.signals,1)[:,np.newaxis]
        if decimation != 0: self._decimate_signals(decimation)
        print("> signals processed")
        return self.signals

    def extract_phases(self):
        """Extract the phase time series from the processed signals using Hilbert transform.
        Return the phase time series."""
        self.phases = np.array([np.angle(hilbert(s)) for s in self.signals])
        print("> phases extracted")
        return self.phases

    def compute_plv(self):
        """Recommend using `compute_plv_cuda()` instead. It offers faster calculation using GPU CUDA.\n
        Return a tuple:\n
        - network PLV, float
        - pairwise PLVs (N*(N-1)/2 elements of the lower-left triangular matrix), numpy.ndarray
        """
        self.pairwiseplv = np.hstack([[np.abs(np.sum(np.exp(1j*(self.phases[i]-self.phases[j])))) for j in range(i)] for i in range(1,self._num_neuron)])/self.phases.shape[1]
        self.networkplv = np.sum(self.pairwiseplv)/(self._num_neuron*(self._num_neuron-1)/2)
        print("> network PLV: {:.4f}".format(self.networkplv))
        return self.networkplv, self.pairwiseplv

    def compute_plv_GPUCUDA(self):
        """Fast implementation of `compute_plv()` using GPU CUDA.\n
        Return a tuple:\n
        - network PLV, float
        - pairwise PLVs (N*(N-1)/2 elements of the lower-left triangular matrix), numpy.ndarray
        """
        if str(cuda.gpus)=="<Managed Device 0>": print("> GPU CUDA is available")
        else: raise SystemError("Error: GPU CUDA is not available")
        gpu_block_size = 1024
        num_neuron, total_timesteps = self.phases.shape
        # GPU code #
        @cuda.jit
        def kernel(phases,_real,_imag):
            i, j, ti = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
            tj = cuda.threadIdx.x
            t = ti*cuda.blockDim.x + tj
            if i < num_neuron and j < i and t < total_timesteps:
                phases_diff = phases[i, t] - phases[j, t]
                cuda.atomic.add(_real, (i,j), np.cos(phases_diff))
                cuda.atomic.add(_imag, (i,j), np.sin(phases_diff))
        _real = np.zeros((self._num_neuron,self._num_neuron)).astype(np.float32)
        _imag = np.zeros((self._num_neuron,self._num_neuron)).astype(np.float32)
        gpu_grid_size = (self._num_neuron,self._num_neuron,int(np.ceil(total_timesteps/gpu_block_size)))
        phases_cuda = cuda.to_device(self.phases)
        _real_cuda = cuda.to_device(_real)
        _imag_cuda = cuda.to_device(_imag)
        kernel[gpu_grid_size,gpu_block_size](phases_cuda,_real_cuda,_imag_cuda)
        _real = _real_cuda.copy_to_host()
        _imag = _imag_cuda.copy_to_host()
        # GPU code #
        pairwiseplv_LLmatrix = np.hypot(_real,_imag)/total_timesteps
        self.pairwiseplv = pairwiseplv_LLmatrix[np.tril_indices(pairwiseplv_LLmatrix.shape[0],k=-1)]
        self.networkplv = np.sum(self.pairwiseplv)/(self._num_neuron*(self._num_neuron-1)/2)
        print("> network PLV: {:.4f}".format(self.networkplv))
        return self.networkplv, self.pairwiseplv

    def save_plv_data(self, filename):
        """Save the followings in a binary file:\n
        - network PLV, float
        - pairwise PLVs (N*(N-1)/2 elements of the lower-left triangular matrix), numpy.ndarray
        - info, list\n
            - number of time steps, int
            - sampling frequency (Hz), float
            - low-pass cut-off frequency (Hz), float
            - decimation, int
        """
        self.pairwiseplv = np.array(self.pairwiseplv,dtype=np.float16)
        self.infos = [self._num_steps, self._samp_freq, self._lowpass_freqs, self._decimation]
        np.save(open(filename,"wb"), np.array([self.networkplv, self.pairwiseplv],dtype=object), self.infos)

    def load_plv_data(self, filename):
        """Return a tuple with three elements:\n
        - network PLV, float
        - pairwise PLVs (N*(N-1)/2 elements of the lower-left triangular matrix), numpy.ndarray
        - info, list\n
            - number of time steps, int
            - sampling frequency (Hz), float
            - low-pass cut-off frequency (Hz), float
            - decimation, int
        """
        networkplv,pairwiseplv,infos = np.load(open(filename,"rb"),allow_pickle=True)
        pairwiseplv = np.array(pairwiseplv,dtype=np.float32)
        return networkplv, pairwiseplv, infos

    def _get_spike_train(self):
        spike_train = np.zeros((self._num_neuron,self._num_steps+1),dtype=int) # +1 to include the initial time step t0
        for i,s in enumerate(spike_train): s[self.spike_steps[i]] = 1
        return spike_train[:,1:]

    def _triangular_spike_interpolation(self, spike_steps, beg_step, end_step):
        """Return the transform spike trains. The transformed spike trains are obtained by
        replacing the i-th spike in a spike train by a triangle, whose left (resp. right)
        side reaches the mid-point of the [i-1]-th ([i+1]-th) and i-th spikes"""
        interp_y = [np.full(spike_steps[i].size,1) for i in range(self._num_neuron)]
        for i in range(self._num_neuron):
            interp_y[i][1::2] *= -1
            interp_y[i] = np.insert(interp_y[i],[0,interp_y[i].size],0)
            spike_steps[i] = np.insert(spike_steps[i],[0,spike_steps[i].size],[beg_step,end_step])
        return np.abs([np.interp(np.arange(beg_step,end_step),spike_steps[i],interp_y[i]) for i in range(self._num_neuron)])

    def _decimate_signals(self, q:int):
        self.signals = self.signals[:,::q]
