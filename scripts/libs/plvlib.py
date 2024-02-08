"""
PLVlib
------

Last update: 18 November 2023
"""

import numpy as np
from scipy.signal import butter, lfilter, hilbert
from numba import cuda

MAXVAL_uint16_t = 65535
MAXVAL_uint8_t = 255



class Compressor:

    def load_encoded(self, filename):
        return np.load(open(filename,"rb")), np.load(open("{}_a".format(filename),"rb"))

    def save_encoded(self, filename, encoded, a):
        np.save(open(filename,"wb"),encoded), np.save(open("{}_a".format(filename),"wb"),a)

    def encode_uintX_t(self, signal, X=8):
        """
        Encode floating point signal into uintX_t, `X=8`:uint8_t, `X=16`:uint16_t\n
        Decode signal using `decode_uintX_t`\n
        Return `(encoded_signal, [upper_cutoff,lower_cutoff,map_const,X])`
        """
        upper_cutoff = np.amax(signal)
        lower_cutoff = np.amin(signal)
        if X==16: map_const = float(MAXVAL_uint16_t)/(upper_cutoff-lower_cutoff)
        elif X==8: map_const = float(MAXVAL_uint8_t)/(upper_cutoff-lower_cutoff)
        signal[signal > upper_cutoff] = upper_cutoff
        signal[signal < lower_cutoff] = lower_cutoff
        return (np.array((upper_cutoff-signal)*map_const,dtype="B"),[upper_cutoff,lower_cutoff,map_const,X])

    def decode_uintX_t(self, encoded, a):
        """Decode signal from `encode_uintX_t`"""
        return a[0]-np.array(encoded,dtype=float)/a[2]

def fill_lower_trimatrix(trimatrix_flatten):
    size = int(np.sqrt(len(trimatrix_flatten)*2))+1
    mask = np.tri(size,dtype=bool,k=-1) # or np.arange(n)[:,None] > np.arange(n)
    out = np.zeros((size,size),dtype=float)
    out[mask] = trimatrix_flatten
    return out

def _band_filter_butter(signal, cutoff, fs, filter_order=2, btype="bandpass"):
    b, a = butter(N=filter_order, Wn=cutoff, btype=btype, fs=fs)
    return lfilter(b, a, signal)

def _wrap_within_period(x, period=2*np.pi): return (x + period/2) % (period) - period/2


class PLVtool:

    def __init__(self) -> None:
        """
        Steps to calculate the phase-locking value (PLV) from the membrane potential signals:\n
        1. (Optional) call functions `configure_bandpass_filter`, `configure_general`.\n
        2. Call function `import_data` to obtain the band-pass filtered voltage signals.\n
        3. Call function `compute_plv` to calculate the pairwise PLVs.\n
        """
        self._isBandFiltered = False
        self._discard_dynamics_time = 0 # seconds
        self._discard_phaseseries_time = 0 # seconds

    def configure_bandpass_filter(self, enable=True, lowcut=1, highcut=500):
        """The Butterworth band-pass filter is used. See `scipy.signal.butter` for more details.
        You can set the low and high frequency cutoffs, which are 1 and 500 Hz by default.
        Please check whether it is reasonable using the power spectra of the membrane potential
        signals."""
        self._isBandFiltered = enable
        self.lowcut, self.highcut = lowcut, highcut

    def configure_general(self, discard_dynamics_time=0, discard_phaseseries_time=0):
        """Unit of time: second. You can choose to discard the transient time at the start of the
        dynamics (`discard_dynamics_time`), and further discard the unwanted time in the extracted
        phase time series (`discard_phaseseries_time`)."""
        self._discard_dynamics_time = discard_dynamics_time
        self._discard_phaseseries_time = discard_phaseseries_time

    def import_data(self, voltage_signals, dt_ms: float, shiftbyMedian=True):
        """Obtain band-pass filtered signals from membrane potential time series.
        Configure band-pass filter settings in `configure_bandpass_filter`."""
        self._num_neuron = voltage_signals.shape[0]
        self._fs = 1000/dt_ms # sampling frequency
        signals_unfiltered = voltage_signals[:,int(self._discard_dynamics_time*self._fs):]
        if shiftbyMedian: signals_unfiltered -= np.median(signals_unfiltered,1)[:,np.newaxis]
        if voltage_signals.shape[0] < 10: self.signals_unfiltered = signals_unfiltered
        print("> data imported - OK")
        # bandpass filter #
        # if self._isBandFiltered: self.signals = np.array([_band_filter_butter(s, [self.lowcut,self.highcut], fs=self._fs, btype="bandpass") for s in signals_unfiltered])
        if self._isBandFiltered: self.signals = np.array([_band_filter_butter(s, self.highcut, fs=self._fs, btype="lowpass") for s in signals_unfiltered])
        else: self.signals = np.array(signals_unfiltered)
        print("> band-pass filtered - OK")
        return self.signals

    def get_instantaneous_phases(self, unwarp=False):
        """Obtain the phase time series extracted by Hilbert transform."""
        # self.phases = self.get_instantaneous_phases()[:,int(self._discard_phaseseries_time*self._fs):]
        if unwarp: self.phases = np.array([np.unwrap(np.angle(hilbert(s))) for s in self.signals])[:,int(self._discard_phaseseries_time*self._fs):]
        else: self.phases = np.array([np.angle(hilbert(s)) for s in self.signals])[:,int(self._discard_phaseseries_time*self._fs):]
        return self.phases

    def compute_plv(self):
        """Return a tuple with two elements:\n
        - global PLV, float
        - pairwise PLVs, numpy.ndarray (lower-left matrix elements)
        """
        total_timesteps = self.phases.shape[1]
        print("> phases obtained from Hilbert transform - OK")
        self.pairwiseplv = np.hstack([[np.abs(np.sum(np.exp(1j*(self.phases[i]-self.phases[j])))) for j in range(i)] for i in range(1,self._num_neuron)])/total_timesteps
        self.globalplv = np.sum(self.pairwiseplv)/(self._num_neuron*(self._num_neuron-1)/2)
        print("> global PLV computed to be {:.4f}".format(self.globalplv))
        return (self.globalplv, fill_lower_trimatrix(self.pairwiseplv))

    def compute_plv_cuda(self):
        """Very fast computation of PLVs using GPU CUDA. Return a tuple with two elements:\n
        - global PLV, float
        - pairwise PLVs, numpy.ndarray (lower-left matrix elements)
        """
        gpu_block_size = 1024
        if str(cuda.gpus)=="<Managed Device 0>": print("> GPU CUDA availablity - OK")
        # phases = self.get_instantaneous_phases()[:,int(self._discard_phaseseries_time*self._fs):]
        num_neuron, total_timesteps = self.phases.shape
        print("> phases obtained from Hilbert transform - OK")

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
        self.globalplv = np.sum(self.pairwiseplv)/(self._num_neuron*(self._num_neuron-1)/2)
        print("> global PLV computed to be {:.4f}".format(self.globalplv))
        return (self.globalplv, fill_lower_trimatrix(self.pairwiseplv))

    def save_plv_data(self, filename):
        """Save the followings in a binary file:\n
        - global PLV, float
        - pairwise PLVs, numpy.ndarray (lower-left matrix elements)
        - filter info, list\n
            - [0] isBandFiltered, bool
            - [1] lowfreq_cutoff, float
            - [2] highfreq_cutoff, float
        """
        self.pairwiseplv = np.array(self.pairwiseplv,dtype=np.float16)
        np.save(open(filename,"wb"), np.array([self.globalplv, self.pairwiseplv, [self._isBandFiltered,self.lowcut,self.highcut]],dtype=object))

    def load_plv_data(self, filename):
        """Return a tuple with three elements:\n
        - global PLV, float
        - pairwise PLVs, numpy.ndarray (lower-left matrix elements)
        - filter info, list\n
            - [0] isBandFiltered, bool
            - [1] lowfreq_cutoff, float
            - [2] highfreq_cutoff, float
        """
        globalplv,pairwiseplv,filtinfo = np.load(open(filename,"rb"),allow_pickle=True)
        pairwiseplv = np.array(pairwiseplv,dtype=np.float32)
        return (globalplv,pairwiseplv,filtinfo)
