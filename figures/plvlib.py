"""
PLVlib
------

Last update: 23 August 2023
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter, hilbert

def myformat():
    plt.rc("font", family="courier", size=20)
    plt.rcParams.update({"mathtext.default":"regular"})
    plt.rcParams["legend.fontsize"] = 16
    plt.rcParams["legend.frameon"] = True
    plt.rcParams["legend.fancybox"] = True
    plt.rcParams["legend.facecolor"] = (1, 1, 1)
    plt.rcParams["legend.edgecolor"] = (0, 0, 0)
    plt.rcParams["legend.framealpha"] = .9
    plt.rcParams["legend.borderpad"] = .4
    plt.rcParams["legend.columnspacing"] = 1.5

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

def _wrap_within_period(x, period=2*np.pi): return (x + period/2) % (period) - period/2

def _fill_lower_trimatrix(trimatrix_flatten):
    size = int(np.sqrt(len(trimatrix_flatten)*2))+1
    mask = np.tri(size,dtype=bool,k=-1) # or np.arange(n)[:,None] > np.arange(n)
    out = np.zeros((size,size),dtype=float)
    out[mask] = trimatrix_flatten
    return out

def _band_filter_butter(signal, cutoff, fs, filter_order=2, btype="bandpass"):
    b, a = butter(N=filter_order, Wn=cutoff, btype=btype, fs=fs)
    return lfilter(b, a, signal)

def _line_plot(x, y, discrete=False, ax=None, **options):
    style, c, a, ms, lw = "-", "k", 1, 10, 2
    label, title = "", ""
    for key, value in options.items():
        if key == "style": style = value
        elif key == "c": c = value
        elif key == "a": a = value
        elif key == "ms": ms = value
        elif key == "lw": lw = value
        elif key == "label": label = value
        elif key == "title": title = value
    if ax == None:
        fig, ax = plt.subplots(figsize=(6,5))
        ax.set_title(title, y=1.02, loc='left')
    else: ax = ax
    if not discrete: ax.plot(x, y, style, color=c, ms=ms, lw=lw, alpha=a, label=label, zorder=5)
    else: ax.step(x, y, style, color=c, ms=ms, lw=lw, alpha=a, label=label, zorder=5)
    ax.set_axisbelow(True)
    ax.grid(True)

def _power_spectrum(signal, sampling_freq, axes):
    """`sampling_freq`=`int(1000/dt_ms)`"""
    fourier_real_part = np.real(np.fft.fft(signal))
    fourier_imag_part = np.imag(np.fft.fft(signal))
    power_spectrum = np.abs(np.fft.fft(signal))**2
    freqs = np.fft.fftfreq(signal.size, 1/sampling_freq)

    idx = np.argsort(freqs)
    # sp_sum = np.sum(power_spectrum[idx])
    _line_plot(freqs[idx], fourier_real_part[idx], c="r", lw=.7, label="real part", ax=axes[0])
    _line_plot(freqs[idx], fourier_imag_part[idx], c="c", lw=.7, label="imaginary part", ax=axes[0])
    _line_plot(freqs[idx], power_spectrum[idx], lw=1, label="power spectrum", ax=axes[1])
    lg1 = axes[0].legend(fontsize=14, ncol=1, facecolor=(1,1,1), edgecolor=(.5,.5,.5), framealpha=.9, loc="upper right")
    lg1.get_frame().set_boxstyle("round", pad=.05, rounding_size=.5)
    lg2 = axes[1].legend(fontsize=14, facecolor=(1,1,1), edgecolor=(.5,.5,.5), framealpha=.9)
    lg2.get_frame().set_boxstyle("round", pad=.05, rounding_size=.5)
    axes[0].set(ylabel="magnitude", xlabel="frequency")
    axes[0].set_title("Fourier transform", loc="left", pad=25)
    axes[1].set(ylabel="power (mV^2/Hz)", xlabel="frequency (Hz)")
    return freqs[idx], power_spectrum[idx]

def _multiple_formatter(denominator=2, number=np.pi, latex="\pi"):
        def gcd(a, b):
            while b:
                a, b = b, a%b
            return a
        def _multiple_formatter(x, pos):
            den = denominator
            num = np.int(np.rint(den*x/number))
            com = gcd(num,den)
            (num,den) = (int(num/com),int(den/com))
            if den==1:
                if num==0:
                    return r'$0$'
                if num==1:
                    return r'$%s$'%latex
                elif num==-1:
                    return r'$-%s$'%latex
                else:
                    return r'$%s%s$'%(num,latex)
            else:
                if num==1:
                    return r'$\frac{%s}{%s}$'%(latex,den)
                elif num==-1:
                    return r'$\frac{-%s}{%s}$'%(latex,den)
                else:
                    return r'$\frac{%s%s}{%s}$'%(num,latex,den)
        return _multiple_formatter

class _Multiple:
    def __init__(self, denominator=2, number=np.pi, latex="\pi"):
        self.denominator = denominator
        self.number = number
        self.latex = latex

    def locator(self):
        return plt.MultipleLocator(self.number / self.denominator)

    def formatter(self):
        return plt.FuncFormatter(_multiple_formatter(self.denominator, self.number, self.latex))


class PhaseLockingValue:

    def __init__(self) -> None:
        """
        [1] configure_bandpass_filter(...), configure_general(...)\n
        [2] import_data(SimulationData)\n
        [3] compute_PLV()\n
        """
        self._isBandFiltered = False
        self._discard_dynamics_time = 0 # seconds
        self._discard_phaseseries_time = 0 # seconds

    def configure_bandpass_filter(self, enable=True, lowcut=1, highcut=400):
        """Butterworth band-pass filter is used. See `scipy.signal.butter`"""
        self._isBandFiltered = enable
        self.lowcut, self.highcut = lowcut, highcut

    def configure_general(self, discard_dynamics_time=0, discard_phaseseries_time=0):
        """unit of time: second"""
        self._discard_dynamics_time = discard_dynamics_time
        self._discard_phaseseries_time = discard_phaseseries_time

    def import_data(self, voltage_signal, neuron_num: int, dt_ms: float, shiftbyMedian=True):
        """Obtain band-pass filtered signals from membrane potential time series.
        Configure band-pass filter settings in `configure_bandpass_filter`."""
        self._neuron_num = neuron_num
        self._fs = 1000/dt_ms # sampling frequency
        signals_unfiltered = voltage_signal[:,int(self._discard_dynamics_time*self._fs):]
        if shiftbyMedian: signals_unfiltered -= np.median(signals_unfiltered,1)[:,np.newaxis]
        if voltage_signal.shape[0] < 10: self.signals_unfiltered = signals_unfiltered
        print("PLV: data imported - OK")
        ### bandpass filter ###
        if self._isBandFiltered: self.signals = np.array([_band_filter_butter(s, [self.lowcut,self.highcut], fs=self._fs, btype="bandpass") for s in signals_unfiltered])
        # if self._isBandFiltered: self.signals = np.array([_band_filter_butter(s, self.lowcut, fs=self._fs, btype="highpass") for s in signals_unfiltered])
        else: self.signals = np.array(signals_unfiltered)
        print("PLV: band-pass filtered - OK")
        return self.signals

    def get_instantaneous_phase(self, unwarp=True):
        ### Hilbert transform to obtain phase ###
        if unwarp: return np.array([np.unwrap(np.angle(hilbert(s))) for s in self.signals])
        else: return np.array([np.angle(hilbert(s)) for s in self.signals])

    def compute_PLV(self):
        """Return: tuple ([1],[2])
        - [1]: global PLV: float
        - [2]: pair-wise PLV lower-left matrix: numpy.ndarray
        """
        phase = self.get_instantaneous_phase()[:,int(self._discard_phaseseries_time*self._fs):]
        print("PLV: phase obtained from Hilbert transform - OK")
        self.plv_pairwise = np.hstack([[np.abs(np.sum(np.exp(1j*(phase[i]-phase[j])))) for j in range(i)] for i in range(1,self._neuron_num)])/phase.shape[-1]
        self.globalPLV = np.sum(self.plv_pairwise)/(self._neuron_num*(self._neuron_num-1)/2)
        print("PLV: the PLV computed to be {:.4f}".format(self.globalPLV))
        return (self.globalPLV, _fill_lower_trimatrix(self.plv_pairwise))

    def save_plv_data(self, filename):
        """Save: tuple ([1],[2],[3]) in binary
        - [1]: global PLV: float
        - [2]: pair-wise PLV lower-left matrix: numpy.ndarray
        - [3]: filter info: array, [isBandFiltered:bool, lowfreq_cutoff:float, highfreq_cutoff:float]\n
        Filename: plv_data
        """
        self.plv_pairwise = np.array(self.plv_pairwise,dtype=np.float16)
        np.save(open(filename,"wb"), np.array([self.globalPLV, self.plv_pairwise, [self._isBandFiltered,self.lowcut,self.highcut]],dtype=object))

    def load_plv_data(self, filename):
        """Return: tuple ([1],[2],[3])
        - [1]: global PLV: float
        - [2]: pair-wise PLVs: numpy.ndarray (lower-left matrix elements)
        - [3]: filter info: array, [isBandFiltered:bool, lowfreq_cutoff:float, highfreq_cutoff:float]
        """
        gplv,plvpair,filt = np.load(open(filename,"rb"),allow_pickle=True)
        plvpair = np.array(plvpair,dtype=np.float32)
        return (gplv,plvpair,filt)
        
    ### checking ###
    def plv2_check(self, original_duration_ms):
        if self.signals.shape[0] != 2: raise ValueError("the number of imported signals must be two")
        duration = original_duration_ms/1000-self._discard_dynamics_time
        num_of_samples = int(self._fs*duration)
        t = np.arange(num_of_samples) / self._fs

        phase_unwarped = self.get_instantaneous_phase(unwarp=True)
        phase = self.get_instantaneous_phase(unwarp=False)
        phase_difference = phase_unwarped[1] - phase_unwarped[0]
        phase_difference = phase_difference[int(self._discard_phaseseries_time*self._fs):]
        print("PLV: the pairwise PLV computed to be {:.6f}".format(np.abs(np.sum(np.exp(1j*phase_difference)))/len(phase_difference)))

        fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(12,6),sharex=True, gridspec_kw={"height_ratios":[1.5,1,1]})
        _line_plot(t, self.signals[0][:-1], c="b", lw=1, label="neuron1", ax=ax1)
        _line_plot(t, self.signals[1][:-1], c="r", lw=1, label="neuron2", ax=ax1)
        _line_plot(t, self.signals_unfiltered[0][:-1], c="b", lw=1, a=.3, ax=ax1)
        _line_plot(t, self.signals_unfiltered[1][:-1], c="r", lw=1, a=.3, ax=ax1)
        _line_plot(t, phase[0][:-1], c="b", lw=1, ax=ax2)
        _line_plot(t, phase[1][:-1], c="r", lw=1, ax=ax2)
        _line_plot(t[int(self._discard_phaseseries_time*self._fs):], _wrap_within_period(phase_difference)[:-1], c="k", lw=1, ax=ax3)
        ax2.set(xlim=(0,duration),ylim=(None,None))
        ax1.set(ylabel="voltage\nsignal")
        ax2.set(ylabel="phase")
        ax3.set(xlabel="time (s)",ylabel="phase\ndifference")
        ax2.yaxis.set_major_locator(plt.MultipleLocator(np.pi))
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(_multiple_formatter()))
        ax3.yaxis.set_major_locator(plt.MultipleLocator(np.pi))
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(_multiple_formatter()))
        lg = ax1.legend(fontsize=13, ncol=3, facecolor=(1,1,1), edgecolor=(0,0,0), framealpha=.9, loc="lower right")
        lg.get_frame().set_boxstyle("round", pad=.1, rounding_size=.5)
        plt.tight_layout()

def power_spectrum_check(voltage_signal, dt_ms: float, highcut=None):
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5),sharex=False)
    _, power = _power_spectrum(voltage_signal,int(1000/dt_ms),(ax1,ax2))
    _line_plot([highcut,highcut],[0,np.amax(power)],c="r",ax=ax2)
    ax2.set(xlim=(0,None),ylim=(0,1.2*np.amax(np.delete(power,np.argmax(power)))))
    plt.tight_layout()
