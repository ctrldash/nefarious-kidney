import os
import re

import h5py
import numpy as np
import scikits.audiolab as au
from scipy.io import wavfile
from pysas.mcep import spec2mcep_from_matrix
from pysas import World


frame_length = 512
# Order of mel-cepstrum
order = 25
alpha = 0.41
hop_length = frameperiod = 160
samplingrate = 16000


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def normalize(x):
    return x.astype(float) / x.max()


def get_mcep_from_spec_mats(fname):
    # signal, samplingrate, _ = waveread(fname)
    samplingrate, signal = wavfile.read(fname)
    # print(signal.dtype)
    world = World(samplingrate, float(hop_length) / samplingrate * 1000)
    f0, spec_mat, aperiod_mat = world.analyze(np.float64(signal))

    mcep_mat = spec2mcep_from_matrix(spec_mat, order, alpha)
    return f0, mcep_mat, aperiod_mat


def get_all_coefs(signal, sr):
    world = World(sr, float(hop_length) / sr * 1000)
    f0, spec_mat, aperiod_mat = world.analyze(np.float64(signal))

    mcep_mat = spec2mcep_from_matrix(spec_mat, order, alpha)
    return f0, mcep_mat, aperiod_mat


def coefs_set(filepath):
    src_f0, src_mcep_mat, src_aper = get_mcep_from_spec_mats(
        filepath)
    return src_f0, src_mcep_mat, src_aper


class Stream(object):
    """Base class for data-containers of some kind of coefficients.
    Takes numpy-array as data input and pack of related parameters.
    """
    _param_names = []

    def __init__(self, data, **params):
        if set(self._param_names) != set(params):
            raise ValueError("Not all params given")

        self._params = params
        self._data = data

    @classmethod
    def read(cls, filename):
        """Load saved h5py.File object into instance of curent class.
        If type (of container-class) of loading file does not correspond to class
        which method *.read() is called then TypeError will be raised.
        """
        with h5py.File(filename, "r") as f:
            params = dict(f.attrs)
            if not params['type'] == cls.__name__:
                raise TypeError("File object type does not correspond to class type."
                                "File type is: {0}. Needed: {1}".format(
                                    params['type'], cls.__name__))
            del params['type']
            data = f.values()[0].value
        return cls(data, **params)

    def write(self, filename):
        """Save instance data and parameters to h5py.File object to disk."""
        with h5py.File(filename, "w") as f:
            f.attributes.update(self.params)
            f.attributes['type'] = self.__class__.__name__
            f.create_dataset("data", data=self.data, dtype="i")

    @property
    def data(self):
        """Represents np.ndarray of class-conteiner's coefficients."""
        return self._data

    @property
    def params(self):
        return self._params

    def clone_with_data(self, data):
        """Create new instance of current class with new pack of data."""
        return self.__class__(data, **self.params)


class MCEPStream(Stream):
    """Container for mcep-matrix coefficients.

        Parameters
        ----------
        data : np.ndarray
        alpha : float
        order : int
    """
    _param_names = ['alpha', 'order']

    def get_voiced_frames(self, f0):
        """Return indices and corresponding mcep-coefficient
        which have energy > 0 (f0 > 0).

            Parameters
            ----------
            f0 : np.ndarray

            Return
            ------
            tuple (indices, mcep-coefficients)

        """
        indices = np.nonzero(f0 != 0)[0]
        return indices, self.data[indices]

    def get_unvoiced_frames(self, f0):
        """Return indices and corresponding mcep-coefficient
        which have no energy (f0 = 0).

            Parameters
            ----------
            f0 : np.ndarray

            Return
            ------
            tuple (indices, mcep-coefficients)

        """
        indices = np.nonzero(f0 == 0)[0]
        return indices, self.data[indices]


class WaveStream(Stream):
    """Contaner for signal coefficients."""
    _param_names = ['sample_rate']

    def play(self):
        au.play(
            self.data.astype(float) / self.data.max(),
            fs=self.params['sample_rate'])

    @classmethod
    def read_from_wav(cls, path):
        """Load data from wav file and return instance of WaveStream class.
        If path is folder then load and join data from all wav files
        in this directory in natural sort order.
        """
        s_data = []
        if os.path.isfile(path):
            sr, signal = wavfile.read(path)
            s_data.append(signal)
        elif os.path.isdir(path):
            for filename in natural_sort(os.listdir(path)):
                if filename.lower().endswith(".wav"):
                    sr, signal = wavfile.read(os.path.join(path, filename))
                    s_data.append(signal)
        else:
            pass
        data = np.hstack(s_data)

        return cls(data, sample_rate=sr)

    @property
    def int_data(self):
        return self.data.astype(int)

    @property
    def float_data(self):
        return self.data.astype(float)


class F0Stream(Stream):
    """Simple container of f0 coefficients (1-d array)."""
    pass


class AperiodicityStream(Stream):
    """Simple container of aperiodicity coefficients."""
    pass


def test():
    w = WaveStream.read_from_wav("testfile.wav")
    raw_f0, raw_mcep_mat, raw_aperiod_mat = get_all_coefs(
        w.data, w.params['sample_rate'])

    # get all coefs
    f0 = F0Stream(raw_f0)
    print(f0.data)
    mcep_mat = MCEPStream(raw_mcep_mat)
    print(mcep_mat.data)
    aperiod_mat = AperiodicityStream(raw_aperiod_mat)
    print(aperiod_mat.data)

    # synth some wav




if __name__ == "__main__":
    test()
