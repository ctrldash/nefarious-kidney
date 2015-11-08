import os
import numpy as np
import librosa
import pysptk
from pysas.mcep import spec2mcep_from_matrix
from scipy.io import wavfile
# import scikits.audiolab as al
from pysas import World, waveread
from pysas.mcep import mcep2spec_from_matrix, spec2mcep_from_matrix

from scipy.io.wavfile import read
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ifolder", help="input folder", required=True)
parser.add_argument(
    "--spec",
    action='store_true',
    help="method to get coefs is implemented by librosa lib",
    default=False)
parser.add_argument(
    "--mc",
    action='store_true',
    help="method to get coefs is implemented by World (pysas' lib)",
    default=False)
parser.add_argument(
    "--method3",
    action='store_true',
    help="method to get coefs is implemented by World (pysas' lib)",
    default=False)
args = parser.parse_args()


frame_length = 512
# Order of mel-cepstrum
order = 25
alpha = 0.41
hop_length = frameperiod = 160
samplingrate = 16000


def normalize(x):
    return x.astype(float) / x.max()


def get_coefs(wav_file_path):
    sample_rate, x = wavfile.read(wav_file_path)
    # al.play(x.astype(float) / x.max(), fs=sample_rate)

    frames = librosa.util.frame(
        x,
        frame_length=frame_length,
        hop_length=hop_length).astype(np.float64).T
    frames *= pysptk.blackman(frame_length)
    f0 = pysptk.swipe(
        x.astype(np.float64),
        fs=sample_rate,
        hopsize=hop_length,
        min=50,
        max=500)

    # order = 40
    # alpha = 0.41
    mc = np.apply_along_axis(
        pysptk.mcep,
        1,
        frames,
        order,
        alpha)

    return sample_rate, f0, mc  # sample_rate- ?, f0, mel-cepstrum coefs


def get_spec_coefs(fname, w=False):
    # signal, samplingrate, _ = waveread(fname)
    samplingrate, signal = read(fname)
    world = World(samplingrate, float(hop_length) / samplingrate * 1000)
    f0, spec_mat, aperiod_mat = world.analyze(signal)

    if w:
        return world, f0, spec_mat, aperiod_mat
    else:
        return f0, spec_mat, aperiod_mat


def get_mcep_from_spec_mats(fname):
    # signal, samplingrate, _ = waveread(fname)
    samplingrate, signal = wavfile.read(fname)
    # print(signal.dtype)
    world = World(samplingrate, float(hop_length) / samplingrate * 1000)
    f0, spec_mat, aperiod_mat = world.analyze(np.float64(signal))

    mcep_mat = spec2mcep_from_matrix(spec_mat, order, alpha)
    return f0, mcep_mat, aperiod_mat


def synt_from_mcep_matrix_to_spec(f0, mcep_mat, aperiod_mat):

    world = World(samplingrate, float(hop_length) / samplingrate * 1000)
    fft_size = world.fftsize()
    spec_from_mcep = mcep2spec_from_matrix(mcep_mat, alpha, fft_size)
    out = world.synthesis(f0, spec_from_mcep, aperiod_mat)
    return out


def save_all_mc_mat_from_spec(folder_in):
    folder_out = folder_in + "_coefs_mcep_mat"
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)

    for filename in os.listdir(folder_in):
        # f0, spec_mat, aperiod_mat
        src_f0, src_mcep_mat, src_aper = get_mcep_from_spec_mats(
            os.path.join(folder_in, filename))
        new_filename = filename.replace(".wav", "")
        np.save(
            os.path.join(folder_out, new_filename + "_f0"),
            src_f0)
        np.save(
            os.path.join(folder_out, new_filename + "_mc_mat"),
            src_mcep_mat)
        np.save(
            os.path.join(folder_out, new_filename + "_aper_mat"),
            src_aper)


def save_all_mc(folder_in):
    folder_out = folder_in + "_coefs_mc"
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)

    for filename in os.listdir(folder_in):
        sample_rate, src_f0, src_mc = get_coefs(
            os.path.join(folder_in, filename))
        np.save(
            os.path.join(folder_out, filename.replace(".wav", "") + "_f0"),
            src_f0)
        np.save(
            os.path.join(folder_out, filename.replace(".wav", "") + "_mc"),
            src_mc)


def save_all_spec(folder_in):
    folder_out = folder_in + "_coefs_spec"
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)

    for filename in os.listdir(folder_in):
        # f0, spec_mat, aperiod_mat
        src_f0, src_spec, src_aper = get_spec_coefs(
            os.path.join(folder_in, filename))
        np.save(
            os.path.join(folder_out, filename.replace(".wav", "") + "_f0"),
            src_f0)
        np.save(
            os.path.join(folder_out, filename.replace(".wav", "") + "_spec"),
            src_spec)
        np.save(
            os.path.join(folder_out, filename.replace(".wav", "") + "_aper"),
            src_aper)


def adapt_f0(src_f0, target_f0):
    mean_s = src_f0[src_f0.nonzero()].mean()
    mean_t = target_f0[target_f0.nonzero()].mean()

    std_s = src_f0[src_f0.nonzero()].std()
    std_t = target_f0[target_f0.nonzero()].std()

    src_f0_new = src_f0.copy()
    src_f0_new[src_f0.nonzero()] = (
        src_f0_new[src_f0.nonzero()] - mean_s) * std_t / std_s + mean_t
    return src_f0_new


if __name__ == "__main__":
    # time python get_coefs.py --ifolder=data/DTW/src_audio --spec
    # time python get_coefs.py --ifolder=data/DTW/tgt_audio --spec
    # time python get_coefs.py --ifolder=data/DTW/src_audio --mc
    # time python get_coefs.py --ifolder=data/DTW/tgt_audio --mc
    # time python get_coefs.py --ifolder=data/DTW/tgt_audio --method3

    if args.mc:
        save_all_mc(args.ifolder)
    elif args.spec:
        save_all_spec(args.ifolder)
    elif args.method3:
        save_all_mc_mat_from_spec(args.ifolder)
