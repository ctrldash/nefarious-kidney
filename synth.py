import os
import numpy as np
import pysptk
from pysas import excite
from scipy.io import wavfile
import librosa
from pysas import World, waveread
from scipy.io.wavfile import read
from pysas.mcep import mcep2spec_from_matrix, spec2mcep_from_matrix
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ifolder", help="input folder", required=True)
parser.add_argument(
    "--synth_all",
    action='store_true',
    help="need synth or smth else",
    default=False)
args = parser.parse_args()


frame_length = 512
# Order of mel-cepstrum
order = 25
alpha = 0.41
hop_length = frameperiod = 160
sample_rate = samplingrate = 16000


def normalize(x):
    return x.astype(float) / x.max()


def normalize_int16(s):
    mi, ma = s.min(), s.max()
    return np.int16(s.astype(float) / max(np.abs(mi), np.abs(ma)) * 32767.0)


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


def adapt_f0(src_f0, target_f0):
    mean_s = src_f0[src_f0.nonzero()].mean()
    mean_t = target_f0[target_f0.nonzero()].mean()

    std_s = src_f0[src_f0.nonzero()].std()
    std_t = target_f0[target_f0.nonzero()].std()

    src_f0_new = src_f0.copy()
    src_f0_new[src_f0.nonzero()] = (
        src_f0_new[src_f0.nonzero()] - mean_s) * std_t / std_s + mean_t
    return src_f0_new


def synthesize_from_mc(sample_rate, f0, mc):
    generator = excite.ExcitePulse(sample_rate, hop_length, False)
    source_excitation = generator.gen(f0)
    # Convert mel-cesptrum to MLSADF coefficients
    b = np.apply_along_axis(pysptk.mc2b, 1, mc, alpha)

    synthesizer = pysptk.synthesis.Synthesizer(
        pysptk.synthesis.MLSADF(
            order=order, alpha=alpha),
        hop_length)

    x_synthesized = synthesizer.synthesis(source_excitation, b)
    # al.play(x_synthesized.astype(float) / x_synthesized.max(), fs=16000)
    # return x_synthesized
    return normalize(x_synthesized)


def synthesize_from_spec(world, f0, spec_mat, aperiod_mat):
    out = world.synthesis(f0, spec_mat, aperiod_mat)
    return normalize(out)


def synt_all_spec(folder_in):
    folder_out = folder_in[:-11] + "_synt_by_spec"
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)

    specs = sorted([item for item in os.listdir(folder_in) if "_spec" in item])
    f0s = sorted([item for item in os.listdir(folder_in) if "_f0" in item])
    apers = sorted([item for item in os.listdir(folder_in) if "_aper" in item])
    world = World(sample_rate, float(hop_length) / sample_rate * 1000)

    for spec_file, f0_file, aper_file in zip(specs, f0s, apers):
        print(spec_file)
        res = synthesize_from_spec(
            world,
            np.load(os.path.join(folder_in, f0_file)),
            np.load(os.path.join(folder_in, spec_file)),
            np.load(os.path.join(folder_in, aper_file)))
        print("writing synth for {0}".format(spec_file))
        wavfile.write(
            os.path.join(
                folder_out,
                spec_file.replace("spec.npy", "") + "synth.wav"),
            sample_rate, res)


def synt_all_mc(folder_in):
    folder_out = folder_in[:-9] + "_synt_by_mc"
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)

    mcs = sorted([item for item in os.listdir(folder_in) if "_mc" in item])
    f0s = sorted([item for item in os.listdir(folder_in) if "_f0" in item])

    for mc_file, f0_file in zip(mcs, f0s):
        print(mc_file, f0_file)
        res = synthesize_from_mc(
            sample_rate,
            np.load(os.path.join(folder_in, f0_file)),
            np.load(os.path.join(folder_in, mc_file)))
        print("writing synth for {0}".format(mc_file))
        wavfile.write(
            os.path.join(folder_out, mc_file.replace("mc.npy", "") + "synth.wav"),
            sample_rate, res)


# def synt_with_new_f0_mc(src_coef, target_coef, new_file):
#     sample_rate, src_f0, src_mc = src_coef
#     sample_rate, target_f0, target_mc = target_coef

#     f0_new = adapt_f0(src_f0, target_f0)
#     res = synthesize_from_mc(sample_rate, f0_new, src_mc)
#     wavfile.write(new_file, sample_rate, res)


# def synt_with_new_f0_spec(src_coef, target_coef, new_file):
#     # set w = True is you want to get world-object
#     src_f0, src_spec, src_aper = src_coef
#     target_f0, target_spec, target_aper = target_coef

#     f0_new = adapt_f0(src_f0, target_f0)
#     w = World(sample_rate, float(hop_length) / sample_rate * 1000)
#     res = synthesize_from_spec(w, f0_new, src_spec, src_aper)
#     wavfile.write(new_file, 16000, res)


def synt_from_mcep_matrix_to_spec(world, f0, mcep_mat, aperiod_mat):

    # world = World(samplingrate, float(hop_length) / samplingrate * 1000)
    fft_size = world.fftsize()
    spec_from_mcep = mcep2spec_from_matrix(mcep_mat, alpha, fft_size)
    out = world.synthesis(f0, spec_from_mcep, aperiod_mat)
    return out
# wavfile.write("norm.wav", 16000, normalize_int16(s))


def synt_all_method3(folder_in):
    folder_out = folder_in[:-15] + "_synt_mcep_mat"
    if not os.path.exists(folder_out):
        os.mkdir(folder_out)

    mcep_mats = sorted([item for item in os.listdir(folder_in) if "_mc_mat" in item])
    f0s = sorted([item for item in os.listdir(folder_in) if "_f0" in item])
    aper_mats = sorted([item for item in os.listdir(folder_in) if "_aper_mat" in item])
    world = World(samplingrate, float(hop_length) / samplingrate * 1000)

    for mcep_mat_file, f0_file, aper_mat_file in zip(mcep_mats, f0s, aper_mats):
        print(mcep_mat_file)
        res = synt_from_mcep_matrix_to_spec(
            world,
            np.load(os.path.join(folder_in, f0_file)),
            np.load(os.path.join(folder_in, mcep_mat_file)),
            np.load(os.path.join(folder_in, aper_mat_file)))
        print("writing synth for {0}".format(mcep_mat_file))
        # wavfile.write("norm.wav", 16000, normalize_int16(s))
        wavfile.write(
            os.path.join(
                folder_out,
                mcep_mat_file.replace("mc_mat.npy", "") + "synth.wav"),
            sample_rate,
            normalize_int16(res))


if __name__ == "__main__":
    # time python synth.py --ifolder=data/DTW/src_audio_coefs_spec --synth_all
    # time python synth.py --ifolder=data/DTW/src_audio_coefs_mc --synth_all
    # time python synth.py --ifolder=data/DTW/tgt_audio_coefs_spec --synth_all
    # time python synth.py --ifolder=data/DTW/tgt_audio_coefs_mc --synth_all

    # time python synth.py --ifolder=data/DTW/tgt_audio_coefs_mcep_mat --synth_all
    if args.synth_all:
        if "_mcep_mat" in args.ifolder:          # get mcep matrix from spec
            synt_all_method3(args.ifolder)
        elif "_mc" in args.ifolder:
            synt_all_mc(args.ifolder)
        elif "_spec" in args.ifolder:
            synt_all_spec(args.ifolder)

    else:
        print("No input - no science :D")
