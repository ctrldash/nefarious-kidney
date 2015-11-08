import os
import numpy as np
import librosa
import pysptk
from pysas import excite
from scipy.io import wavfile
# import scikits.audiolab as al
from pysas import World, waveread
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
args = parser.parse_args()


frame_length = 512
hop_length = 160
# Order of mel-cepstrum
order = 40
alpha = 0.41


def normalize(x):
    return x.astype(float) / x.max()


def get_synt_wav(wav_file_path):
    # # Synthesis from mel-cepstrum
    sample_rate, x = wavfile.read(wav_file_path)
    # assert sample_rate == 16000
    # al.play(x.astype(float) / x.max(), fs=sample_rate)  # Audio(x, rate=sample_rate)

    # all of pysptk functions assume input array is C-contiguous
    # and np.float4 element type
    frames = librosa.util.frame(
        x,
        frame_length=frame_length,
        hop_length=hop_length).astype(np.float64).T

    # Windowing
    frames *= pysptk.blackman(frame_length)

    # assert frames.shape[1] == frame_length

    # F0 estimation
    f0 = pysptk.swipe(
        x.astype(np.float64),
        fs=sample_rate,
        hopsize=hop_length,
        min=50,
        max=500)

    generator = excite.ExcitePulse(sample_rate, hop_length, False)
    source_excitation = generator.gen(f0)

    # apply function along with `time` axis (=1)
    mc = np.apply_along_axis(
        pysptk.mcep,
        1,
        frames,
        order,
        alpha)

    # Convert mel-cesptrum to MLSADF coefficients
    b = np.apply_along_axis(pysptk.mc2b, 1, mc, alpha)

    synthesizer = pysptk.synthesis.Synthesizer(
        pysptk.synthesis.MLSADF(
            order=order, alpha=alpha),
        hop_length)

    x_synthesized = synthesizer.synthesis(source_excitation, b)
    # Audio(x_synthesized, rate=sample_rate)
    # al.play(x_synthesized.astype(float) / x_synthesized.max(), fs=sample_rate)
    return x_synthesized


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
    # al.play(x_synthesized.astype(float) / x_synthesized.max(), fs=sample_rate)
    # return x_synthesized
    return normalize(x_synthesized)


def synthesize_from_spec(
        world,
        f0,
        spec_mat,
        aperiod_mat):
    out = world.synthesis(f0, spec_mat, aperiod_mat)
    return normalize(out)


def get_spec_coefs(fname, w=False):
    signal, samplingrate, _ = waveread(fname)
    world = World(samplingrate, float(hop_length) / samplingrate * 1000)
    f0, spec_mat, aperiod_mat = world.analyze(signal)
    if w:
        return world, f0, spec_mat, aperiod_mat
    else:
        return f0, spec_mat, aperiod_mat


def main_mc(
        src_path="example/dima.wav",
        target_path="example/dasha.1.wav"):
    sample_rate, src_f0, src_mc = get_coefs(src_path)
    sample_rate, target_f0, target_mc = get_coefs(target_path)

    f0_synth = adapt_f0(src_f0, target_f0)
    res = synthesize_from_mc(sample_rate, f0_synth, src_mc)
    wavfile.write("example/synth_dima.wav", sample_rate, res)


def main_spec():
    # set w = True is you want to get world-object
    w, src_f0, src_spec, src_aper = get_spec_coefs("example/dima.wav", w=True)
    w, target_f0, target_spec, target_aper = get_spec_coefs("example/dasha.1.wav", w=True)

    f0_synth = adapt_f0(src_f0, target_f0)
    res = synthesize_from_spec(w, f0_synth, src_spec, src_aper)
    wavfile.write("example/synth_dima_spec.wav", 16000, res)


def save_mc(folder_in):
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


def save_spec(folder_in):
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


if __name__ == "__main__":
    if args.mc:
        save_mc(args.ifolder)
    elif args.spec:
        save_spec(args.ifolder)
