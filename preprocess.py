"""
Preprocess dataset

usage:
    preprocess.py [options] <wav-dir>...

options:
     --output-dir=<dir>      Directory where processed outputs are saved. [default: data_dir].
    -h, --help              Show help message.
"""
import os
from docopt import docopt
import numpy as np
import math, pickle, os
from audio import *
from hparams import hparams as hp
from utils import *
from tqdm import tqdm


class MelVocoder:
    def __init__(self):
        self._mel_basis = None

    def fft(self, y, sample_rate, use_preemphasis=False):
        if use_preemphasis:
            pre_y = self.preemphasis(y)
        else:
            pre_y = y
        D = self._stft(pre_y, sample_rate)
        return D.transpose()

    def ifft(self, y, sample_rate):
        y = y.transpose()
        return self._istft(y, sample_rate)

    def melspectrogram(self, y, sample_rate, num_mels, use_preemphasis=False):
        if use_preemphasis:
            pre_y = self.preemphasis(y)
        else:
            pre_y = y
        D = self._stft(pre_y, sample_rate)
        S = self._amp_to_db(self._linear_to_mel(np.abs(D), sample_rate, num_mels))
        return self._normalize(S)

    def preemphasis(self, x):
        return signal.lfilter([1, -0.97], [1], x)

    def _istft(self, y, sample_rate):
        n_fft, hop_length, win_length = self._stft_parameters(sample_rate)
        return librosa.istft(y, hop_length=hop_length, win_length=win_length)

    def _stft(self, y, sample_rate):
        n_fft, hop_length, win_length = self._stft_parameters(sample_rate)
        return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann')

    def _linear_to_mel(self, spectrogram, sample_rate, num_mels):
        if self._mel_basis is None:
            self._mel_basis = self._build_mel_basis(sample_rate, num_mels)
        return np.dot(self._mel_basis, spectrogram)

    def _build_mel_basis(self, sample_rate, num_mels):
        n_fft = 1024
        return librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels, fmax=7600)

    def _normalize(self, S):
        min_level_db = -100.0
        return np.clip((S - min_level_db) / -min_level_db, 0, 1)

    def _stft_parameters(self, sample_rate):
        n_fft = 1024
        hop_length = 256
        win_length = n_fft
        return n_fft, hop_length, win_length

    def _denormalize(self, S):
        min_level_db = -100.0
        # return np.clip((S - min_level_db) / -min_level_db, 0, 1)
        return S * (-min_level_db) + min_level_db

    def _db_to_amp(self, x):
        reference = 20.0
        # return 20 * np.log10(np.maximum(1e-5, x)) - reference
        return np.power(10.0, (x + reference) * 0.05)

    def _amp_to_db(self, x):
        reference = 20.0
        return 20 * np.log10(np.maximum(1e-5, x)) - reference

    def griffinlim(self, spectrogram, n_iter=50, sample_rate=16000):
        n_fft, hop_length, win_length = self._stft_parameters(sample_rate)
        return self._griffinlim(spectrogram.transpose(), n_iter=n_iter, n_fft=n_fft, hop_length=hop_length)

    def _griffinlim(self, spectrogram, n_iter=100, window='hann', n_fft=2048, hop_length=-1, verbose=False):
        if hop_length == -1:
            hop_length = n_fft // 4

        angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

        t = tqdm(range(n_iter), ncols=100, mininterval=2.0, disable=not verbose)
        for i in t:
            full = np.abs(spectrogram).astype(np.complex) * angles
            inverse = librosa.istft(full, hop_length=hop_length, window=window)
            rebuilt = librosa.stft(inverse, n_fft=n_fft, hop_length=hop_length, window=window)
            angles = np.exp(1j * np.angle(rebuilt))

            if verbose:
                diff = np.abs(spectrogram) - np.abs(rebuilt)
                t.set_postfix(loss=np.linalg.norm(diff, 'fro'))

        full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = librosa.istft(full, hop_length=hop_length, window=window)

        return inverse


mvc = MelVocoder()


def _normalize(data):
    m = np.max(np.abs(data))
    data = (data / m) * 0.999
    return data


def get_wav_mel(path):
    """Given path to .wav file, get the quantized wav and mel spectrogram as numpy vectors

    """
    wav = load_wav(path)
    wav = _normalize(wav)
    mel = mvc.melspectrogram(wav, sample_rate=hp.sample_rate, num_mels=hp.num_mels)  # melspectrogram(wav)
    if hp.input_type == 'raw' or hp.input_type == 'mixture':
        return wav.astype(np.float32), mel
    elif hp.input_type == 'mulaw':
        quant = mulaw_quantize(wav, hp.mulaw_quantize_channels)
        return quant.astype(np.int), mel
    elif hp.input_type == 'bits':
        quant = quantize(wav)
        return quant.astype(np.int), mel
    else:
        raise ValueError("hp.input_type {} not recognized".format(hp.input_type))


def process_data(wav_dirs, output_path, mel_path, wav_path):
    """
    given wav directory and output directory, process wav files and save quantized wav and mel
    spectrogram to output directory
    """
    dataset_ids = []
    # get list of wav files
    wav_files = []
    for wav_dir in wav_dirs:
        thisdir = os.listdir(wav_dir)
        thisdir = [os.path.join(wav_dir, thisfile) for thisfile in thisdir]
        wav_files += thisdir

    # check wav_file
    assert len(wav_files) != 0 or wav_files[0][-4:] == '.wav', "no wav files found!"
    # create training and testing splits
    test_wav_files = wav_files[:4]
    wav_files = wav_files[4:]
    for i, wav_file in enumerate(tqdm(wav_files)):
        # get the file id
        # from ipdb import set_trace
        # set_trace()
        file_id = '{:d}'.format(i).zfill(5)
        wav, mel = get_wav_mel(wav_file)
        # save
        np.save(os.path.join(mel_path, file_id + ".npy"), mel)
        np.save(os.path.join(wav_path, file_id + ".npy"), wav)
        # add to dataset_ids
        dataset_ids.append(file_id)

    # save dataset_ids
    with open(os.path.join(output_path, 'dataset_ids.pkl'), 'wb') as f:
        pickle.dump(dataset_ids, f)

    # process testing_wavs
    test_path = os.path.join(output_path, 'test')
    os.makedirs(test_path, exist_ok=True)
    for i, wav_file in enumerate(test_wav_files):
        wav, mel = get_wav_mel(wav_file)
        # save test_wavs
        np.save(os.path.join(test_path, "test_{}_mel.npy".format(i)), mel)
        np.save(os.path.join(test_path, "test_{}_wav.npy".format(i)), wav)

    print(
        "\npreprocessing done, total processed wav files:{}.\nProcessed files are located in:{}".format(len(wav_files),
                                                                                                        os.path.abspath(
                                                                                                            output_path)))


if __name__ == "__main__":
    args = docopt(__doc__)
    wav_dir = args["<wav-dir>"]
    output_dir = args["--output-dir"]

    # create paths
    output_path = os.path.join(output_dir, "")
    mel_path = os.path.join(output_dir, "mel")
    wav_path = os.path.join(output_dir, "wav")

    # create dirs
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(mel_path, exist_ok=True)
    os.makedirs(wav_path, exist_ok=True)

    # process data
    process_data(wav_dir, output_path, mel_path, wav_path)


def test_get_wav_mel():
    wav, mel = get_wav_mel('sample.wav')
    print(wav.shape, mel.shape)
    print(wav)
