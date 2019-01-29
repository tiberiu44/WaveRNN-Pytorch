"""Synthesis script for WaveRNN vocoder

usage: synthesize.py [options] <mel_input.npy>

options:
    --checkpoint-dir=<dir>       Directory where model checkpoint is saved [default: checkpoints].
    --output-dir=<dir>           Output Directory [default: model_outputs]
    --hparams=<params>           Hyper parameters [default: ].
    --preset=<json>              Path of preset parameters (json).
    --checkpoint=<path>          Restore model from checkpoint path if given.
    --no-cuda                    Don't run on GPU
    -h, --help                   Show this help message and exit
"""
import os
import librosa
import glob

from docopt import docopt
from model import *
from hparams import hparams
from utils import num_params_count
import pickle
import time
import numpy as np
import scipy as sp


def torch_test(model, checkpoint, mel):
    state=checkpoint['state_dict']
    rnn1 = model.rnn1

    weight_ih_l0 = rnn1.weight_ih_l0.detach().cpu().numpy()
    weight_hh_l0 = rnn1.weight_hh_l0.detach().cpu().numpy()
    bias_ih_l0 = rnn1.bias_ih_l0.detach().cpu().numpy()
    bias_hh_l0 = rnn1.bias_hh_l0.detach().cpu().numpy()

    W_ir,W_iz,W_in=np.vsplit(weight_ih_l0, 3)
    W_hr,W_hz,W_hn=np.vsplit(weight_hh_l0, 3)

    b_ir,b_iz,b_in=np.split(bias_ih_l0, 3)
    b_hr,b_hz,b_hn=np.split(bias_hh_l0, 3)

    gru_cell = nn.GRUCell(rnn1.input_size, rnn1.hidden_size).cpu()
    gru_cell.weight_hh.data = rnn1.weight_hh_l0.cpu().data
    gru_cell.weight_ih.data = rnn1.weight_ih_l0.cpu().data
    gru_cell.bias_hh.data = rnn1.bias_hh_l0.cpu().data
    gru_cell.bias_ih.data = rnn1.bias_ih_l0.cpu().data

    hx_ref = torch.randn(1, rnn1.hidden_size)
    hx = hx_ref.clone()
    x_ref = torch.randn(1, rnn1.input_size)
    x = x_ref.clone()

    hx_gru = gru_cell(x, hx)

    x = x_ref.clone().numpy().T
    h = hx_ref.clone().numpy().T

    sigmoid = sp.special.expit
    r = sigmoid( np.matmul(W_ir, x).squeeze() + b_ir + np.matmul(W_hr, h).squeeze() + b_hr)
    z = sigmoid( np.matmul(W_iz, x).squeeze() + b_iz + np.matmul(W_hz, h).squeeze() + b_hz)
    n = np.tanh( np.matmul(W_in, x).squeeze() + b_in + r * (np.matmul(W_hn, h).squeeze() + b_hn))
    hout = (1-z)*n+z*h.squeeze()

    hx_gru=hx_gru.detach().numpy().squeeze()
    dif = hx_gru-hout

    print()



if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    output_path = args["--output-dir"]
    checkpoint_path = args["--checkpoint"]
    preset = args["--preset"]
    no_cuda = args["--no-cuda"]

    device = torch.device("cpu" if no_cuda else "cuda")
    print("using device:{}".format(device))

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])

    mel_file_name = args['<mel_input.npy>']
    mel = np.load(mel_file_name)

    flist = glob.glob(f'{checkpoint_dir}/checkpoint_*.pth')
    latest_checkpoint = max(flist, key=os.path.getctime)
    print('Loading: %s'%latest_checkpoint)
    # build model, create optimizer
    model = build_model().to(device)
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    torch_test(model, checkpoint, mel)

    print("I: %.3f million"%(num_params_count(model.I)))
    print("Upsample: %.3f million"%(num_params_count(model.upsample)))
    print("rnn1: %.3f million"%(num_params_count(model.rnn1)))
    #print("rnn2: %.3f million"%(num_params_count(model.rnn2)))
    print("fc1: %.3f million"%(num_params_count(model.fc1)))
    print("fc2: %.3f million"%(num_params_count(model.fc2)))
    #print("fc3: %.3f million"%(num_params_count(model.fc3)))


    #onnx export
    model.train(False)
    wav = np.load('WaveRNN-Pytorch/checkpoint/test_0_wav.npy')

    #doesn't work torch.onnx.export(model, (torch.tensor(wav),torch.tensor(mel)), checkpoint_dir+'/wavernn.onnx', verbose=True, input_names=['mel_input'], output_names=['wav_output'])


    #mel = np.pad(mel,(24000,0),'constant')
    # n_mels = mel.shape[1]
    # n_mels = hparams.batch_size_gen * (n_mels // hparams.batch_size_gen)
    # mel = mel[:, 0:n_mels]


    mel0 = mel.copy()
    start = time.time()
    output0 = model.generate(mel0, batched=True, target=2000, overlap=64)
    total_time = time.time() - start
    frag_time = len(output0) / hparams.sample_rate
    print("Generation time: {}. Sound time: {}, ratio: {}".format(total_time, frag_time, frag_time/total_time))

    librosa.output.write_wav(os.path.join(output_path, os.path.basename(mel_file_name)+'_orig.wav'), output0, hparams.sample_rate)

    #mel = mel.reshape([mel.shape[0], hparams.batch_size_gen, -1]).swapaxes(0,1)
    #output, out1 = model.batch_generate(mel)
    #bootstrap_len = hp.hop_size * hp.resnet_pad
    #output=output[:,bootstrap_len:].reshape(-1)
    # librosa.output.write_wav(os.path.join(output_path, os.path.basename(mel_file_name)+'.wav'), output, hparams.sample_rate)
    with open(os.path.join(output_path, os.path.basename(mel_file_name)+'.pkl'), 'wb') as f:
        pickle.dump((output0,), f)
    print('done')
