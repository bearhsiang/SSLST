import argparse
import torchaudio
import math
import numpy as np
import torch

def get_white_noise(signal,SNR) :
    #RMS value of signal
    RMS_s=math.sqrt(np.mean(signal**2))
    #RMS values of noise
    RMS_n=math.sqrt(RMS_s**2/(pow(10,SNR/10)))
    #Additive white gausian noise. Thereore mean=0
    #Because sample length is large (typically > 40000)
    #we can use the population formula for standard daviation.
    #because mean=0 STD=RMS
    STD_n=RMS_n
    noise=np.random.normal(0, STD_n, signal.shape[0])
    return noise

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    parser.add_argument('-n', type=float)
    args = parser.parse_args()
    
    return args

def main(args):

    wav, sr = torchaudio.load(args.input)
    for c in range(wav.size(0)):
        noise = get_white_noise(wav[c].numpy(), args.n)
        wav[c] += noise
    wav = wav.type(torch.float32)
    print(wav, noise)
    torchaudio.save(args.output, wav, sr)
    

if __name__ == '__main__':

    args = get_args()
    main(args)