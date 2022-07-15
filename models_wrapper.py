from tokenize import Token
from text import symbols,  text_to_sequence
from models import Encoder, Decoder
import commons
import utils

import IPython.display as ipd
import onnxruntime as ort
import numpy as np
import torch.nn as nn
import torch
_device = torch.device("cpu") # cpu inference


def to_device(tensor):
    return tensor.to(_device)


class EncoderWrapper(nn.Module):
    def __init__(self, checkpoint_path, hps_path):
        super(EncoderWrapper, self).__init__()
        hps = utils.get_hparams_from_file(hps_path)
        self.hps = hps
        self.encoder = Encoder(
            n_symbols=len(symbols), 
            n_spec_channels=hps.data.filter_length // 2 + 1, 
            n_speakers=hps.data.n_speakers, 
            **hps.model).to(_device)
        _ = utils.load_checkpoint(checkpoint_path, self.encoder, None)
        
    def forward(self, x, x_lengths, sid=None): 
        x, x_lengths = to_device(x), to_device(x_lengths)
        self.encoder.eval()
        with torch.no_grad():
            (z_hat, *_), _, g = self.encoder.infer(
                x, x_lengths, sid=sid, noise_scale=0.667, noise_scale_w=0.8, length_scale=1.)
        return z_hat, g
        
        
class DecoderWrapper(nn.Module):
    def __init__(self, checkpoint_path, hps_path):
        super(DecoderWrapper, self).__init__()
        hps = utils.get_hparams_from_file(hps_path)
        self.hps = hps
        self.decoder = Decoder(
            n_speakers=hps.data.n_speakers, **hps.model).to(_device)
        _ = utils.load_checkpoint(checkpoint_path, self.decoder, None)
        
    def forward(self, z, g=None):
        z = to_device(z)
        self.decoder.eval()
        with torch.no_grad():
            o_hat = self.decoder(z, g=g)
        return o_hat
    
    
class Tokenizer(object):
    def __init__(self, hps):
        self.hps = hps
        
    def get_text(self, text):
        text_norm = text_to_sequence(text)
        if self.hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        text_lengths = torch.LongTensor([text_norm.size(0)])
        return text_norm, text_lengths
        
    def tokenize(self, text):
        return self.get_text(text)
    
    def get_items(self, text):
        x, x_lengths = self.tokenize(text)
        x = x.unsqueeze(0)
        return (x, x_lengths)
    
    
class NixttsTorchSession(object):
    def __init__(self, enc_ckpt_path, dec_ckpt_path, enc_hps_path, dec_hps_path):
        self.enc_hps_path = enc_hps_path
        self.dec_hps_path = dec_hps_path
        self.enc_ckpt_path = enc_ckpt_path
        self.dec_ckpt_path = dec_ckpt_path
        self.initialize()
        
    def initialize(self):
        self.encoder_wrapper = EncoderWrapper(
            self.enc_ckpt_path, self.enc_hps_path)
        self.decoder_wrapper = DecoderWrapper(
            self.dec_ckpt_path, self.dec_hps_path)
        self.tokenizer = Tokenizer(self.encoder_wrapper.hps)
        
    def __call__(self, text, sid=0):
        x, x_lengths = self.tokenizer.get_items(text)
        sid = torch.LongTensor([sid])
        z_hat, g = self.encoder_wrapper(x, x_lengths, sid=sid)
        o_hat = self.decoder_wrapper(z_hat, g=g)[0][0].data.cpu().numpy()
        return o_hat
