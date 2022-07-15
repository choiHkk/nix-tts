from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
import torch.nn.functional as F
import torch.nn as nn
import torch
import math

import commons
import monotonic_align
from commons import init_weights, get_padding
from modules import (
    PositionalEncoding, 
    TextResidualBlock, 
    LinearNorm, 
    ConvNorm, 
    LayerNorm, 
)
import modules


class StochasticDurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
    super().__init__()
    filter_channels = in_channels # it needs to be removed from future version.
    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.log_flow = modules.Log()
    self.flows = nn.ModuleList()
    self.flows.append(modules.ElementwiseAffine(2))
    for i in range(n_flows):
      self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.flows.append(modules.Flip())

    self.post_pre = nn.Conv1d(1, filter_channels, 1)
    self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    self.post_flows = nn.ModuleList()
    self.post_flows.append(modules.ElementwiseAffine(2))
    for i in range(4):
      self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.post_flows.append(modules.Flip())

    self.pre = nn.Conv1d(in_channels, filter_channels, 1)
    self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

  def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
    x = torch.detach(x)
    x = self.pre(x)
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)
    x = self.convs(x, x_mask)
    x = self.proj(x) * x_mask

    if not reverse:
      flows = self.flows
      assert w is not None

      logdet_tot_q = 0 
      h_w = self.post_pre(w)
      h_w = self.post_convs(h_w, x_mask)
      h_w = self.post_proj(h_w) * x_mask
      e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
      z_q = e_q
      for flow in self.post_flows:
        z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
        logdet_tot_q += logdet_q
      z_u, z1 = torch.split(z_q, [1, 1], 1) 
      u = torch.sigmoid(z_u) * x_mask
      z0 = (w - u) * x_mask
      logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1,2])
      logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1,2]) - logdet_tot_q

      logdet_tot = 0
      z0, logdet = self.log_flow(z0, x_mask)
      logdet_tot += logdet
      z = torch.cat([z0, z1], 1)
      for flow in flows:
        z, logdet = flow(z, x_mask, g=x, reverse=reverse)
        logdet_tot = logdet_tot + logdet
      nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1,2]) - logdet_tot
      return nll + logq # [b]
    else:
      flows = list(reversed(self.flows))
      flows = flows[:-2] + [flows[-1]] # remove a useless vflow
      z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
      for flow in flows:
        z = flow(z, x_mask, g=x, reverse=reverse)
      z0, z1 = torch.split(z, [1, 1], 1)
      logw = z0
      return logw


class TextEncoder(nn.Module):
    def __init__(self, n_symbols, hidden_channels, kernel_size, dilations, max_len):
        super(TextEncoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.emb = nn.Embedding(n_symbols, hidden_channels, padding_idx=0)
        self.emb_p = PositionalEncoding(max_len, hidden_channels)

        self.prenet = nn.Sequential(
            modules.DDSConv(hidden_channels, 1, 3, p_dropout=0.1),
            nn.SiLU(),
            LayerNorm(hidden_channels)
        )

        self.res_blocks = nn.Sequential(*[
            TextResidualBlock(hidden_channels, kernel_size, d, n=2)
            for d in dilations
        ])

        self.post_net1 = nn.Sequential(
            modules.DDSConv(hidden_channels, 1, 3, p_dropout=0.1),
        )

        self.post_net2 = nn.Sequential(
            nn.SiLU(),
            LayerNorm(hidden_channels), 
            modules.DDSConv(hidden_channels, 1, 5, p_dropout=0.1),
        )

    def forward(self, x, x_lengths):
        x = self.emb(x) + self.emb_p(x)
        x = x.transpose(1,2)
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.prenet(x * x_mask) * x_mask
        x = self.res_blocks(x) * x_mask
        x = self.post_net1(x) + x
        x = self.post_net2(x * x_mask) * x_mask
        return x, x_mask

    
class AlignmentEncoder(torch.nn.Module):
    """ Alignment Encoder for Unsupervised Duration Modeling """
    """From comprehensive transformer tts"""

    def __init__(self, n_spec_channels, n_att_channels, 
                 n_text_channels, gin_channels, temperature, multi_speaker):
        super(AlignmentEncoder, self).__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)

        self.key_proj = nn.Sequential(
            ConvNorm(
                n_text_channels,
                n_text_channels * 2,
                kernel_size=3,
                bias=True,
                w_init_gain='relu'
            ),
            nn.ReLU(),
            ConvNorm(
                n_text_channels * 2,
                n_att_channels,
                kernel_size=1,
                bias=True,
            ),
        )

        self.query_proj = nn.Sequential(
            ConvNorm(
                n_spec_channels,
                n_spec_channels * 2,
                kernel_size=3,
                bias=True,
                w_init_gain='relu',
            ),
            nn.ReLU(),
            ConvNorm(
                n_spec_channels * 2,
                n_spec_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.ReLU(),
            ConvNorm(
                n_spec_channels,
                n_att_channels,
                kernel_size=1,
                bias=True,
            ),
        )

        if multi_speaker:
            self.key_spk_proj = LinearNorm(gin_channels, n_text_channels)
            self.query_spk_proj = LinearNorm(gin_channels, n_spec_channels)

    def forward(self, queries, keys, mask=None, attn_prior=None, g=None):
        """Forward pass of the aligner encoder.
        Args:
            queries (torch.tensor): B x C x T1 tensor (probably going to be mel data).
            keys (torch.tensor): B x C2 x T2 tensor (text data).
            mask (torch.tensor): uint8 binary mask for variable length entries (should be in the T2 domain).
            attn_prior (torch.tensor): prior for attention matrix.
            g (torch.tensor): B x C tnesor of speaker embedding for multi-speaker scheme.
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask. Final dim T2 should sum to 1.
            attn_logprob (torch.tensor): B x 1 x T1 x T2 log-prob attention mask.
        """
        if g is not None:
            keys = keys + self.key_spk_proj(g.unsqueeze(1).expand(
                -1, keys.shape[-1], -1
            )).transpose(1, 2)
            queries = queries + self.query_spk_proj(g.unsqueeze(1).expand(
                -1, queries.shape[-1], -1
            )).transpose(1, 2)
        keys_enc = self.key_proj(keys)  # B x n_attn_dims x T2
        queries_enc = self.query_proj(queries)

        # Simplistic Gaussian Isotopic Attention
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None]) ** 2  # B x n_attn_dims x T1 x T2
        attn = -self.temperature * attn.sum(1, keepdim=True)

        if attn_prior is not None:
            #print(f"AlignmentEncoder \t| mel: {queries.shape} phone: {keys.shape} mask: {mask.shape} attn: {attn.shape} attn_prior: {attn_prior.shape}")
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None] + 1e-8)
            #print(f"AlignmentEncoder \t| After prior sum attn: {attn.shape}")

        attn_logprob = attn.clone()

        if mask is not None:
            mask = mask.unsqueeze(1)
            # attn.data.masked_fill_(mask.bool(), -float("inf"))
            mask = mask.data.eq(0).float()
            mask[mask==1] = (-float("inf"))
            attn = attn + mask
            
        attn = self.softmax(attn)  # softmax along T2
        return attn, attn_logprob
    
    
class LatentEncoder(nn.Module):
    def __init__(self, hidden_channels, output_channels, kernel_size, dilations):
        super(LatentEncoder, self).__init__()
        self.output_channels = output_channels
        self.prenet = nn.Sequential(
            modules.DDSConv(hidden_channels, 1, 3, p_dropout=0.1),
            nn.SiLU(),
            LayerNorm(hidden_channels)
        )

        self.res_blocks = nn.Sequential(*[
            TextResidualBlock(hidden_channels, kernel_size, d, n=2)
            for d in dilations
        ])

        self.post_net1 = nn.Sequential(
            modules.DDSConv(hidden_channels, 1, 3, p_dropout=0.1),
        )

        self.post_net2 = nn.Sequential(
            nn.SiLU(),
            LayerNorm(hidden_channels), 
            modules.DDSConv(hidden_channels, 1, 5, p_dropout=0.1),
        )
        self.projection = nn.Sequential(
            nn.Conv1d(hidden_channels, output_channels*2, 1),
        )

    def forward(self, x, x_mask):
        x = self.prenet(x * x_mask) * x_mask
        x = self.res_blocks(x) * x_mask
        x = self.post_net1(x) + x
        x = self.post_net2(x * x_mask) * x_mask
        x = self.projection(x) * x_mask
        m, logs = torch.split(x, self.output_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs

    
class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.DSResBlock if resblock == 'ds' else modules.ResBlock1

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(128, 256, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(256, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(512, 512, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(512, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=32, padding=20)),
            norm_f(nn.Conv1d(512, 512, 41, 4, groups=128, padding=20)),
            norm_f(nn.Conv1d(512, 512, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(512, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2,3,5,7,11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class Encoder(nn.Module):
    def __init__(self, 
        n_symbols, 
        hidden_channels, 
        inter_channels, 
        kernel_size, 
        dilations, 
        max_len,
        n_spec_channels, 
        temperature, 
        p_dropout, 
        n_flows, 
        n_speakers, 
        gin_channels, 
        **kwargs):
        super(Encoder, self).__init__()
        self.n_speakers = n_speakers
        
        if n_speakers > 0:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
            multi_speaker = True
        else:
            multi_speaker = False
            gin_channels = 0

        self.text_encoder = TextEncoder(
            n_symbols=n_symbols, 
            hidden_channels=hidden_channels, 
            kernel_size=kernel_size, 
            dilations=dilations, 
            max_len=max_len
        )
        
        self.aligner = AlignmentEncoder(
            n_spec_channels=n_spec_channels, 
            n_att_channels=hidden_channels, 
            n_text_channels=hidden_channels, 
            gin_channels=gin_channels, 
            temperature=temperature, 
            multi_speaker=multi_speaker
        )
        
        self.dp = StochasticDurationPredictor(
            in_channels=hidden_channels, 
            filter_channels=hidden_channels, 
            kernel_size=3, 
            p_dropout=p_dropout, 
            n_flows=n_flows, 
            gin_channels=gin_channels
        )
        
        self.latent_encoder = LatentEncoder(
            hidden_channels=hidden_channels, 
            output_channels=inter_channels, 
            kernel_size=kernel_size, 
            dilations=dilations, 
        )
        
    def forward(self, x, x_lengths, y, y_lengths, sid=None):
        x, x_mask = self.text_encoder(x, x_lengths)
        
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, y.size(2)), 1).to(y.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)
        else:
            g = None
            
        attn_s, attn_logprob = self.aligner(
            queries=y, keys=x, mask=x_mask, g=g.squeeze(-1) if g is not None else None)
        attn_h = monotonic_align.maximum_path(attn_s.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1).detach()
        
        w = attn_h.sum(2)
        l_length = self.dp(x, x_mask, w, g=g)
        
        x = torch.matmul(x, attn_h.squeeze(1).transpose(1,2))
        
        z_hat, m_hat, logs_hat = self.latent_encoder(x, y_mask)
        return (z_hat, m_hat, logs_hat, l_length), (attn_h, attn_s, attn_logprob), (x_mask, y_mask, attn_mask), g
    
    def infer(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
        x, x_mask = self.text_encoder(x, x_lengths)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)
        else:
            g = None
            
        logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn_h = commons.generate_path(w_ceil, attn_mask)
        
        x = torch.matmul(x, attn_h.squeeze(1).transpose(1,2))
        
        _, m_hat, logs_hat = self.latent_encoder(x, y_mask)
        z_hat = m_hat + torch.randn_like(m_hat) * torch.exp(logs_hat) * noise_scale
        return (z_hat, m_hat, logs_hat, attn_h), (x_mask, y_mask, attn_mask), g
        
        
class Decoder(nn.Module):
    def __init__(self, 
        inter_channels, 
        resblock, 
        resblock_kernel_sizes, 
        resblock_dilation_sizes, 
        upsample_rates, 
        upsample_initial_channel, 
        upsample_kernel_sizes, 
        n_speakers, 
        gin_channels, 
        **kwargs):
        super(Decoder, self).__init__()
        self.n_speakers = n_speakers
        
        self.generator = Generator(
            initial_channel=inter_channels, 
            resblock=resblock, 
            resblock_kernel_sizes=resblock_kernel_sizes, 
            resblock_dilation_sizes=resblock_dilation_sizes, 
            upsample_rates=upsample_rates, 
            upsample_initial_channel=upsample_initial_channel, 
            upsample_kernel_sizes=upsample_kernel_sizes, 
            gin_channels=gin_channels
        )
        
    def forward(self, z, g=None):
        if not self.n_speakers > 0:
            g = None
        o_hat = self.generator(z, g=g)
        return o_hat
