import torch 
import torch.nn as nn
from torch.nn import functional as F
from stft_loss import MultiResolutionSTFTLoss
        
        
class ForwardSumLoss(nn.Module):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=3)
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: query_lens[bid], :, : key_lens[bid] + 1]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss += loss

        total_loss /= attn_logprob.shape[0]
        return total_loss


class BinLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hard_attention, soft_attention):
        log_sum = torch.log(torch.clamp(soft_attention[hard_attention == 1], min=1e-12)).sum()
        return -log_sum / hard_attention.sum()
    
    
class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, m_t, logs_t, m_s, logs_s, z_mask):
        return self.kl_loss(m_t, logs_t, m_s, logs_s, z_mask)
    
    def kl_loss(self, m_t, logs_t, m_s, logs_s, z_mask):
        """
        m_t, logs_t: [b, h, t_t]
        m_s, logs_s: [b, h, t_t]
        """
        m_t = m_t.float()
        logs_t = logs_t.float()
        m_s = m_s.float()
        logs_s = logs_s.float()
        z_mask = z_mask.float()
        
        kl = logs_t - logs_s - 0.5

        kl += 0.5 * (torch.exp(2. * logs_s) + ((m_s - m_t)**2)) / torch.exp(2. * logs_t)
        kl = torch.sum(kl * z_mask)
        l = kl / torch.sum(z_mask)
        return l
        
    
class AdversarialDisciriminatorLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
    
    def forward(self, disc_real_outputs, disc_generated_outputs):
        return self.discriminator_loss(disc_real_outputs, disc_generated_outputs)
    
    def discriminator_loss(self, disc_real_outputs, disc_generated_outputs):
        loss = 0
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            dr = dr.float()
            dg = dg.float()
            r_loss = torch.mean((1-dr)**2)
            g_loss = torch.mean(dg**2)
            loss += (r_loss + g_loss)
        return loss

    
class AdversarialGeneratorLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        
    def forward(self, disc_outputs):
        return self.generator_loss(disc_outputs)
    
    def generator_loss(self, disc_outputs):
        loss = 0
        for dg in disc_outputs:
            dg = dg.float()
            l = torch.mean((1-dg)**2)
            loss += l
        return loss
    

class FeatureMatchingLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        
    def forward(self, fmap_r, fmap_g):
        return self.feature_loss(fmap_r, fmap_g)

    def feature_loss(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                rl = rl.float().detach()
                gl = gl.float()
                loss += torch.mean(torch.abs(rl - gl))
        return loss * 2 
    

class EncoderLoss(nn.Module):
    def __init__(self, binarization_loss_enable_steps=18000, binarization_loss_warmup_steps=10000):
        super(EncoderLoss, self).__init__()
        """
        L_e = L_forwardsum + L_bin + L_kl
        """
        self.binarization_loss_enable_steps = binarization_loss_enable_steps
        self.binarization_loss_warmup_steps = binarization_loss_warmup_steps
        self.forawrd_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()
        self.kl_loss = KLDivergenceLoss()
    
    def forward(self, attn_logprob, in_lens, out_lens, hard_attention, soft_attention, m_t, logs_t, m_s, logs_s, z_mask, step):
        l_forwardsum = self.forawrd_loss(attn_logprob, in_lens, out_lens)
        if step < self.binarization_loss_enable_steps:
            bin_loss_weight = 0.
        else:
            bin_loss_weight = min((step-self.binarization_loss_enable_steps) / self.binarization_loss_warmup_steps, 1.0) * 1.0
        l_bin = self.bin_loss(hard_attention, soft_attention) * bin_loss_weight
        l_kl = self.kl_loss(m_t, logs_t, m_s, logs_s, z_mask)
        loss = l_forwardsum + l_bin + l_kl
        losses = {"l_forwardsum": l_forwardsum, "l_bin": l_bin, "l_kl": l_kl}
        return loss, losses

    
class DecoderLoss(nn.Module):
    def __init__(self, c_mel=20., c_stft=1.):
        super(DecoderLoss, self).__init__()
        """
        L_d = L_adv,disc + L_adv,gen + L_fmatch + L_recon + L_ged(L_stft)
        """
        self.c_mel = c_mel
        self.c_stft = c_stft
        self.advdisc_loss = AdversarialDisciriminatorLoss()
        self.advgen_loss = AdversarialGeneratorLoss()
        self.fm_loss = FeatureMatchingLoss()
        self.recon_loss = torch.nn.L1Loss()
        self.mr_stft_loss_fn = MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 512], 
            hop_sizes=[128, 256, 64], 
            win_lengths=[1024, 2048, 512])
        
    def disc_loss_fn(self, disc_real_outputs, disc_generated_outputs):
        l_advdisc = self.advdisc_loss(disc_real_outputs, disc_generated_outputs)
        losses = {"l_advdisc": l_advdisc}
        return l_advdisc, losses
    
    def gen_loss_fn(self, disc_outputs, fmap_r, fmap_g, y_hat_mel, y_mel, o_hats, o):
        o_hat_w_s, o_w_t = o_hats
        l_advgen = self.advgen_loss(disc_outputs)
        l_fmatch = self.fm_loss(fmap_r, fmap_g)
        l_recon = self.c_mel * self.recon_loss(y_hat_mel, y_mel)
        l_stft = (self.c_stft * self.mr_stft_loss_fn(o_hat_w_s.squeeze(1), o.squeeze(1))) + self.mr_stft_loss_fn(o_hat_w_s.squeeze(1), o_w_t.squeeze(1))
        
        loss = l_advgen + l_fmatch + l_recon + l_stft
        losses = {"l_advgen": l_advgen, "l_fmatch": l_fmatch, "l_recon": l_recon, "l_stft": l_stft}
        return loss, losses
    