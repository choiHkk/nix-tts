import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import commons
import utils
from data_utils import (
  TextAudioSpeakerLoader,
  TextAudioSpeakerCollate,
  DistributedBucketSampler
)
from vits.models import SynthesizerTrn as TeacherSynthesizerTrn
from models import Encoder, Decoder, MultiPeriodDiscriminator
from losses import DecoderLoss
from text import symbols
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
import natsort

torch.backends.cudnn.benchmark = True
global_step = 0


def synthesizer_train():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  # n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '80000'

  hps_dec = utils.get_hparams(True, 'config_dec.json')
  hps_enc = utils.get_hparams_from_file(hps_dec.data.encoder_model_config)
  hps_teacher = utils.get_hparams_from_file(hps_dec.data.teacher_model_config)

  total_gpu_nums = list(map(str, list(range(torch.cuda.device_count()))))
  if hps_dec.gpu_nums is None:
      gpu_nums = torch.cuda.device_count()
  elif hps_dec.gpu_nums is not None:
      gpu_nums = list(map(str, hps_dec.gpu_nums.split(",")))
      gpu_nums = list(set(total_gpu_nums).intersection(gpu_nums))
  os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(list(map(str, gpu_nums)))
  gpu_nums = list(map(int, gpu_nums))
  gpu_nums = natsort.natsorted(gpu_nums)
  n_gpus = len(gpu_nums)
  print(f"Running DDP with model parallel example on rank {gpu_nums}.")
  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps_dec, hps_enc, hps_teacher,))


def run(rank, n_gpus, hps_dec, hps_enc, hps_teacher):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps_dec.model_dir)
    logger.info(hps_dec)
    utils.check_git_hash(hps_dec.model_dir)
    writer = SummaryWriter(log_dir=hps_dec.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps_dec.model_dir, "eval"))

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps_dec.train.seed)
  torch.cuda.set_device(rank)

  train_dataset = TextAudioSpeakerLoader(hps_dec.data.training_files, hps_dec.data)
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps_dec.train.batch_size,
      [32,300,400,500,600,700,800,900,1000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextAudioSpeakerCollate()
  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler)
  if rank == 0:
    eval_dataset = TextAudioSpeakerLoader(hps_dec.data.validation_files, hps_dec.data)
    eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=False,
        batch_size=hps_dec.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)
    
  loss_dec_fn = DecoderLoss()
    
  # teacher model load
  net_teacher = TeacherSynthesizerTrn(
      len(symbols),
      hps_teacher.data.filter_length // 2 + 1,
      hps_teacher.train.segment_size // hps_teacher.data.hop_length,
      n_speakers=hps_teacher.data.n_speakers,
      **hps_teacher.model).cuda(rank)
  net_teacher, _, _, _ = utils.load_checkpoint(
      utils.latest_checkpoint_path(hps_dec.data.teacher_model_dir, "G_*.pth"), net_teacher, None)
  net_teacher = DDP(net_teacher, device_ids=[rank])
  
  # student decoder model load
  net_dec = Decoder(
    n_speakers=hps_dec.data.n_speakers, 
    **hps_dec.model).cuda(rank)
  net_d = MultiPeriodDiscriminator(hps_dec.model.use_spectral_norm).cuda(rank)
  optim_dec = torch.optim.AdamW(
      net_dec.parameters(), 
      hps_dec.train.learning_rate, 
      betas=hps_dec.train.betas, 
      eps=hps_dec.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps_dec.train.learning_rate, 
      betas=hps_dec.train.betas, 
      eps=hps_dec.train.eps)
  net_dec = DDP(net_dec, device_ids=[rank], find_unused_parameters=True)
  net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)

  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps_dec.model_dir, "DECODER_*.pth"), net_dec, optim_dec)
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps_dec.model_dir, "DECODER_D_*.pth"), net_d, optim_d)
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    epoch_str = 1
    global_step = 0

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_dec, gamma=hps_dec.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps_dec.train.lr_decay, last_epoch=epoch_str-2)

  scaler = GradScaler(enabled=hps_dec.train.fp16_run)

  for epoch in range(epoch_str, hps_dec.train.epochs + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps_dec, [net_dec, net_teacher, net_d], [optim_dec, optim_d], [scheduler_g, scheduler_d], 
                         scaler, [train_loader, eval_loader], logger, [writer, writer_eval], loss_dec_fn)
    else:
      train_and_evaluate(rank, epoch, hps_dec, [net_dec, net_teacher, net_d], [optim_dec, optim_d], [scheduler_g, scheduler_d], 
                         scaler, [train_loader, None], None, None, loss_dec_fn)
    scheduler_g.step()
    scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, loss_fn):
  net_dec, net_teacher, net_d = nets
  optim_dec, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders
  
  if writers is not None:
    writer, writer_eval = writers

  train_loader.batch_sampler.set_epoch(epoch)
  global global_step
  
  net_teacher.eval()
  net_dec.train()
  net_d.train()
  for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(train_loader):
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
    speakers = speakers.cuda(rank, non_blocking=True)
    
    with torch.no_grad():
        z, _, _, g = net_teacher.module.infer_encoder_teacher_mode(
            spec, spec_lengths, speakers)
        
        z_slice, ids_slice = commons.rand_slice_segments(z, spec_lengths, net_teacher.module.segment_size)
        y_hat_t = net_teacher.module.infer_decoder_teacher_mode(z_slice, g)

    with autocast(enabled=hps.train.fp16_run):
      y_hat_s = net_dec(z_slice, g=g)

      # target mel-spectrogram
      mel = spec_to_mel_torch(
          spec, 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate,
          hps.data.mel_fmin, 
          hps.data.mel_fmax)
      y_mel = commons.slice_segments(mel, ids_slice, net_teacher.module.segment_size)
      
      # student mel-spectrogram from teacher model
      y_hat_mel_s = mel_spectrogram_torch(
          y_hat_s.squeeze(1), 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate, 
          hps.data.hop_length, 
          hps.data.win_length, 
          hps.data.mel_fmin, 
          hps.data.mel_fmax
      )
      
      y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice 
      
      # Discriminator
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat_s.detach())
      with autocast(enabled=False):
        loss_dec_disc, losses_dec_disc = loss_fn.disc_loss_fn(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_dec_disc
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    with autocast(enabled=hps.train.fp16_run):
      # decoder
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat_s)
      with autocast(enabled=False):
        loss_dec_gen, losses_dec_gen = loss_fn.gen_loss_fn(
          y_d_hat_g, fmap_r, fmap_g, y_hat_mel_s, y_mel, [y_hat_s, y_hat_t], y)
        
        loss_dec_all = loss_dec_gen
        
    optim_dec.zero_grad()
    scaler.scale(loss_dec_all).backward()
    scaler.unscale_(optim_dec)
    grad_norm_dec = commons.clip_grad_value_(net_dec.parameters(), None)
    scaler.step(optim_dec)
    scaler.update()

    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_dec.param_groups[0]['lr']
        losses_dec = losses_dec_disc
        for k, v in losses_dec_gen.items():
          losses_dec[k] = v
        losses = list(losses_dec.values())
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [grad_norm_dec, global_step, lr])
        
        scalar_dict = {"loss/dec/total": loss_dec_all, "loss/disc/total": loss_disc_all, "learning_rate": lr, "grad_norm_dec": grad_norm_dec, "grad_norm_d": grad_norm_d}
        for k, v in losses_dec.items():
          scalar_dict.update({f"loss/dec/{k}": v})
        image_dict = { 
            "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel_s[0].data.cpu().numpy()), 
            "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
        }
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images=image_dict,
          scalars=scalar_dict)

      if global_step % hps.train.eval_interval == 0:
        evaluate(hps, [net_dec, net_teacher], eval_loader, writer_eval)
        utils.save_checkpoint(net_dec, optim_dec, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "DECODER_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "DECODER_D_{}.pth".format(global_step)))
    global_step += 1
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))

 
def evaluate(hps, nets, eval_loader, writer_eval):
    net_dec, net_teacher = nets
    net_dec.eval()
    with torch.no_grad():
      for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(eval_loader):
        x, x_lengths = x.cuda(0), x_lengths.cuda(0)
        spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
        y, y_lengths = y.cuda(0), y_lengths.cuda(0)
        # speakers = speakers.cuda(rank, non_blocking=True)
        
        speakers = torch.LongTensor([0]).cuda(0)

        # remove else
        x = x[:1]
        x_lengths = x_lengths[:1]
        spec = spec[:1]
        spec_lengths = spec_lengths[:1]
        y = y[:1]
        y_lengths = y_lengths[:1]
        speakers = speakers[:1]
        break
      
      z, _, _, g = net_teacher.module.infer_encoder_teacher_mode(spec, spec_lengths, speakers)
      y_hat = net_dec(z, g)

      mel = spec_to_mel_torch(
        spec, 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate,
        hps.data.mel_fmin, 
        hps.data.mel_fmax)
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
      )
    image_dict = {
      "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
      "gen/audio": y_hat[0,:,:]
    }
    if global_step == 0:
      image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
      audio_dict.update({"gt/audio": y[0,:,:y_lengths[0]]})

    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      images=image_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sampling_rate
    )
    net_dec.train()

                           
if __name__ == "__main__":
  synthesizer_train()
