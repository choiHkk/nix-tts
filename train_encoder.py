import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import os
import torch
import torch.nn as nn
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
from models import Encoder
from losses import EncoderLoss
from text import symbols
from vits.models import SynthesizerTrn as TeacherSynthesizerTrn
import natsort

torch.backends.cudnn.benchmark = True
global_step = 0


def synthesizer_train():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  # n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '80000'

  hps = utils.get_hparams(True, 'config_enc.json')
  hps_teacher = utils.get_hparams_from_file(hps.data.teacher_model_config)

  total_gpu_nums = list(map(str, list(range(torch.cuda.device_count()))))
  if hps.gpu_nums is None:
      gpu_nums = torch.cuda.device_count()
  elif hps.gpu_nums is not None:
      gpu_nums = list(map(str, hps.gpu_nums.split(",")))
      gpu_nums = list(set(total_gpu_nums).intersection(gpu_nums))
  os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(list(map(str, gpu_nums)))
  gpu_nums = list(map(int, gpu_nums))
  gpu_nums = natsort.natsorted(gpu_nums)
  n_gpus = len(gpu_nums)
  print(f"Running DDP with model parallel example on rank {gpu_nums}.")
  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps, hps_teacher,))


def run(rank, n_gpus, hps, hps_teacher):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)

  train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [32,300,400,500,600,700,800,900,1000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextAudioSpeakerCollate()
  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler)
    
  loss_enc_fn = EncoderLoss()
    
  # teacher model load
  net_teacher = TeacherSynthesizerTrn(
      len(symbols),
      hps_teacher.data.filter_length // 2 + 1,
      hps_teacher.train.segment_size // hps_teacher.data.hop_length,
      n_speakers=hps_teacher.data.n_speakers,
      **hps_teacher.model).cuda(rank)
  net_teacher, _, _, _ = utils.load_checkpoint(
      utils.latest_checkpoint_path(hps.data.teacher_model_dir, "G_*.pth"), net_teacher, None)
  net_teacher = DDP(net_teacher, device_ids=[rank], find_unused_parameters=True)
  
  # student model load
  net_enc = Encoder(
    n_symbols=len(symbols), 
    n_spec_channels=hps.data.filter_length // 2 + 1, 
    n_speakers=hps.data.n_speakers, 
    **hps.model).cuda(rank)

  optim_enc = torch.optim.AdamW(
      net_enc.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  net_enc = DDP(net_enc, device_ids=[rank], find_unused_parameters=True)
  
  ## version 1 
  net_enc.module.emb_g = net_teacher.module.emb_g

  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "ENCODER_*.pth"), net_enc, optim_enc)
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    epoch_str = 1
    global_step = 0

  scheduler_enc = torch.optim.lr_scheduler.ExponentialLR(optim_enc, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scaler = GradScaler(enabled=hps.train.fp16_run)
  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train(rank, epoch, hps, [net_teacher, net_enc], optim_enc, scheduler_enc, 
                         scaler, train_loader, logger, writer, loss_enc_fn)
    else:
      train(rank, epoch, hps, [net_teacher, net_enc], optim_enc, scheduler_enc, 
                         scaler, [train_loader, None], None, None, loss_enc_fn)
    scheduler_enc.step()


def train(rank, epoch, hps, nets, optim, scheduler, scaler, train_loader, logger, writer, loss_fn):
  net_teacher, net_enc = nets
  
  train_loader.batch_sampler.set_epoch(epoch)
  global global_step
  
  net_teacher.eval()
  net_enc.train()
  for batch_idx, (x, x_lengths, spec, spec_lengths, _, _, speakers) in enumerate(train_loader):
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
    speakers = speakers.cuda(rank, non_blocking=True)
    
    with torch.no_grad():
        _, m, logs, _ = net_teacher.module.infer_encoder_teacher_mode(
            spec, spec_lengths, speakers)

    with autocast(enabled=hps.train.fp16_run):
      (_, m_hat, logs_hat, l_length),\
        (attn_h, attn_s, attn_logprob), (_, y_mask, _), _ = net_enc(
          x, x_lengths, spec, spec_lengths, speakers)
      with autocast(enabled=False):
        loss_dur = torch.sum(l_length.float())
        loss_enc, losses_enc = loss_fn(
          attn_logprob, x_lengths, spec_lengths, attn_h, attn_s, m, logs, m_hat, logs_hat, y_mask, global_step)
        loss_enc_all = loss_enc + loss_dur
        
    optim.zero_grad()
    scaler.scale(loss_enc_all).backward()
    scaler.unscale_(optim)
    grad_norm_enc = commons.clip_grad_value_(net_enc.parameters(), 5.0)
    scaler.step(optim)
    scaler.update()

    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim.param_groups[0]['lr']
        losses_enc['l_dur'] = loss_dur
        losses = list(losses_enc.values())
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([l.item() for l in losses] + [grad_norm_enc, global_step, lr])
        
        scalar_dict = {"loss/enc/total": loss_enc_all, "learning_rate": lr, "grad_norm_enc": grad_norm_enc}
        for k, v in losses_enc.items():
          scalar_dict.update({f"loss/enc/{k}": v})
        image_dict = { 
            "all/attn": utils.plot_alignment_to_numpy(attn_h[0,0].data.cpu().numpy())
        }
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images=image_dict,
          scalars=scalar_dict)
        
      if global_step % hps.train.eval_interval == 0:
        utils.save_checkpoint(net_enc, optim, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "ENCODER_{}.pth".format(global_step)))
    global_step += 1
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))

if __name__ == "__main__":
  synthesizer_train()
