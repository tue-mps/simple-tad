import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from copy import deepcopy
from pathlib import Path

import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from timm.models import create_model
from optim_factory import create_optimizer
from datasets_frame import build_pretraining_dataset
from engine_for_pretraining import train_one_epoch_double
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils


class CyclicDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)

    def set_epoch(self, epoch):
        if hasattr(self.dataloader.sampler, "set_epoch"):
            self.dataloader.sampler.set_epoch(epoch)


def get_args():
    parser = argparse.ArgumentParser('VideoMAE pre-training script', add_help=False)
    parser.add_argument('--batch_size1', default=64, type=int)
    parser.add_argument('--batch_size2', default=64, type=int)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)

    # Model parameters
    parser.add_argument('--model', default='pretrain_videomae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--from_ckpt', default=None, type=str, help='Path of the ckpt with which initialize the model')

    parser.add_argument('--decoder_depth', default=4, type=int,
                        help='depth of decoder')

    parser.add_argument('--mask_type', default='tube', choices=['random', 'tube'],
                        type=str, help='masked strategy of video tokens/patches')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')

    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size for backbone')

    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
                        
    parser.add_argument('--normlize_target', default=True, type=bool,
                        help='normalized the target patch pixels')
    parser.add_argument('--tubelet_size', type=int, default=2)

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--use_checkpoint', action='store_true')
    parser.set_defaults(use_checkpoint=False)

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.0, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--aa', type=str, default='rand-m3-n3-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m3-n3-mstd0.5-inc1)'),
    parser.add_argument('--transforms_finetune_align', action='store_true')
    parser.set_defaults(transforms_finetune_align=False)

    # Dataset parameters
    parser.add_argument('--data_path1', default='/path/to/list_kinetics-400', type=str,
                        help='dataset path')
    parser.add_argument('--data_path2', default='/path/to/list_kinetics-400', type=str,
                        help='dataset path')
    parser.add_argument('--view_fps', type=int, default=10)
    parser.add_argument('--data_set1', default='Kinetics-400',
                        choices=['Kinetics-400', 'SSV2', 'UCF101', 'HMDB51', 'DoTA', 'DADA2K', 'CAP-DATA', 'SHIFT', 'BDD100K', 'image_folder'],
                        type=str, help='dataset')
    parser.add_argument('--data_set2', default='Kinetics-400',
                        choices=['Kinetics-400', 'SSV2', 'UCF101', 'HMDB51', 'DoTA', 'DADA2K', 'CAP-DATA', 'SHIFT', 'BDD100K', 'image_folder'],
                        type=str, help='dataset')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_frames', type=int, default= 16)
    parser.add_argument('--sampling_rate1', type=int, default= 4)
    parser.add_argument('--sampling_rate2', type=int, default= 4)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--nb_samples_per_epoch', default=0, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        decoder_depth=args.decoder_depth,
        use_checkpoint=args.use_checkpoint
    )
    return model


def main(args):
    #print("os environ")
    #print(os.environ)

    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // args.tubelet_size, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    # get datasets
    args1 = deepcopy(args)
    args2 = deepcopy(args)
    args1.data_set = args.data_set1
    args1.data_path = args.data_path1
    args1.sampling_rate = args.sampling_rate1
    args2.data_set = args.data_set2
    args2.data_path = args.data_path2
    args2.sampling_rate = args.sampling_rate2
    dataset_train1 = build_pretraining_dataset(is_train=True, args=args1)
    dataset_train2 = build_pretraining_dataset(is_train=True, args=args2)

    print("Datasets are ready, full len: ")
    print(f"\t - 1: {args.data_set1} {len(dataset_train1)}")
    print(f"\t - 2: {args.data_set2} {len(dataset_train2)}")

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_rank = global_rank

    total_batch_size = (args.batch_size1 + args.batch_size2) * num_tasks
    num_batches = int(round(args.nb_samples_per_epoch / (args.batch_size1 + args.batch_size2)))
    nb_samples_per_epoch1 = args.batch_size1 * num_batches
    nb_samples_per_epoch2 = args.batch_size2 * num_batches

    if args.nb_samples_per_epoch and (nb_samples_per_epoch1 < len(dataset_train1)):
        sampler_train1 = utils.ShortDistributedSampler(
            dataset_train1, num_replicas=num_tasks, rank=sampler_rank, shuffle=True,
            num_samples_per_epoch=nb_samples_per_epoch1
        )
    else:
        sampler_train1 = torch.utils.data.DistributedSampler(
            dataset_train1, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
        )
    if args.nb_samples_per_epoch and (nb_samples_per_epoch2 < len(dataset_train2)):
        sampler_train2 = utils.ShortDistributedSampler(
            dataset_train2, num_replicas=num_tasks, rank=sampler_rank, shuffle=True,
            num_samples_per_epoch=nb_samples_per_epoch2
        )
    else:
        sampler_train2 = torch.utils.data.DistributedSampler(
            dataset_train2, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
        )
    num_training_steps_per_epoch = (sampler_train1.total_size + sampler_train2.total_size) // total_batch_size
    
    print("Sampler_train = ")
    print(f"\t- 1: {str(sampler_train1)}, len: {sampler_train1.total_size}")
    print(f"\t- 2: {str(sampler_train2)}, len: {sampler_train2.total_size}")
    print(f"num_training_steps_per_epoch: {num_training_steps_per_epoch}")

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train1 = torch.utils.data.DataLoader(
        dataset_train1, sampler=sampler_train1,
        batch_size=args.batch_size1,  # Half-batch for dataset 1
        num_workers=args.num_workers // 2,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=utils.seed_worker,
        persistent_workers=True,
        prefetch_factor=2,
    )
    data_loader_train2 = torch.utils.data.DataLoader(
        dataset_train2, sampler=sampler_train2,
        batch_size=args.batch_size2,  # Half-batch for dataset 2
        num_workers=args.num_workers // 2,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=utils.seed_worker,
        persistent_workers=True,
        prefetch_factor=2,
    )
        # Find the smaller dataset
    if len(data_loader_train1) < len(data_loader_train2):
        smaller_loader, larger_loader = data_loader_train1, data_loader_train2
    else:
        smaller_loader, larger_loader = data_loader_train2, data_loader_train1
        if len(data_loader_train1) != len(data_loader_train2):
            print(f"WARNING len(data_loader_train1) != len(data_loader_train2): {len(data_loader_train1)} != {len(data_loader_train2)}")
    # Cycle through the smaller DataLoader
    smaller_loader = CyclicDataLoader(smaller_loader)

    if args.from_ckpt:
        if args.from_ckpt.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.from_ckpt, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.from_ckpt, map_location='cpu')

        print("Load ckpt from %s" % args.from_ckpt)
        checkpoint_model = None
        for model_key in ("module", "model"):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint

        # interpolate position embedding
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
            num_patches = model.encoder.patch_embed.num_patches #
            num_extra_tokens = model.encoder.pos_embed.shape[-2] - num_patches # 0/1

            # height (== width) for the checkpoint position embedding
            orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(args.num_frames // model.encoder.patch_embed.tubelet_size)) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int((num_patches // (args.num_frames // model.encoder.patch_embed.tubelet_size) )** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(-1, args.num_frames // model.encoder.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, args.num_frames // model.encoder.patch_embed.tubelet_size, new_size, new_size, embedding_size)
                pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

                new_dict = {}

        # Only for fine-tune weights
        new_dict = {}
        for k in checkpoint_model.keys():
            if k == "pos_embed":
                value = checkpoint_model[k]
                new_dict[f"encoder.{k}"] = value
            if k.startswith("patch_embed"):
                value = checkpoint_model[k]
                new_dict[f"encoder.{k}"] = value
            if k.startswith("blocks."):
                value = checkpoint_model[k]
                new_dict[f"encoder.{k}"] = value
            if k.startswith("fc_norm"):
                value = checkpoint_model[k]
                k = k.replace("fc_norm", "norm")
                new_dict[f"encoder.{k}"] = value
        checkpoint_model = new_dict

        utils.load_state_dict(model, checkpoint_model)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params: {} M'.format(n_parameters / 1e6))

    args._base_lr = args.lr
    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    if args.output_dir and utils.is_main_process():
        with open(os.path.join(args.output_dir, "params.json"), mode="w") as f:
            json.dump(vars(args), f, indent=2)
        grad_norm_dir = os.path.join(args.output_dir, "grad_norms")
        os.makedirs(grad_norm_dir, exist_ok=True)

    # 
    print("Try to do auto-load...")
    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, latest=True)
    torch.cuda.empty_cache()
    print(f"Start training for {args.epochs} epochs")

    print(f"BEFORE any forward and back props:")
    utils.print_memory_usage()

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if epoch > 11:
            print("We only train for 12 epochs!")
            break
        
        if args.distributed:
            larger_loader.sampler.set_epoch(epoch)
            smaller_loader.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats, grad_norms = train_one_epoch_double(
            model, larger_loader, smaller_loader,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            patch_size=patch_size[0],
            normlize_target=args.normlize_target,
        )

        if utils.is_main_process() and grad_norms is not None:
            assert np.max(grad_norms["qkv"]) > 0., "grad_norms < 0!! "
            print(f"Epoch {epoch}, max grad_norms qkv: {np.max(grad_norms["qkv"]):.2f}")
            np.savez(os.path.join(grad_norm_dir, f"gradnorm_ep{epoch}.npz"), **grad_norms)

        if log_writer is not None:
            log_writer.update(iter=epoch * num_training_steps_per_epoch, head="my_train", step=epoch)
            log_writer.update(epoch=epoch, head="my_train", step=epoch * num_training_steps_per_epoch)
        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model_weights_only(
                    args=args, epoch=epoch, model_without_ddp=model_without_ddp)
            # save last model with all the parameters so we can continue from it
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, epoch_name="last"
                )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
