import os
import sys
import time
import argparse
import traceback
import torch
import torch.nn as nn
from lib import utility
from lib.utils import AverageMeter, convert_secs2time

os.environ["MKL_THREADING_LAYER"] = "GNU"


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def train(args):
    device_ids = args.device_ids
    nprocs = len(device_ids)
    if nprocs > 1:
        torch.multiprocessing.spawn(
            train_worker, args=(nprocs, 1, args), nprocs=nprocs,
            join=True)
    elif nprocs == 1:
        train_worker(device_ids[0], nprocs, 1, args)
    else:
        assert False


def train_worker(world_rank, world_size, nodes_size, args):
    # initialize config.
    config = utility.get_config(args)
    config.device_id = world_rank if nodes_size == 1 else world_rank % torch.cuda.device_count()
    # set environment
    utility.set_environment(config)
    # initialize instances, such as writer, logger and wandb.
    if world_rank == 0:
        config.init_instance()

    if config.logger is not None:
        config.logger.info("\n" + "\n".join(["%s: %s" % item for item in config.__dict__.items()]))
        config.logger.info("Loaded configure file %s: %s" % (config.type, config.id))

    # worker communication
    if world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl", init_method="tcp://localhost:23456" if nodes_size == 1 else "env://",
            rank=world_rank, world_size=world_size)
        torch.cuda.set_device(config.device)

    # model
    net = utility.get_net(config)
    if world_size > 1:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = net.float().to(config.device)
    net.train(True)
    if config.ema and world_rank == 0:
        net_ema = utility.get_net(config)
        if world_size > 1:
            net_ema = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net_ema)
        net_ema = net_ema.float().to(config.device)
        net_ema.eval()
        utility.accumulate_net(net_ema, net, 0)
    else:
        net_ema = None

    # multi-GPU training
    if world_size > 1:
        net_module = nn.parallel.DistributedDataParallel(net, device_ids=[config.device_id],
                                                         output_device=config.device_id, find_unused_parameters=True)
    else:
        net_module = net

    criterions = utility.get_criterions(config)
    optimizer = utility.get_optimizer(config, net_module)
    scheduler = utility.get_scheduler(config, optimizer)

    # load pretrain model
    if args.pretrained_weight is not None:
        if not os.path.exists(args.pretrained_weight):
            pretrained_weight = os.path.join(config.work_dir, args.pretrained_weight)
        else:
            pretrained_weight = args.pretrained_weight

        try:
            checkpoint = torch.load(pretrained_weight)
            net.load_state_dict(checkpoint["net"], strict=False)
            if net_ema is not None:
                net_ema.load_state_dict(checkpoint["net_ema"], strict=False)
            if config.logger is not None:
                config.logger.warn("Successed to load pretrain model %s." % pretrained_weight)
            start_epoch = checkpoint["epoch"]
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
        except:
            start_epoch = 0
            if config.logger is not None:
                config.logger.warn("Failed to load pretrain model %s." % pretrained_weight)
    else:
        start_epoch = 0

    if config.logger is not None:
        config.logger.info("Loaded network")

    # data - train, val
    train_loader = utility.get_dataloader(config, "train", world_rank, world_size)
    if world_rank == 0:
        val_loader = utility.get_dataloader(config, "val")
    if config.logger is not None:
        config.logger.info("Loaded data")

    # forward & backward
    if config.logger is not None:
        config.logger.info("Optimizer type %s. Start training..." % (config.optimizer))
    if not os.path.exists(config.model_dir) and world_rank == 0:
        os.makedirs(config.model_dir)

    # training
    best_metric, best_net = None, None
    epoch_time, eval_time = AverageMeter(), AverageMeter()
    for i_epoch, epoch in enumerate(range(config.max_epoch + 1)):
        try:
            epoch_start_time = time.time()
            if epoch >= start_epoch:
                # forward and backward
                if epoch != start_epoch:
                    utility.forward_backward(config, train_loader, net_module, net, net_ema, criterions, optimizer,
                                             epoch)

                if world_size > 1:
                    torch.distributed.barrier()

                # validating
                if epoch % config.val_epoch == 0 and epoch != 0 and world_rank == 0:
                    eval_start_time = time.time()
                    epoch_nets = {"net": net, "net_ema": net_ema}
                    for net_name, epoch_net in epoch_nets.items():
                        if epoch_net is None:
                            continue
                        result, metrics = utility.forward(config, val_loader, epoch_net)
                        for k, metric in enumerate(metrics):
                            if config.logger is not None and len(metric) != 0:
                                config.logger.info(
                                    "Val_{}/Metric{:3d} in this epoch: [NME {:.6f}, FR {:.6f}, AUC {:.6f}]".format(
                                        net_name, k, metric[0], metric[1], metric[2]))

                        # update best model.
                        cur_metric = metrics[config.key_metric_index][0]
                        if best_metric is None or best_metric > cur_metric:
                            best_metric = cur_metric
                            best_net = epoch_net
                            current_pytorch_model_path = os.path.join(config.model_dir, "best_model.pkl")
                            # current_onnx_model_path = os.path.join(config.model_dir, "train.onnx")
                            utility.save_model(
                                config,
                                epoch,
                                best_net,
                                net_ema,
                                optimizer,
                                scheduler,
                                current_pytorch_model_path)
                    if best_metric is not None:
                        config.logger.info(
                            "Val/Best_Metric%03d in this epoch: %.6f" % (config.key_metric_index, best_metric))
                    eval_time.update(time.time() - eval_start_time)

                # saving model
                if epoch == config.max_epoch and world_rank == 0:
                    current_pytorch_model_path = os.path.join(config.model_dir, "last_model.pkl")
                    # current_onnx_model_path = os.path.join(config.model_dir, "model_epoch_%s.onnx" % epoch)
                    utility.save_model(
                        config,
                        epoch,
                        net,
                        net_ema,
                        optimizer,
                        scheduler,
                        current_pytorch_model_path)

                if world_size > 1:
                    torch.distributed.barrier()

            # adjusting learning rate
            if epoch > 0:
                scheduler.step()
            epoch_time.update(time.time() - epoch_start_time)
            last_time = convert_secs2time(epoch_time.avg * (config.max_epoch - i_epoch), True)
            if config.logger is not None:
                config.logger.info(
                    "Train/Epoch: %d/%d, Learning rate decays to %s, " % (
                        epoch, config.max_epoch, str(scheduler.get_last_lr())) \
                    + last_time + 'eval_time: {:4.2f}, '.format(eval_time.avg) + '\n\n')

        except:
            traceback.print_exc()
            config.logger.error("Exception happened in training steps")

    if config.logger is not None:
        config.logger.info("Training finished")

    try:
        if config.logger is not None and best_metric is not None:
            new_folder_name = config.folder + '-fin-{:.4f}'.format(best_metric)
            new_work_dir = os.path.join(config.ckpt_dir, config.data_definition, new_folder_name)
            os.system('mv {} {}'.format(config.work_dir, new_work_dir))
    except:
        traceback.print_exc()

    if world_size > 1:
        torch.distributed.destroy_process_group()
