import torch
import argparse
import yaml
from utils import *
from engines import *
import numpy as np
import random
import time
import datetime
import json
from vae import VAE
from pathlib import Path
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import (
    DistributedSampler, 
    BatchSampler,
    RandomSampler,
    SequentialSampler,
    DataLoader,
)
from tqdm import tqdm   
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
                        '-c',
                        dest="filename",
                        metavar='FILE',
                        help="path to config yaml",
                        default='./config.yaml',
                        type=str)
    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise RuntimeError("配置文件导入出错！") from exc
    return DictX(config)

def main(config):
    init_distributed_mode(config)
    
    device = torch.device(config.device)
    seed = config.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = VAE(config.in_channels, config.latent_dim, config.hidden_dim)
    model.to(device)

    model_without_ddp = model
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    optimizer = optim.Adam(model_without_ddp.parameters(), lr=config.LR, weight_decay=config.weight_decay)
    
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.scheduler_gamma)

    transform=transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = MNIST(config.data_path, train=True, download=True, transform=transform)
    val_dataset = MNIST(config.data_path, train=False, download=True, transform=transform)

    if config.distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)
    
    batch_train_sampler = BatchSampler(train_sampler, config.batch_size, drop_last=True)
    batch_val_sampler = BatchSampler(val_sampler, config.batch_size, drop_last=False)

    train_loader = DataLoader(train_dataset, 
                              batch_sampler=batch_train_sampler, 
                              num_workers=config.num_workers,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset,
                            batch_sampler=batch_val_sampler,
                            drop_last=False,
                            num_workers=config.num_workers,
                            collate_fn=collate_fn)

    output_dir = Path(config.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if config.resume:
        if config.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                config.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(config.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            config.start_epoch = checkpoint['epoch'] + 1
    
    print("Start training")
    start_time = time.time()
    loss_lst = []
    recons_loss_lst = []
    kld_loss_lst = []
    for epoch in tqdm(range(config.epochs)):
        if config.distributed:
            train_sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model, train_loader, optimizer, **config)
        lr_scheduler.step()

        checkponit_paths = [output_dir / f"checkpoint.pth"]
        if (epoch+1) % 10 == 0:
            checkponit_paths.append(output_dir / f"checkpoint{epoch+1}.pth")
            for checkpoint_path in checkponit_paths:
                save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "config": config,
                    },
                    checkpoint_path
                )
        loss_lst.append(train_stats["loss"])
        recons_loss_lst.append(train_stats["Reconstruction_loss"])
        kld_loss_lst.append(train_stats["KLD"])
    
        log_stats = {**{f"train_{k}" : v for k, v in train_stats.items()},
                     "epoch" : epoch,
                     "n_parameters" : n_parameters,}

        if is_main_process():
            with open(output_dir / "log.txt", "a") as f:
                f.write(json.dumps(log_stats) + "\n")


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # plots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    axes[0].plot(loss_lst)
    axes[0].set_title("Total loss")
    axes[1].plot(recons_loss_lst)
    axes[1].set_title("Reconstruction loss")
    axes[2].plot(kld_loss_lst)
    axes[2].set_title("KLD loss")
    plt.savefig(output_dir / "loss.png")
    plt.close()




if __name__ == "__main__":
    config = parse_args()
    main(config)

