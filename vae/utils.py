import os
import torch
import torch.distributed as dist


class DictX(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return '<DictX ' + dict.__repr__(self) + '>'

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True    

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(config):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        config.rank = int(os.environ["RANK"])
        config.world_size = int(os.environ['WORLD_SIZE'])
        config.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        config.rank = int(os.environ['SLURM_PROCID'])
        config.gpu = config.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        config.distributed = False
        return

    config.distributed = True

    torch.cuda.set_device(config.gpu)
    config.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        config.rank, config.dist_url), flush=True)
    torch.distributed.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                         world_size=config.world_size, rank=config.rank)
    torch.distributed.barrier()
    setup_for_distributed(config.rank == 0)

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def collate_fn(batch):
    real_batch = []
    for tensor, label in batch:
        real_batch.append(tensor)
    return torch.stack(real_batch)

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict