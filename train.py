from jet import get_model
from src.data_utils import get_dataloader
from src.dist_utils import setup, cleanup, is_torchrun
from src.file_utils import PathFetcher, args_from_config_file
from src.training_loop import train
import torch
import torch.distributed as dist
import argparse
import os
import yaml
import wandb
import importlib

def main(args):

    paths = PathFetcher(args, is_torchrun())

    if not args.no_wandb and args.rank == 0:
        wandb.init(project='jet',name=paths.wandb,config=args)
        wandb.define_metric("Effective Batch Number")        

    model = get_model(args)
    optimizer = get_optimizer(args.optimizer, model, args)
    lr_scheduler = get_lr_scheduler(args.lr_schedule, optimizer, args)
    train_dataloader = get_dataloader(paths.encoded_corpus, args, 'train')
    eval_dataloader = get_dataloader(paths.encoded_corpus, args, 'eval')
    
    model = train(model,train_dataloader,eval_dataloader,optimizer,lr_scheduler,args)

    if args.rank == 0:
        os.makedirs(os.path.dirname(paths.model))
        torch.save(model.state_dict(),paths.model)
        with open(paths.config,'w') as file:
            yaml.dump(vars(args), file)

def get_optimizer(name, model, args):
    
    optimizer_module = importlib.import_module("optimizers." + name, package=".")
    optimizer = optimizer_module.get_optimizer(model, args)

    return optimizer

def get_lr_scheduler(name, optimizer, args):
    
    lr_scheduler_module = importlib.import_module("lr_schedulers." + name, package=".")
    lr_scheduler = lr_scheduler_module.get_lr_scheduler(optimizer, args)

    return lr_scheduler

def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        default="jet",
        type=str,
        help="Used to name run folder."
    )

    parser.add_argument(
        "--corpus",
        type=str,
        help="The text dataset to train jet with."
    )

    parser.add_argument(
        "--encoded_format",
        choices=['mmap','shards'],
        type=str,
        help="The format of the encoded corpus file."
    )

    parser.add_argument(
        "--tokenizer_corpus",
        default=None,
        type=str,
        help="Name of corpus used to train tokenizer."
    )

    parser.add_argument(
        "--vocab_size",
        type=int,
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="The number of iterations to train for."
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        help="The optimizer to use for training."
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0,
        help="The momentum to be used by the optimizer during training."
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0,
        help="The weight decay to be used by the optimizer during training"#
    )

    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="exponential",
        help="The learning rate schedule used for training."
    )

    parser.add_argument(
        "--lr_gamma",
        type=str,
        default=0.1,
        help="Multiplicative factor of learning rate decay."
    )

    parser.add_argument(
        "--lr_warmup_end",
        type=float,
        default=0.4,
        help="The number of epochs to warm up the learning rate for"
    )

    parser.add_argument(
        "--lr_max",
        type=float,
        default=0.2,
        help="The maximum learning rate to use for training"
    )

    parser.add_argument(
        "--dropout",
        default=0.2,
        type = float,
        help="Proportion of elements to zero in layer where dropout is used."
    )

    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Batch size used for training."
    )
    
    parser.add_argument(
        "--grad_accumulation_steps",
        type=int,
        default=0,
        help="The number of batches to accumulate into one effective batch. Used with distributed training."
    )

    parser.add_argument(
        "--seq_len",
        default=16,
        type=int,
        help="Max sequence length used for training."
    )

    parser.add_argument(
        "--embed_dim",
        default=16,
        type=int,
        help="Dimension tokens are embedded into."
    )

    parser.add_argument(
        "--num_heads",
        default=16,
        type=int,
        help="Number of attention heads in each attention layer."
    )

    parser.add_argument(
        "--mask_type",
        default="causal",
        type=str,
        help="How the attention is masked."
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device on which experiments are to be ran."
    )

    parser.add_argument(
        "--rank",
        default=0,
        help="Global rank of current process."
    )

    parser.add_argument(
        "--world_size",
        default=1,
        help="Total number of processes."
    )

    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        help="Path to optional config file."
    )

    parser.add_argument(
        "--no-wandb",
        "--no_wandb",
        action="store_true",
        help="If set, wandb logging is disabled."
    )

    parser.add_argument(
        "--eff_batch_per_log",
        default=50,
        help="Number of effective batches per log."
    )

    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()

    if args.config_file:
        args = args_from_config_file(args)

    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    setup(backend) 
    args.device += f":{os.getenv('LOCAL_RANK','0')}"
    args.rank, args.world_size = dist.get_rank(), dist.get_world_size() 

    main(args)

    cleanup()
