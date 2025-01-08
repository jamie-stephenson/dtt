from dtt.utils.config import Config

from bpekit import encode_dataset, train_tokenizer
from bpekit.utils import get_rank_and_world_size
import yaml
import argparse


def tokenize_data(cfg: Config):
    paths = cfg.get_paths()

    if paths.tokens.exists() and paths.tokens.is_dir():
        raise FileExistsError(
            f"A directory named `{paths.tokens}` already exists. "
            f"Have you already used `{paths.tokenizer}` to encode `{paths.dataset}`?"
        )

    with open(paths.dataset_config, "r") as file:
        dataset_cfg = yaml.safe_load(file)

    rank, world_size = get_rank_and_world_size()

    train_tokenizer(
        path=paths.dataset,
        vocab_size=cfg.vocab_size,
        merges_path=paths.tokenizer,
        rank=rank,
        world_size=world_size,
        **dataset_cfg,
    )

    encode_dataset(
        path=paths.dataset,
        merges_path=paths.tokenizer,
        tokens_path=paths.tokens,
        rank=0,
        world_size=1,
        **dataset_cfg,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config", help="Path to config file.")

    args = parser.parse_args()

    cfg = Config.build_from(args.config)

    tokenize_data(cfg)
