import argparse
import ast
import copy
import logging
import os
import time

import jinja2
import torch
import yaml
from jinja2 import meta
from torch import distributed as dist

from ultra import datasets


logger = logging.getLogger(__file__)


class AttrDict(dict):
    """Dictionary with attribute access, recursively applied to YAML config."""

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error


def _to_attr_dict(value):
    if isinstance(value, dict):
        return AttrDict({key: _to_attr_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_attr_dict(item) for item in value]
    return value


def detect_variables(cfg_file):
    with open(cfg_file, "r", encoding="utf-8") as fin:
        raw = fin.read()
    return meta.find_undeclared_variables(jinja2.Environment().parse(raw))


def load_config(cfg_file, context=None):
    with open(cfg_file, "r", encoding="utf-8") as fin:
        raw = fin.read()
    rendered = jinja2.Template(raw).render(context or {})
    return _to_attr_dict(yaml.safe_load(rendered))


def literal_eval(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-s", "--seed", type=int, default=1024)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--sym_ratio_threshold", type=float, default=0.5)
    parser.add_argument("--sym_score_threshold", type=float, default=0.4)
    parser.add_argument("--cooccur_ratio_threshold", type=float, default=0.1)
    parser.add_argument("--cooccur_score_threshold", type=float, default=0.4)
    parser.add_argument("--min_cooccur_support", type=int, default=5)
    parser.add_argument("--max_iterations", type=int, default=2)
    parser.add_argument("--patience", type=int, default=3)
    args, unparsed = parser.parse_known_args()

    dynamic_parser = argparse.ArgumentParser()
    for variable in detect_variables(args.config):
        dynamic_parser.add_argument(f"--{variable}", required=True)
    dynamic_args = dynamic_parser.parse_known_args(unparsed)[0]
    context = {
        key: literal_eval(value)
        for key, value in dynamic_args._get_kwargs()
    }
    return args, context


def get_root_logger(file=True):
    log_format = "%(asctime)-10s %(message)s"
    date_format = "%H:%M:%S"
    logging.basicConfig(format=log_format, datefmt=date_format)
    root_logger = logging.getLogger("")
    root_logger.setLevel(logging.INFO)
    if file:
        handler = logging.FileHandler("log.txt")
        handler.setFormatter(logging.Formatter(log_format, date_format))
        root_logger.addHandler(handler)
    return root_logger


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def synchronize():
    if get_world_size() > 1:
        dist.barrier()


def get_device(cfg):
    if cfg.train.gpus:
        return torch.device(cfg.train.gpus[get_rank()])
    return torch.device("cpu")


def create_working_directory(cfg):
    world_size = get_world_size()
    if cfg.train.gpus is not None and len(cfg.train.gpus) != world_size:
        raise ValueError(
            f"world size is {world_size}, but config declares "
            f"{len(cfg.train.gpus)} GPU(s)"
        )
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://")

    working_dir = os.path.abspath(
        os.path.join(
            os.path.expanduser(cfg.output_dir),
            cfg.model["class"],
            cfg.dataset["class"],
            time.strftime("%Y-%m-%d-%H-%M-%S"),
        )
    )
    marker = os.path.abspath("working_dir.tmp")
    if get_rank() == 0:
        os.makedirs(working_dir, exist_ok=False)
        with open(marker, "w", encoding="utf-8") as fout:
            fout.write(working_dir)
    synchronize()
    if get_rank() != 0:
        with open(marker, "r", encoding="utf-8") as fin:
            working_dir = fin.read()
    synchronize()
    if get_rank() == 0:
        os.remove(marker)
    os.chdir(working_dir)
    return working_dir


def build_dataset(cfg):
    data_config = copy.deepcopy(cfg.dataset)
    class_name = data_config.pop("class")
    dataset = getattr(datasets, class_name)(**data_config)
    if get_rank() == 0:
        train_data, valid_data, test_data = dataset._data
        logger.warning("%s dataset", class_name)
        logger.warning(
            "#train: %d, #valid: %d, #test: %d",
            sum(item.target_edge_index.shape[1] for item in train_data),
            sum(item.target_edge_index.shape[1] for item in valid_data),
            sum(item.target_edge_index.shape[1] for item in test_data),
        )
    return dataset
