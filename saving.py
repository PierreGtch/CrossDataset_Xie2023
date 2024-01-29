from dataclasses import dataclass
from pathlib import Path
import pickle

import torch
from torch import nn
from skorch import NeuralNet
from braindecode import EEGClassifier

import logging

logger = logging.getLogger(__name__)


######################
# Paths

@dataclass
class NetPaths:
    base: Path
    skorch: dict[str, Path]
    torch: Path
    kwargs: Path


def _get_net_paths(dataset, data_dir=None) -> NetPaths:
    if data_dir is None:
        data_dir = Path('~/') / 'data'
    base_dir = Path(
        data_dir).expanduser() / 'pretrained_models' / 'xie2023' / dataset.__class__.__name__
    skorch_paths = dict(
        f_params=base_dir / f'model-params.pkl',
        f_optimizer=base_dir / f'opt.pkl',
        f_history=base_dir / f'history.json',
    )
    torch_path = base_dir / f'model.pkl'
    kwargs_path = base_dir / f'kwargs.pkl'
    return NetPaths(
        base=base_dir,
        skorch=skorch_paths,
        torch=torch_path,
        kwargs=kwargs_path,
    )


######################
# Kwargs

def save_net_kwargs(dataset, module_cls: nn.Module, module_kwargs: dict, data_dir=None):
    paths = _get_net_paths(dataset, data_dir=data_dir)
    paths.base.mkdir(parents=True, exist_ok=True)
    out = dict(
        module_cls=module_cls,
        module_kwargs=module_kwargs,
    )
    with open(paths.kwargs, 'wb') as f:
        pickle.dump(out, f)
    print(f'saved kwargs {out} to {paths.kwargs}')
    return paths.kwargs


def load_net_kwargs(dataset, data_dir=None):
    paths = _get_net_paths(dataset, data_dir=data_dir)
    with open(paths.kwargs, 'rb') as f:
        out = pickle.load(f)
    return out


######################
# Net

def save_net(net: NeuralNet, dataset, data_dir=None):
    paths = _get_net_paths(dataset, data_dir=data_dir)
    paths.base.mkdir(parents=True, exist_ok=True)
    # with open(paths.net, 'wb') as f:
    #     pickle.dump(net, f)
    torch.save(net.module_, paths.torch)  # needed to be able to load th model on another device
    # net.save_params(**paths.skorch) # already done by checkpoint
    return paths


def load_net(dataset, data_dir=None):
    paths = _get_net_paths(dataset, data_dir=data_dir)
    net_kwargs = load_net_kwargs(dataset, data_dir=data_dir)
    module_cls = net_kwargs['module_cls']
    module_kwargs = {f'module__{k}': v for k, v in net_kwargs['module_kwargs'].items()}
    net = EEGClassifier(module_cls, **module_kwargs).initialize()
    net.load_params(**paths.skorch)
    return net
