from copy import deepcopy
from pathlib import Path
from tempfile import mkdtemp
from shutil import rmtree
from math import prod

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from skorch.callbacks import Freezer, Unfreezer, Callback, EarlyStopping, Checkpoint, EpochScoring
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from saving import save_net, load_net
from data import pretrain_datasets, finetune_datasets, get_data
from braindecode import EEGClassifier

# Arbitrary choice because not documented in Xie2023:
# we include the (trainable) batch-norm in the group following.
EEGNetv4_layer_groups = [
    ['conv_temporal'],
    ['bnorm_temporal', 'conv_spatial'],
    ['bnorm_1', 'conv_separable_depth', 'conv_separable_point', 'bnorm_2'],
    ['final_layer'],
]


def get_finetuning_callbacks(finetuned_groups, layer_groups=EEGNetv4_layer_groups):
    return [
        ('freeze', Freezer(patterns='*')),
        ('unfreeze',
         Unfreezer(patterns=[f'{l}.*' for g in finetuned_groups for l in layer_groups[g]])),
    ]


EEGNetv4_finetuning_callbacks = {
    'Scheme 1': get_finetuning_callbacks([3]),
    'Scheme 2': get_finetuning_callbacks([1, 2, 3]),
    'Scheme 3': get_finetuning_callbacks([2, 3]),
    'Scheme 4': get_finetuning_callbacks([0, 1, 2, 3]),
}

csv_columns = [
    'pretrain_dur',
    'finetune_phase1_dur',
    'finetune_phase2_dur',
    'pretrain_n_epochs',
    'finetune_phase1_n_epochs',
    'finetune_phase2_n_epochs',
    'pretrain_train_size',
    'pretrain_valid_size',
    'finetune_phase1_train_size',
    'finetune_phase1_valid_size',
    'finetune_phase2_train_size',
    'finetune_phase2_valid_size',
    'test_size',
    'seed',
    'pretrain_dataset',
    'finetune_dataset',
    'finetune_subject',
    'finetune_scheme',
    'finetune_fold',
    'pretrain_train_acc',
    'pretrain_valid_acc',
    'finetune_phase1_train_acc',
    'finetune_phase1_valid_acc',
    'finetune_phase2_valid_acc',
    'test_acc',
]


class ThresholdStopping(Callback):
    """Stop training when a monitored metric has reached or is better than a certain threshold.
    """

    def __init__(self, threshold, lower_is_better=True, monitor='valid_loss'):
        self.threshold = threshold
        self.lower_is_better = lower_is_better
        self.monitor = monitor

    def _passed_threshold(self, score):
        if self.lower_is_better:
            return score <= self.threshold
        return score >= self.threshold

    def on_epoch_end(self, net, **kwargs):
        current_score = net.history[-1, self.monitor]
        if self._passed_threshold(current_score):
            if net.verbose:
                print(f"Stopping since {self.monitor} has reached {current_score} "
                      f"and is better than the threshold {self.threshold}.")
            raise KeyboardInterrupt


def set_dropout(module, dropout):
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            m.p = dropout


def finetune(pretrained_net, X, y, X_test, y_test, callbacks, csv_path='results.csv',
             device='cpu', seed: int = None, fold_info: dict = None, debug: bool = False):
    callbacks = [(
        'train_acc',
        EpochScoring(
            'accuracy',
            name='train_acc',
            lower_is_better=False,
            on_train=True, )
    ), ] + list(callbacks)
    out = dict(**fold_info) if fold_info is not None else dict()
    # Split train/valid:
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, stratify=y,
                                                          random_state=seed, test_size=0.2)
    valid_ds = Dataset(X_valid, y_valid)
    finetuning_kwargs = dict(
        max_epochs=5000,
        optimizer=Adam,
        optimizer__lr=0.0006,
        batch_size=60,
        train_split=predefined_split(valid_ds),
        warm_start=True,
        device=device,
    )
    # Log pre-training metrics:
    out['pretrain_valid_acc'] = pretrained_net.history[-1, 'valid_acc']
    out['pretrain_train_acc'] = pretrained_net.history[-1, 'train_acc']
    out['pretrain_n_epochs'] = pretrained_net.history[-1, 'epoch']
    out['pretrain_dur'] = sum(pretrained_net.history[:, 'dur'])
    out['pretrain_train_size'] = sum(pretrained_net.history[-1, 'batches', :, 'train_batch_size'])
    out['pretrain_valid_size'] = sum(pretrained_net.history[-1, 'batches', :, 'valid_batch_size'])
    # Update module:
    pretrained_module = deepcopy(pretrained_net.module_)
    # Set dropout:
    set_dropout(pretrained_module, 0.5)
    # Update clf layer:
    n_classes = len(set(y))
    emb_size = prod(pretrained_module.final_layer.conv_classifier.weight.shape[1:])
    if seed is None:
        torch.seed()
    else:
        torch.manual_seed(seed)
    final_layer = nn.Sequential(
        nn.Flatten(),
        nn.Linear(emb_size, n_classes),
    )
    pretrained_module.final_layer = final_layer

    # Set early-stopping and finetuning callbacks:
    checkpoint_dirname = mkdtemp()
    # Schirrmeister2017:
    phase1_callbacks = (
            callbacks +
            [
                ('early_stopping', EarlyStopping(
                    patience=100 if not debug else 1,
                    # patience not given in Xie2023 nor Schirrmeister2017,
                    # so used the same as for pre-training
                    monitor='valid_acc',
                    lower_is_better=False,
                    load_best=False,
                )),
                ('checkpoint', Checkpoint(
                    monitor='valid_acc_best',
                    load_best=True,
                    dirname=checkpoint_dirname,
                )),
            ]
    )
    # Described in Xie2023 despite them also claiming they do the same as in Schirrmeister2017:
    # phase1_callbacks = (
    #         [
    #             ('early_stopping', EarlyStopping(
    #                 patience=100,
    #                 # patience not given in Xie2023 nor Schirrmeister2017,
    #                 # so used the same as for pre-training
    #                 monitor='train_loss',
    #                 lower_is_better=True,
    #                 load_best=False
    #             )),
    #             ('checkpoint', Checkpoint(
    #                 monitor='train_loss_best',
    #                 lower_is_better=True,
    #                 load_best=True,
    #                 dirname = checkpoint_dirname,
    #             )),
    #         ]
    #         + callbacks
    # )
    # Train Phase 1:
    net = EEGClassifier(
        pretrained_module,
        callbacks=phase1_callbacks,
        **finetuning_kwargs,
    )
    net.partial_fit(X_train, y_train)
    # Log Phase 1 metrics:
    out['finetune_phase1_valid_acc'] = net.history[-1, 'valid_acc']
    out['finetune_phase1_train_acc'] = net.history[-1, 'train_acc']
    out['finetune_phase1_n_epochs'] = net.history[-1, 'epoch']
    out['finetune_phase1_dur'] = sum(net.history[:, 'dur'])
    out['finetune_phase1_train_size'] = sum(net.history[-1, 'batches', :, 'train_batch_size'])
    out['finetune_phase1_valid_size'] = sum(net.history[-1, 'batches', :, 'valid_batch_size'])
    last_history_len = len(net.history)
    # Set threshold stopping and finetuning callbacks:
    train_loss_end_phase_1 = net.history[-1, 'train_loss']
    phase2_callbacks = callbacks + [
        ('threshold_stopping', ThresholdStopping(
            threshold=train_loss_end_phase_1,
            monitor='valid_loss',
            lower_is_better=True,
        )),
    ]
    # Train phase 2:
    net.set_params(
        callbacks=phase2_callbacks,
    )
    net.initialize_callbacks()
    # Continue on the whole dataset (train+valid):
    net.partial_fit(X, y)
    # Log Phase 2 metrics:
    out['finetune_phase2_valid_acc'] = net.history[-1, 'valid_acc']
    out['finetune_phase2_n_epochs'] = net.history[-1, 'epoch']
    out['finetune_phase2_dur'] = sum(net.history[last_history_len:, 'dur'])
    out['finetune_phase2_train_size'] = sum(net.history[-1, 'batches', :, 'train_batch_size'])
    out['finetune_phase2_valid_size'] = sum(net.history[-1, 'batches', :, 'valid_batch_size'])
    # Remove temp checkpoint:
    rmtree(checkpoint_dirname)
    # Testing
    score_test = net.score(X_test, y_test)
    out['test_acc'] = score_test
    out['test_size'] = len(y_test)
    # Save results to CSV:
    assert set(out.keys()) == set(csv_columns)
    df = pd.DataFrame([[out[k] for k in csv_columns]], columns=csv_columns)
    with pd.option_context("display.max_columns", len(csv_columns), "display.width", None):
        print(df)
    df.to_csv(csv_path, index=False, mode='a', header=False)
    return df


def finetune_main(pretrain_dataset, finetune_dataset, finetune_subject, n_folds=5,
                  device='cpu', debug_datadir=None, overwrite=False):
    # Initialize results' CSV:
    base_dir = Path('./' if debug_datadir is None else debug_datadir)
    csv_path = base_dir / 'results' / f'pre-{pretrain_dataset.__class__.__name__}_fin-{finetune_dataset.__class__.__name__}_sub-{finetune_subject}.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=csv_columns).to_csv(csv_path, index=False, mode='w' if overwrite else 'x',
                                             header=True)
    print(f'Writing results to {csv_path}')
    # Load pre-trained model:
    pretrained_net = load_net(pretrain_dataset, data_dir=debug_datadir)
    # Load data:
    Xy, metadata = get_data(finetune_dataset, subjects=[finetune_subject],
                            data_dir=debug_datadir, return_metadata=True)
    # Split train/test:
    if any('train' in x for x in metadata.session):
        split_key = 'session'
    elif any('run' in x for x in metadata.session):
        split_key = 'run'
    else:
        raise ValueError()
    train_mask = metadata[split_key].str.contains('train')
    test_mask = metadata[split_key].str.contains('test')
    assert train_mask.sum() + test_mask.sum() == len(Xy)
    X_train, y_train = Xy.X[train_mask], Xy.y[train_mask]
    X_test, y_test = Xy.X[test_mask], Xy.y[test_mask]
    # Finetune:
    results = []
    for finetune_scheme, finetune_callback in EEGNetv4_finetuning_callbacks.items():
        for fold in range(n_folds):
            fold_info = dict(
                pretrain_dataset=pretrain_dataset.__class__.__name__,
                finetune_dataset=finetune_dataset.__class__.__name__,
                finetune_subject=finetune_subject,
                finetune_scheme=finetune_scheme,
                finetune_fold=fold,
                seed=fold,
            )
            results.append(finetune(
                pretrained_net,
                X=X_train, y=y_train,
                X_test=X_test, y_test=y_test,
                callbacks=finetune_callback,
                csv_path=csv_path,
                device=device,
                seed=fold,
                fold_info=fold_info,
                debug=debug_datadir is not None,
            ))
    return pd.concat(results)


class TestFinetuning:
    def test_finetuning(self, tmp_path):
        from pretraining import pretrain
        pretrain_dataset = pretrain_datasets[0]
        finetune_dataset = finetune_datasets[0]
        _ = pretrain(pretrain_dataset, device='cpu', debug_datadir=tmp_path)
        results = finetune_main(
            pretrain_dataset=pretrain_dataset,
            finetune_dataset=finetune_dataset,
            finetune_subject=finetune_dataset.subject_list[0],
            device='cpu',
            debug_datadir=tmp_path,
            overwrite=False,
            n_folds=1,
        )
        assert len(results) == 4
        assert set(results.columns) == set(csv_columns)
        results_csv = pd.read_csv(
            tmp_path / 'results' / f'pre-{pretrain_dataset.__class__.__name__}_fin-{finetune_dataset.__class__.__name__}_sub-{finetune_dataset.subject_list[0]}.csv')
        assert len(results_csv) == 20
        assert set(results_csv.columns) == set(csv_columns)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--overwrite', action='store_true')
    # parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    results = []
    for pretrain_dataset in pretrain_datasets:
        for finetune_dataset in finetune_datasets:
            for finetune_subject in finetune_dataset.subject_list:
                results.append(finetune_main(
                    pretrain_dataset=pretrain_dataset,
                    finetune_dataset=finetune_dataset,
                    finetune_subject=finetune_subject,
                    device=args.device,
                    debug_datadir=None,
                    overwrite=args.overwrite,
                ))
    results = pd.concat(results)
    print(results)
