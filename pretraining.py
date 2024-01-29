from shutil import rmtree
import logging

from torch.optim import Adam
from skorch.callbacks import EarlyStopping, Checkpoint, EpochScoring
from skorch.dataset import ValidSplit
from braindecode.models import EEGNetv4
from braindecode import EEGClassifier

from data import get_data, pretrain_datasets
from saving import save_net, save_net_kwargs, load_net, _get_net_paths

logger = logging.getLogger(__name__)

# 1. Model:
module_cls = EEGNetv4
# default params lead to EEGNet-8,2, similar to Xie2023
# kernels length left unchanged despite 250Hz as in Xie2023
module_kwargs = dict(
    drop_prob=0.3,
)

# 2. Training:
optimizer = Adam
lr = 0.0006
batch_size = 60
validation_size = 0.1

# 3. Training end
max_epochs = 5000
es_patience = 100


def pretrain(dataset, device='cpu', debug_datadir=None):
    # subjects = None
    if debug_datadir is not None:
        logging.basicConfig(level=logging.DEBUG)
    #     max_epochs = 1
    #     subjects = [1]

    module_skorch_kwargs = {'module__' + k: v for k, v in
                            module_kwargs.items()}
    train_set = get_data(dataset=dataset, overwrite_data=False,
                         subjects=None if debug_datadir is None else [1],
                         data_dir=debug_datadir)
    paths = _get_net_paths(dataset, data_dir=debug_datadir)
    callbacks = [
        ('train_acc', EpochScoring(
            'accuracy',
            name='train_acc',
            lower_is_better=False,
            on_train=True, )),
        ("early_stopping", EarlyStopping(
            monitor='valid_loss', lower_is_better=True,
            patience=es_patience if debug_datadir is None else 1,
            load_best=False)),
        ("checkpoint", Checkpoint(monitor='valid_loss_best', load_best=True,
                                  dirname=paths.base,
                                  **{k: p.name for k, p in paths.skorch.items()})),
    ]

    trainer = EEGClassifier(
        module_cls,
        **module_skorch_kwargs,
        optimizer=optimizer,
        optimizer__lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        callbacks=callbacks,
        train_split=ValidSplit(cv=validation_size, stratified=True, random_state=12),
        iterator_train__shuffle=True,
        iterator_train__drop_last=True,
        iterator_train__num_workers=0,
        iterator_valid__num_workers=0,
        iterator_train__pin_memory=False,
        iterator_valid__pin_memory=False,
        device=device,
    )
    trainer.fit(train_set.X, y=train_set.y)

    # signal-specific parameters only added at fit:
    all_module_kwargs = trainer.get_params_for('module')
    save_net_kwargs(module_cls=module_cls, module_kwargs=all_module_kwargs, dataset=dataset,
                    data_dir=debug_datadir)

    model_paths = save_net(trainer, dataset=dataset, data_dir=debug_datadir)

    return model_paths


class TestPretrain:
    def test_save_models(self, tmp_path):
        dataset = pretrain_datasets[0]
        net_paths = pretrain(dataset, device='cpu', debug_datadir=tmp_path)
        assert net_paths.base.exists()
        net = load_net(dataset, data_dir=tmp_path)
        assert net is not None


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    # parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    net_paths = []
    for dataset in pretrain_datasets:
        p = pretrain(dataset, device=args.device)
        net_paths.append(p)
    print(net_paths)
