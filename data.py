from pathlib import Path

from sklearn.preprocessing import LabelEncoder
import mne
from moabb.utils import set_log_level
from moabb.paradigms import MotorImagery
from skorch.dataset import Dataset
from moabb.datasets import BNCI2014_001, BNCI2014_004, Lee2019_MI, PhysionetMI, Schirrmeister2017

import logging

logger = logging.getLogger(__name__)

# Preprocessing steps in Xie2023:
# 1. Pick channels:
channels = ['C3', 'Cz', 'C4']

# 2. Re-reference using left mastoid
# ref = 'M1'
# -> canceled because not all datasets have this channel.
# -> instead, use average reference

# 3. Resampling
resample = 250

# 4. Epoching
tmin = -0.5
tmax = 4

# 5. Bandpass filtering
l_freq = 0
h_freq = 40


# 6. Channel-wise standardization


def _set_metadata(epochs, metadata, target=None, labels=None):
    if labels is not None:
        metadata['labels'] = labels
    if target is not None:
        metadata['target'] = target
    epochs.metadata = metadata


def _epochs_to_dataset(epochs):
    if epochs is None:
        return None
    X = epochs.get_data(units='uV').astype('float32')
    # Channel-wise standardization:
    # Slightly different from Xie2023, which used exponential moving channel-wise standardization,
    # but they initialized its parameters on the first 4 seconds of each 4.5s epoch.
    # So we are nearly equivalent.
    mu = X.mean(axis=2, keepdims=True)
    sigma = X.std(axis=2, keepdims=True)
    X = (X - mu) / sigma
    y = epochs.metadata['target'].values
    return Dataset(X, y)


def preprocess_data(dataset) -> mne.Epochs:  # tuple[mne.Epochs, mne.Epochs]:
    paradigm = MotorImagery(channels=channels, resample=resample, tmin=tmin, tmax=tmax)
    epochs, labels, metadata = paradigm.get_data(dataset, return_epochs=True)
    epochs, _ = mne.set_eeg_reference(epochs, ref_channels='average', copy=False)
    # third-order Butterworth bandpass filter of 0–40 Hz applied on epochs as in Xie2023
    epochs.filter(l_freq=l_freq, h_freq=h_freq, method="iir",
                  iir_params=dict(order=3, ftype='butter'))
    le = LabelEncoder()
    id_labels = le.fit_transform(labels)
    _set_metadata(epochs, metadata, target=id_labels, labels=labels)
    return epochs


def get_data(dataset, subjects: list[int] = None,
             overwrite_data: bool = False,
             data_dir=None, return_metadata=False) -> Dataset:  # tuple[Dataset, Dataset]:
    set_log_level('info')
    dataset_name = dataset.__class__.__name__
    if data_dir is None:
        data_dir = Path('~/') / 'data'
    preprocessed_data_dir = Path(data_dir).expanduser() / 'preprocessed' / 'xie2023'
    path = preprocessed_data_dir / f'{dataset_name}-epo.fif'
    if path.exists() and not overwrite_data:
        logger.info(f'Loading pre-processed data from {path}')
        epochs = mne.read_epochs(path, preload=False)
    else:
        logger.info('Pre-processing data')
        epochs = preprocess_data(dataset)
        logger.info(f'Saving pre-processed data to {path}')
        preprocessed_data_dir.mkdir(parents=True, exist_ok=True)
        assert preprocessed_data_dir.exists()
        epochs.save(path, overwrite=True)
    if subjects is not None:
        epochs = epochs[epochs.metadata.subject.isin(subjects)]
    Xy = _epochs_to_dataset(epochs)
    if return_metadata:
        return Xy, epochs.metadata
    return Xy


pretrain_datasets = [
    BNCI2014_001(),
    BNCI2014_004(),
    Lee2019_MI(),
    PhysionetMI(),
    Schirrmeister2017(),
]

finetune_datasets = [
    BNCI2014_001(),
    BNCI2014_004(),
    Schirrmeister2017(),
]


class TestData:

    def test_save_data(self, tmp_path):
        subjects = [1]
        dataset = pretrain_datasets[0]

        # preprocess and save data:
        out = get_data(
            dataset,
            subjects=subjects,
            overwrite_data=False,  # should still create the data
            data_dir=tmp_path,
        )

        # load data:
        out1 = get_data(
            dataset,
            subjects=subjects,
            overwrite_data=False,
            data_dir=tmp_path,
        )
        train_set = out
        print(train_set.X.std())
        for X, X1 in zip([out], [out1]):
            assert X.X.shape == X1.X.shape
            assert len(X.y) == len(X1.y)
            assert X.X.dtype == X1.X.dtype


if __name__ == '__main__':
    for dataset in pretrain_datasets:
        _ = get_data(dataset, overwrite_data=False)
