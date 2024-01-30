# Cross-dataset transfer

In this repo, we reproduce the results of Xie et al. (2023) [1],
using other datasets for pre-training selected according to the results of Guetschel et al. (2023) [2].

## Requirements

The necessary packages can be installed using [poetry](https://python-poetry.org/):

```bash
poetry install
```

## Usage

The main entry points of this repository are:

1. `data.py` for downloading and preprocessing the data.
2. `pretraining.py` for pre-training the models on all the subjects from the datasets.
3. `finetuning.py` for fine-tuning the models on the subjects individually.

## References

[1] Y. Xie et al., “Cross-dataset transfer learning for motor imagery signal classification via multi-task learning and
pre-training,” J. Neural Eng., vol. 20, no. 5, p. 056037, Oct. 2023, doi: 10.1088/1741-2552/acfe9c.

[2] P. Guetschel and M. Tangermann, “Transfer Learning between Motor Imagery Datasets using Deep Learning - Validation
of Framework and Comparison of Datasets.” Delft, Netherlands, Nov. 08, 2023. doi: 10.48550/arXiv.2311.16109.
