# classical-cv-corruptions

How much do hand-crafted image features suffer when images are degraded, and
does training on corrupted images help against corruptions the classifier never
saw in training?

The project compares four classical feature types (downsampled pixels, color
histograms, HOG, LBP) with linear classifiers on CIFAR-10, under six synthetic
corruptions at five severities each. It needs no GPU and no deep learning
framework.

## Questions the experiment is set up to answer

1. Which features lose the most accuracy under which corruption? Intensity
   statistics, gradient orientations and local binary patterns depend on the
   image in different ways, so the degradation curves should differ.
2. Does mixing corrupted images into the training set recover accuracy on the
   corruptions it contains?
3. Does it also help on a corruption that was held out of training? The
   `aug_no_blur` regime trains on every corruption except Gaussian blur, so
   comparing it with `clean` and `aug_all` on the blur rows answers this.

## Setup

**Features** (`ccr/features.py`). A spec such as `hog+color_hist` concatenates
base features.

| name | description | dims at 32x32 |
|---|---|---|
| `tiny` | RGB image area-downsampled to 16x16, scaled to [0, 1] | 768 |
| `color_hist` | 16-bin histogram per RGB channel | 48 |
| `hog` | grayscale, 9 orientations, 8x8 cells, 2x2 blocks, L2-Hys | 324 |
| `lbp` | uniform LBP (8 points, radius 1), 4x4 grid of cell histograms | 160 |

**Classifiers** (`ccr/models.py`). Both standardize the features first.
`logreg` is L2 logistic regression with C=0.1, `linear_svm` is `LinearSVC` with
C=0.01. The hyperparameters are fixed and were not tuned per feature.

**Corruptions** (`ccr/corruptions.py`). Images are uint8 RGB. Severity runs from
1 to 5.

| name | parameter by severity |
|---|---|
| `gaussian_noise` | noise std 0.02, 0.04, 0.06, 0.08, 0.10 (fraction of 255) |
| `gaussian_blur` | sigma 0.5, 0.75, 1.0, 1.5, 2.0 px |
| `brightness` | add 0.1, 0.2, 0.3, 0.4, 0.5 (fraction of 255), clipped |
| `contrast` | scale around the per-channel mean by 0.75, 0.5, 0.4, 0.3, 0.15 |
| `jpeg` | quality 60, 40, 25, 15, 8 |
| `occlusion` | black square at a random position, side 10%, 20%, 30%, 40%, 50% of the image |

The severity levels follow the idea of Hendrycks and Dietterich (2019) but are
not the CIFAR-10-C values, so numbers are not comparable with that benchmark.

**Training regimes** are defined in the config. `clean` trains on the original
images. An augmenting regime corrupts each training image with probability `p`,
picking one corruption from its list and one severity from its severity list at
random. `exclude` removes corruptions from the list, which is how held-out
corruptions are set up.

## Protocol

For each seed the code draws a class-stratified subset of the training set and
of the test set. Every regime and every model is evaluated on the same test
images, once clean and once per (corruption, severity) pair. Corruptions are
only applied at test time to the test subset, and the random streams used for
training augmentation and test corruption are separate.

`results.csv` has one row per run of seed, feature, model, regime, corruption
and severity, with the accuracy and a 95% Wilson interval for it. That interval
only reflects the finite test subset. Spread across seeds, which also covers
the choice of training subset, is reported as `corrupt_acc_std` in `summary.csv`.

`summary.csv` has, per feature, model and regime:

- `clean_acc`: accuracy on uncorrupted test images
- `corrupt_acc`: mean accuracy over all corruptions and severities in the run
- `acc_drop`: `clean_acc - corrupt_acc`
- `retention`: `corrupt_acc / clean_acc`

`by_corruption.csv` gives mean accuracy over severities for each corruption.

## Install

```
pip install -e .
```

Python 3.10 or newer. The dependencies are numpy, pandas, scikit-learn,
scikit-image, opencv-python-headless, matplotlib and pyyaml.

## Run

```
python -m ccr run --config configs/cifar10.yaml --out runs/cifar10
```

The first run downloads CIFAR-10 (about 170 MB) into `data/` and checks its
MD5. Use `--no-download` to fail instead if the archive is missing. Each seed
fits 15 models with the default config, so a full run takes a while on one
core. For a quick pass, lower `seeds`, `n_train` or the number of features.

To check that everything works without downloading anything:

```
python -m ccr run --config configs/digits_smoke.yaml --out runs/digits
```

This uses scikit-learn's 8x8 digits upsampled to 32x32. It is a smoke test and
says little about natural images.

Tables and figures can be rebuilt from an existing `results.csv`:

```
python -m ccr report runs/cifar10/results.csv
```

## Config

```yaml
dataset: cifar10          # cifar10 or digits
data_root: data
n_train: 10000
n_test: 2000
seeds: [0, 1, 2]
features: [tiny, color_hist, hog, lbp, hog+color_hist]
models: [logreg]          # logreg, linear_svm
corruptions: all          # or a list of names
severities: [1, 2, 3, 4, 5]
regimes:
  clean: {}
  aug_all: {corruptions: all, severities: [1, 2, 3], p: 0.5}
  aug_no_blur: {corruptions: all, exclude: [gaussian_blur], severities: [1, 2, 3], p: 0.5}
```

Unknown keys and invalid values raise an error before anything runs.

## Output

```
runs/cifar10/
  config.yaml            resolved config used for the run
  results.csv            one row per seed x feature x model x regime x condition
  summary.csv
  by_corruption.csv
  figures/
    curves_<model>_<regime>.png   accuracy against severity, one panel per corruption
    regimes_<model>.png           mean corrupted accuracy by feature and training regime
```

No CIFAR-10 results are included in this repository. Run the config to
produce them.

## Layout

```
ccr/
  corruptions.py   corruption functions and training augmentation
  features.py      feature extractors
  models.py        classifiers
  data.py          CIFAR-10 and digits loading, stratified subsets
  config.py        config parsing and validation
  experiment.py    training and evaluation loop
  metrics.py       Wilson interval, summary tables
  plots.py         figures
  report.py        writes tables and figures
  cli.py
configs/
tests/
```

## Tests

```
python -m unittest discover -s tests -t .
```

The tests cover each corruption (shape, determinism, severity ordering, exact
effects such as the occluded area), the feature dimensions and normalisation,
the CIFAR-10 loader against a synthetic archive with the same layout, config
validation, the Wilson interval and summary arithmetic, and a short end-to-end
run on the digits data.

## Limitations

- Only classical features and linear models. The conclusions say nothing about
  convolutional networks.
- Hyperparameters are fixed. A different C could change the ranking of features.
- The corruptions are synthetic and applied to 32x32 images, so JPEG and blur
  behave differently than they would on full-size photographs.
- The default training set is a 10,000 image subset of CIFAR-10, which lowers
  absolute accuracy compared with using all 50,000.
- Three seeds give only a rough idea of the spread.

## References

- Hendrycks and Dietterich. Benchmarking neural network robustness to common
  corruptions and perturbations. ICLR 2019.
- Dalal and Triggs. Histograms of oriented gradients for human detection. CVPR 2005.
- Ojala, Pietikainen and Maenpaa. Multiresolution gray-scale and rotation
  invariant texture classification with local binary patterns. TPAMI 2002.
- Krizhevsky. Learning multiple layers of features from tiny images. 2009.
- Torralba, Fergus and Freeman. 80 million tiny images. TPAMI 2008.
