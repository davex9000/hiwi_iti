# Autoencoder-based Channel Coding over AWGN

Implements the end-to-end learning approach from O'Shea & Hoydis (2017) —
an autoencoder that jointly learns an encoder (transmitter) and decoder (receiver)
to minimise symbol error rate over an AWGN channel.

## Concept

| Block | Role |
|---|---|
| Encoder (Dense layers) | Maps one-hot symbol → normalised n-dimensional signal |
| BatchNormalization | Enforces average power constraint |
| GaussianNoise | Simulates AWGN channel at a fixed training Eb/N0 |
| Decoder (Dense layers) | Recovers the original symbol via softmax classification |

The system is parameterised by `(k, n)`: `k` information bits, `n` channel uses,
giving code rate `R = k/n`. The standard benchmark is `(3, 7)`, comparable to Hamming(7,4).

## Notebooks

| Notebook | Encoder/Decoder | Activation | Notes |
|---|---|---|---|
| `autoencoder_.ipynb` | Standard Dense | ReLU / Softmax | Baseline implementation |
| `autoencoder_custom_binary_dense_lib_.ipynb` | Binary-Dense | ReLU / Linear | Quantised weights via `binarization_utils` |
| `autoencoder_custom_lib_sinh_activation.ipynb` | Binary-Dense | Sign (`tf.sign`) / Softmax | Hard binarisation at every layer |

Each notebook outputs a BER vs Eb/N0 curve compared against uncoded BPSK.

## Requirements

```
tensorflow >= 2.4
numpy
matplotlib
scipy
```

## Setup & Running

```bash
pip install tensorflow scipy matplotlib jupyter
```

Run any notebook:

```bash
jupyter notebook
```

Or execute a notebook non-interactively:

```bash
jupyter nbconvert --to notebook --execute autoencoder_.ipynb --inplace
```

### `binarization_utils.py`

Included in this repo. Provides a `binary_dense` layer with two modes:

- `BINARY=False` (default) — standard Dense layer; all notebooks use this mode
- `BINARY=True` — weights are binarised to {-1, +1} using a straight-through
  estimator so gradients still flow during training

## Usage

Open any notebook and run all cells. Key parameters at the top of `autoenco()`:

- `k` — information bits per symbol (default 3)
- `n` — channel uses / encoded dimension (default 7)
- `SNRs` — Eb/N0 sweep range in dB

Trained weights are saved as `ae_{k}_{n}.weights.h5` and reloaded automatically
for BER evaluation.

## Reference

T. O'Shea and J. Hoydis, "An Introduction to Deep Learning for the Physical Layer,"
*IEEE Transactions on Cognitive Communications and Networking*, 2017.
