[![DOI](https://zenodo.org/badge/670384512.svg)](https://zenodo.org/badge/latestdoi/670384512)

# emlearn-micropython

[Micropython](https://micropython.org) integration for the [emlearn](https://emlearn.org) Machine Learning library for microcontrollers.

The goal is to enable applications to run ML inference on the microcontroller,
without having to touch any C code.

## Status
**Minimally useful**

- Has been tested on `armv6m` (RP2040) and `x64` (Unix port) 
- Pre-built modules are available for the most common architectures/devices

## Features

- Classification with [RandomForest](https://en.wikipedia.org/wiki/Random_forest)/DecisionTree models
- Classification and on-device learning with [K-Nearest Neighbors (KNN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- Installable as a MicroPython native module. No rebuild/flashing needed
- Models can be loaded at runtime from a .CSV file in disk/flash

## Prerequisites

Minimally you will need

- Python 3.10+ on host
- MicroPython 1.20+ running onto your device

#### Download repository

Download the repository with examples etc
```
git clone https://github.com/emlearn/emlearn-micropython
```

## Installing from a release

#### Find architecture

Identify which CPU architecture your device uses.
You need to specify `ARCH` to install the correct module version.

| ARCH          | Description                       | Examples              |
|---------------|-----------------------------------|---------------------- |
| x64           | x86 64 bit                        | PC                    |
| x86           | x86 32 bit                        |                       |
| armv6m        | ARM Thumb (1)                     | Cortex-M0             |
| armv7m        | ARM Thumb 2                       | Cortex-M3             |
| armv7emsp     | ARM Thumb 2, single float         | Cortex-M4F, Cortex-M7 |
| armv7emdp     | ARM Thumb 2, double floats        | Cortex-M7             |
| xtensa        | non-windowed                      | ESP8266               |
| xtensawin     | windowed with window size 8       | ESP32                 |

#### Download release files

Download from [releases](https://github.com/emlearn/emlearn-micropython/releases).

#### Install on device

Copy the .mpy file for the correct `ARCH` to your device.
```
mpremote cp emltrees.mpy :emltrees.mpy
mpremote cp emlneighbors.mpy :emlneighbors.mpy
```

NOTE: If there is no ready-made build for your device/architecture,
then you will need to build the .mpy module yourself.

## Usage

NOTE: Make sure to install the module first (see above)

Train a model with scikit-learn
```
pip install emlearn scikit-learn
python examples/xor_train.py
```

Copy model file to device

```
mpremote cp xor_model.csv :xor_model.csv
```

Run program that uses the model

```
mpremote run examples/xor_run.py
```

## Benchmarks

#### UCI handwriting digits

UCI ML hand-written digits datasets dataset from
[sklearn.datasets.load_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).
8x8 image, 64 features. Values are 4-bit integers (16 levels). 10 classes.

Running with a very simple RandomForest, 7 trees.
Reaches approx 86% accuracy.
Tested on Raspberry PI Pico, with RP2040 microcontroller (ARM Cortex M0 @ 133 MHz).

![Inferences per second](./benchmarks/digits_bench.png)

NOTE: over half of the time for emlearn case,
is spent on converting the Python lists of integers into a float array.
Removing that bottleneck would speed up things considerably.


## Developing locally

#### Prerequisites
These come in addition to the prequisites described above.

Make sure you have the dependencies needed to build for your platform.
See [MicroPython: Building native modules](https://docs.micropython.org/en/latest/develop/natmod.html).

We assume that micropython is installed in the same place as this repository.
If using another location, adjust `MPY_DIR` accordingly.

NOTE: As of August 2023, an out-of-tree patch is needed for MicroPython.
[micropython#12123: mpy_ld.py: Support complex RO sections](https://github.com/micropython/micropython/pull/12123).
This will hopefully be fixed in the coming months.

#### Build

Build the .mpy native module
```
make dist ARCH=armv6m MPY_DIR=../micropython
```

Install it on device
```
mpremote cp dist/armv6m*/emltrees.mpy :emltrees.mpy
```

#### Run tests

To build and run tests on host
```
make check
```

## Citations

If you use `emlearn-micropython` in an academic work, please reference it using:

```tex
@misc{emlearn_micropython,
  author       = {Jon Nordby},
  title        = {{emlearn-micropython: Efficient Machine Learning engine for MicroPython}},
  month        = aug,
  year         = 2023,
  doi          = {10.5281/zenodo.8212731},
  url          = {https://doi.org/10.5281/zenodo.8212731}
}
```

