# Installation

Must have Eigen and FFTW3 installed globally.

```bash
git submodule update --init --recursive

# Build libsndfile
pushd third-party/libsndfile
mkdir build; cd build
cmake .. -G'Unix Makefiles'
cmake --build . --parallel
popd

# Build the project.
mkdir build; cd build
cmake .. -G'Unix Makefiles'
cmake --build .
```

# Usage

## Extract

Extract a feature vector from a wav file.

```bash
$ rm training_features.txt
$ for f in training/*.wav; do ./build/extract "$f" >> training_features.txt; done
"training/0_george_0.feat"
"training/0_george_1.feat"
"training/0_george_10.feat"
"training/0_george_11.feat"
"training/0_george_12.feat"
...
```

## Compute Basis

Pass a file listing the the `.feat` files from `./extract`.

```bash
$ ./build/basis training_features.txt
"training_features.scree"
"training_features.basis"
"training_features.mean"
```

## Reduce Dimensions

Use the precomputed mean and basis to reduce the dimensions of feature.

```bash
./build/reduce training_features.txt training_features 12
"training/0_george_0.reduced"
"training/0_george_1.reduced"
"training/0_george_10.reduced"
"training/0_george_11.reduced"
"training/0_george_12.reduced"
...
```

The last argument chooses how many dimensions to keep. Use the `training_features.scree` file to decide.
