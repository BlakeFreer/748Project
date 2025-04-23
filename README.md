# Spoken Digit Recognition

## Installation

Must have Eigen, FFTW3, CMake, and Python3 installed globally.

```bash
git submodule update --init --recursive
python3 -m pip install -r requirements.txt

# Build libsndfile
pushd third-party/libsndfile
mkdir build; cd build
cmake .. -G'Unix Makefiles'
cmake --build . --parallel
popd

# Build my project.
mkdir build; cd build
cmake .. -G'Unix Makefiles'
cmake --build .
```

## Usage

### 1. Partition training and testing data

```bash
$ python3 partition.py example
Found 3000 audio files in free-spoken-digit-dataset/recordings.
Copied 2400 training files to example/train_data.
Copied 600 testing files to example/test_data.
```

`example` is the name of the folder which holds this dataset.

The Free Spoken Digit Dataset contains 3000 recordings: 6 speakers with 50 recordings for digits 0-9.

By default, `partition.py` will use 40 recordings from each speaker and digit for training and the remaining 10 recordings for testing.

You can change the partitioning scheme by passing a regex search string with the `--regex` flag. Any filename containing this regex will be used for testing while all others will be used for training.

For example, to test on the "Theo" recordings and train on all others, partition with

```bash
python3 partition.py example2 --regex theo
```

### 2. Run the classifier

```bash
source pipeline.sh example
```

Where `example` is the name of the partition folder. Some steps in the script take up to a minute so be patient.

### 3. Plot the results

```bash
python3 plot.py example
```

Plots the confusion matrix and PCA scree plot for the `example` folder.
