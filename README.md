# Installation

Must have Eigen and FFTW3 installed globally.

```bash
git submodule update --init --recursive

# Build libsndfile
cd libsndfile
mkdir build; cd build
cmake .. -G'Unix Makefiles'
cmake --build .
cd ../../

# Build the project
mkdir build; cd build
cmake .. -G'Unix Makefiles'
cmake --build .
```
