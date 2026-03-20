# FHE-MP-CNN Test

Note that 384GB RAM is required (refer to FHE-MP-CNN) to run the code. Or the program will break in "create_galois_keys".

## Run the following commands to build an conda env for test:
```bash
conda create -n mpcnn python=3.9
conda activate mpcnn
conda install -c conda-forge gxx_linux-64 cmake make m4

wget https://gmplib.org/download/gmp/gmp-6.2.1.tar.xz
tar xf gmp-6.2.1.tar.xz
cd gmp-6.2.1

make distclean
./configure --prefix=$CONDA_PREFIX CC=gcc CFLAGS="-O2" --build=x86_64-pc-linux-gnu
make -j4
make check 
make install
cd ..

wget https://libntl.org/ntl-11.4.3.tar.gz
tar xzf ntl-11.4.3.tar.gz
cd ntl-11.4.3/src

./configure PREFIX=$CONDA_PREFIX NTL_GMP_LIP=on GMP_PREFIX=$CONDA_PREFIX
make -j4
make check
make install
cd ../..

cd cnn_ckks/cpu-ckks/single-key/seal-modified-3.6.6

cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5

cmake --build build -j4
cmake --install build
```

## Run the following commands to compile and test (./cnn layer_num dataset startnum endnum precison_par ../../pretrained_parameters/model_path threads_num):

```bash
conda activate mpcnn
cd cnn_ckks/build
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
./cnn 20 10 0 8 deg13 ../../pretrained_parameters/xxxxxx 8
```