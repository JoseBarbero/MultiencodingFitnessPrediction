# !/bin/bash

# To avoid unintented multithreading:
# https://stackoverflow.com/questions/19257070/unintended-multithreading-in-python-scikit-learn/42124978#42124978
# terminal: export OPENBLAS_NUM_THREADS=1
# To know numpy/scipy config: https://stackoverflow.com/questions/9000164/how-to-check-blas-lapack-linkage-in-numpy-and-scipy

# Run Masking experiments por every dataset directory in ../data/
for dataset in ../data/*/
do
    echo "Running Masking experiments for $dataset"
    python Masking_experiments.py --data $dataset --cpus 100
    #python Masking_extrapolation_experiments.py --data $dataset --cpus 32 --trainvariants 1 --testvariants 2
done
