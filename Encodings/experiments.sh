# !/bin/bash

# Run Masking experiments por every dataset directory in ../data/
for dataset in ../data/*/
do
    echo "Running Masking experiments for $dataset"
    python Masking_experiments.py --data $dataset --cpus 50
    #python Masking_extrapolation_experiments.py --data $dataset --cpus 32 --trainvariants 1 --testvariants 2
done
