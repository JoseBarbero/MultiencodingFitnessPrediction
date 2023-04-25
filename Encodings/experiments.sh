# !/bin/bash

# To avoid unintented multithreading:
# https://stackoverflow.com/questions/19257070/unintended-multithreading-in-python-scikit-learn/42124978#42124978
# terminal: export OPENBLAS_NUM_THREADS=1
# To know numpy/scipy config: https://stackoverflow.com/questions/9000164/how-to-check-blas-lapack-linkage-in-numpy-and-scipy

# Run Masking experiments por every dataset directory in ../data/
# for dataset in ../data/*/
# do
#     echo "Running Masking experiments for $dataset"
#     python Masking_experiments.py --data $dataset --cpus 50
#     #python Masking_extrapolation_experiments.py --data $dataset --cpus 32 --trainvariants 1 --testvariants 2
# done

python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/PABP_YEAST_Fields2013 --model DecisionTreeRegressor --normalize True --cpus 30 --trainvariants 1 --testvariants 2 --outdir masking_experiments_StandardScaler_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize True --cpus 30 --trainvariants 1 --testvariants 2 --outdir masking_experiments_StandardScaler_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize True --cpus 30 --trainvariants 1 --testvariants 3 --outdir masking_experiments_StandardScaler_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize True --cpus 30 --trainvariants 1 --testvariants 4 --outdir masking_experiments_StandardScaler_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize True --cpus 30 --trainvariants 1 --testvariants 5 --outdir masking_experiments_StandardScaler_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize True --cpus 30 --trainvariants 1 --testvariants 6 --outdir masking_experiments_StandardScaler_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize True --cpus 30 --trainvariants 1 --testvariants 7 --outdir masking_experiments_StandardScaler_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize True --cpus 30 --trainvariants 1 --testvariants 8 --outdir masking_experiments_StandardScaler_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize True --cpus 30 --trainvariants 1 --testvariants 9 --outdir masking_experiments_StandardScaler_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize True --cpus 30 --trainvariants 1 --testvariants 10 --outdir masking_experiments_StandardScaler_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize True --cpus 30 --trainvariants 1 --testvariants 11 --outdir masking_experiments_StandardScaler_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize True --cpus 30 --trainvariants 1 --testvariants 12 --outdir masking_experiments_StandardScaler_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize True --cpus 30 --trainvariants 1 --testvariants 13 --outdir masking_experiments_StandardScaler_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize True --cpus 30 --trainvariants 1 --testvariants 14 --outdir masking_experiments_StandardScaler_WeightedDecisionTree

python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/PABP_YEAST_Fields2013 --model DecisionTreeRegressor --normalize False --cpus 30 --trainvariants 1 --testvariants 2 --outdir masking_experiments_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize False --cpus 30 --trainvariants 1 --testvariants 2 --outdir masking_experiments_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize False --cpus 30 --trainvariants 1 --testvariants 3 --outdir masking_experiments_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize False --cpus 30 --trainvariants 1 --testvariants 4 --outdir masking_experiments_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize False --cpus 30 --trainvariants 1 --testvariants 5 --outdir masking_experiments_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize False --cpus 30 --trainvariants 1 --testvariants 6 --outdir masking_experiments_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize False --cpus 30 --trainvariants 1 --testvariants 7 --outdir masking_experiments_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize False --cpus 30 --trainvariants 1 --testvariants 8 --outdir masking_experiments_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize False --cpus 30 --trainvariants 1 --testvariants 9 --outdir masking_experiments_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize False --cpus 30 --trainvariants 1 --testvariants 10 --outdir masking_experiments_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize False --cpus 30 --trainvariants 1 --testvariants 11 --outdir masking_experiments_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize False --cpus 30 --trainvariants 1 --testvariants 12 --outdir masking_experiments_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize False --cpus 30 --trainvariants 1 --testvariants 13 --outdir masking_experiments_WeightedDecisionTree
python Masking_extrapolation_experiments_WeightedTrees.py --data ../data/avgfp/ --model DecisionTreeRegressor --normalize False --cpus 30 --trainvariants 1 --testvariants 14 --outdir masking_experiments_WeightedDecisionTree