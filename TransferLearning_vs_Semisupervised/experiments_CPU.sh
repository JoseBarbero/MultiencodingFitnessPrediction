#!/bin/bash

python Semisupervised_experiments.py --data ../data/avgfp/ --enc One_hot --cpus 30 --ssmethod SelfTraining --basemodel LogisticRegression --normalize False --trainvariants 1 --testvariants 2
python Supervised_experiments.py --data ../data/avgfp/ --enc One_hot --cpus 30 --model LogisticRegression --normalize False --trainvariants 1 --testvariants 2
python Supervised_experiments.py --data ../data/avgfp/ --enc One_hot --cpus 30 --model Ridge --normalize False --trainvariants 1 --testvariants 2

python Semisupervised_experiments.py --data ../data/avgfp/ --enc One_hot --cpus 30 --ssmethod SelfTraining --basemodel LogisticRegression --normalize False --trainvariants 1 2 3 4 5 6 7 8 9 10 --testvariants 1 2 3 4 5 6 7 8 9 10
python Supervised_experiments.py --data ../data/avgfp/ --enc One_hot --cpus 30 --model LogisticRegression --normalize False --trainvariants 1 2 3 4 5 6 7 8 9 10 --testvariants 1 2 3 4 5 6 7 8 9 10
python Supervised_experiments.py --data ../data/avgfp/ --enc One_hot --cpus 30 --model Ridge --normalize False --trainvariants 1 2 3 4 5 6 7 8 9 10 --testvariants 1 2 3 4 5 6 7 8 9 10