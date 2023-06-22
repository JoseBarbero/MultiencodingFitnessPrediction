#!/bin/bash

python ProteinBERT_experiments.py --data ../data/avgfp/ --trainvariants 1 --testvariants 2 --predmode regression
python ProteinBERT_experiments.py --data ../data/avgfp/ --trainvariants 1 2 3 4 5 6 7 8 9 10 --testvariants 1 2 3 4 5 6 7 8 9 10 --predmode regression