# !/bin/bash
# for f in ../data/*_*/ ; do
#     model="LogisticRegression"
#     if [[ -d "./results/multiview_experiments_$(basename -- ${f})_${model}" ]]; then
#         echo "Dir ./results/multiview_experiments_$(basename -- ${f})_${model} already exists. Skipping."
#     else
#         python TestSSLearn_experiments.py $f 25
#     fi
# done

#python TestSSLearn_experiments.py ../data/0.6M_BMIMI 25
python TestSSLearn_experiments.py ../data/BRCA1_HUMAN_Fields2015_y2h 25
#python TestSSLearn_experiments.py ../data/avgfp 25
