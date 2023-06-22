import os
import sys
import pandas as pd
import numpy as np
from IPython.display import display

from tensorflow import keras

from sklearn.model_selection import train_test_split

from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
from Bio import SeqIO
import pickle as pkl
import time 

OUTPUT_TYPE = OutputType(False, 'numeric')
OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE)

# Split X and y in a dict where the key is the number of variants and the value are que indexes of the samples in X with that number of variants
# Count the number of variants between the wild type sequence and each sample
wt_sequence_file = "../../data/avgfp/avgfp_wt.fasta"
wt_sequence = SeqIO.read(wt_sequence_file, "fasta").seq

# Loading the dataset
X_file = "../../data/avgfp/avgfp_X.pkl"
y_file = "../../data/avgfp/avgfp_y.pkl"

print("Loading data...")
X = pkl.load(open(X_file, "rb"))
y = pkl.load(open(y_file, "rb"))

variants_dict = dict()
for i, seq in enumerate(X):
    variants = sum([1 for i in range(len(seq)) if seq[i] != wt_sequence[i]])
    if variants in variants_dict:
        variants_dict[variants].append(i)
    else:
        variants_dict[variants] = [i]
print(f"\tNumber of samples: {len(X)}")
# Sort variants_dict by key and print the number of samples with each number of variants
for k, v in sorted(variants_dict.items()):
    print(f"\tNumber of samples with {k} variants: {len(v)}")

train_indexes = []
train_variants = [1]
test_variants = [2]
for variant in train_variants:
    train_indexes.extend(variants_dict[variant])
test_indexes = []
for variant in test_variants:
    test_indexes.extend(variants_dict[variant])

# Getting the length of the sequences
seq_len = len(X[0])

# Train/test by variant number
X_train = np.array([X[i] for i in train_indexes])
y_train = np.array([y[i] for i in train_indexes])
X_test = np.array([X[i] for i in test_indexes])
y_test = np.array([y[i] for i in test_indexes])


# Val set
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42)


print("Train set length: ", len(X_train))
print("Validation set length: ", len(X_val))
print("Test set length: ", len(X_test))
print("Sequence length: ", seq_len)
print("Instance type: ", type(X_train[0]))
print("Label type: ", type(y_train[0]))

# batch_size: 8
# max_epochs_per_stage: 1000
# lr: 0.0001
# lr_with_frozen_pretrained_layers: 0.0001
# final_lr: 1e-05
# n_final_epochs: 200
# final_seq_len: 237
# dropout_rate: 0.5
# min_lr: 1e-09
# factor: 0.1
# patience_red: 50
# patience_early: 60
# begin_with_frozen_pretrained_layers: True

bath_size_values = [8]
max_epochs_per_stage_values = [200, 1000]
lr_values = [1e-04]
lr_with_frozen_pretrained_layers_values = [1e-04]
final_lrs = [1e-05]
n_final_epochs_values = [200]
final_seq_len_values = [seq_len+2]
dropout_rate_values = [0.5, 0.75]
min_lr = 1e-09
factors = [0.1]
patiences_red = [10, 50]
patiences_early = [20, 60]
begin_with_frozen_pretrained_layers_values = [True, False]

# Grid search
results_file = "results.txt"
for batch_size in bath_size_values:
    for max_epochs_per_stage in max_epochs_per_stage_values:
        for lr in lr_values:
            for lr_with_frozen_pretrained_layers in lr_with_frozen_pretrained_layers_values:
                for final_lr in final_lrs:
                    for n_final_epochs in n_final_epochs_values:
                        for final_seq_len in final_seq_len_values:
                            for dropout_rate in dropout_rate_values:
                                for factor in factors:
                                    for patience_red in patiences_red:
                                        for patience_early in patiences_early:
                                            for begin_with_frozen_pretrained_layers in begin_with_frozen_pretrained_layers_values:
                                                start = time.time()
                                                print(f"batch_size: {batch_size}")
                                                print(f"max_epochs_per_stage: {max_epochs_per_stage}")
                                                print(f"lr: {lr}")
                                                print(f"lr_with_frozen_pretrained_layers: {lr_with_frozen_pretrained_layers}")
                                                print(f"final_lr: {final_lr}")
                                                print(f"n_final_epochs: {n_final_epochs}")
                                                print(f"final_seq_len: {final_seq_len}")
                                                print(f"dropout_rate: {dropout_rate}")
                                                print(f"min_lr: {min_lr}")
                                                print(f"factor: {factor}")
                                                print(f"patience_red: {patience_red}")
                                                print(f"patience_early: {patience_early}")
                                                print(f"begin_with_frozen_pretrained_layers: {begin_with_frozen_pretrained_layers}")
                                                print("")

                                                # Loading the pre-trained model and fine-tuning it on the loaded dataset
                                                pretrained_model_generator, input_encoder = load_pretrained_model()

                                                # get_model_with_hidden_layers_as_outputs gives the model output access to the hidden layers (on top of the output)
                                                model_generator = FinetuningModelGenerator(pretrained_model_generator, OUTPUT_SPEC,
                                                                                        pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs, dropout_rate=dropout_rate)

                                                training_callbacks = [
                                                    keras.callbacks.ReduceLROnPlateau(
                                                        patience=patience_red, factor=factor, min_lr=min_lr, verbose=1),
                                                    keras.callbacks.EarlyStopping(patience=patience_early, restore_best_weights=True),
                                                ]
                                                finetune(model_generator, input_encoder, OUTPUT_SPEC, X_train, y_train, X_val, y_val, seq_len=seq_len+2, batch_size=batch_size, max_epochs_per_stage=max_epochs_per_stage, lr=lr,
                                                            begin_with_frozen_pretrained_layers=begin_with_frozen_pretrained_layers, lr_with_frozen_pretrained_layers=lr_with_frozen_pretrained_layers, n_final_epochs=n_final_epochs, final_seq_len=final_seq_len, final_lr=final_lr, callbacks=training_callbacks)
                                                # Evaluating the performance on the test-set

                                                results, confusion_matrix = evaluate_by_len(
                                                    model_generator, input_encoder, OUTPUT_SPEC, X_test, y_test, start_seq_len=seq_len+2, start_batch_size=32)

                                                end = time.time()

                                                # Saving the results
                                                with open(results_file, "a") as f:
                                                    f.write(f"---------------------------------------------------------------------------\n")
                                                    f.write(f"batch_size: {batch_size}\n")
                                                    f.write(f"max_epochs_per_stage: {max_epochs_per_stage}\n")
                                                    f.write(f"lr: {lr}\n")
                                                    f.write(f"lr_with_frozen_pretrained_layers: {lr_with_frozen_pretrained_layers}\n")
                                                    f.write(f"final_lr: {final_lr}\n")
                                                    f.write(f"n_final_epochs: {n_final_epochs}\n")
                                                    f.write(f"final_seq_len: {final_seq_len}\n")
                                                    f.write(f"dropout_rate: {dropout_rate}\n")
                                                    f.write(f"min_lr: {min_lr}\n")
                                                    f.write(f"factor: {factor}\n")
                                                    f.write(f"patience_red: {patience_red}\n")
                                                    f.write(f"patience_early: {patience_early}\n")
                                                    f.write(f"begin_with_frozen_pretrained_layers: {begin_with_frozen_pretrained_layers}\n")
                                                    f.write(f"\n")
                                                    f.write(f"Time (H:M:S): {time.strftime('%H:%M:%S', time.gmtime(end-start))}\n")
                                                    f.write(f"\n")
                                                    f.write(f"Spearman's rho: {results}\n")
                                                    f.write(f"---------------------------------------------------------------------------\n")
