import pandas as pd
import numpy as np
import pickle as pk
import os
from collections.abc import Iterable
from Bio import SeqIO
from utils import read_variants, read_a2m, generate_unambiguous_homologs

class SequencesEncoder: 
    
    def __init__(self, wt, start_pos): 
        
        self.wt = wt
        self.start_pos = start_pos


class SequencesOneHotEncoder(SequencesEncoder): 
    
    def __init__(self, wt, start_pos): 
        
        super().__init__(wt, start_pos)
        
        
    def encode(self, sequences): 
        
        return self._one_hot(sequences)
        

    
    def _one_hot(self, sequences): 
        
        aa_to_onehot = {
            "A": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "R": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "N": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "D": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "C": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "Q": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "E": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "G": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "H": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "I": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "L": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "K": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "M": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            "F": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            "P": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            "S": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            "T": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            "W": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            "Y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "V": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            ".": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            "-": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            #"X": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
            #"B": [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            #"J": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            #"Z": [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        
        sequences_encoded = []
        for instance in sequences: 
            instance_encoded = []
            for value in instance: 
                instance_encoded.append(aa_to_onehot[value])
            sequences_encoded.append(instance_encoded)
            
        return np.array(sequences_encoded)



class SequencesDCAEncoder(SequencesEncoder): 
    
    def __init__(self, wt, start_pos, params_file): 
        
        super().__init__(wt, start_pos)
        
        self.params_file = params_file
        
        #read plmc params 
        self._read_plmc_v2()
        
        self.index_map = {b: a for a, b in enumerate(self.index_list)}
        self.alphabet_map = {s: i for i, s in enumerate(self.alphabet)}
        
        # in non-gap mode, focus sequence is still coded with a gap character,
        # but gap is not part of model alphabet anymore; so if mapping crashes
        # that means there is a non-alphabet character in sequence array
        # and therefore there is no focus sequence.
        try:
            self.target_seq_mapped = np.array([self.alphabet_map[x] for x in self.target_seq])
            self.has_target_seq = (np.sum(self.target_seq_mapped) > 0)
        except KeyError:
            self.target_seq_mapped = np.zeros((self.L), dtype=np.int32)
            self.has_target_seq = False
        
    def encode_variants(self, sequences, y, variants): 
        encoded_sequences, encoded_y, indexes = self._encode_variants(sequences, y, variants)
        return encoded_sequences, encoded_y, indexes

    def encode_homologs(self, sequences):  
        return self._encode_homologs(sequences)

    def encode_wt(self): 
        return self._encode_wt()
        
    
    def _read_plmc_v2(self): 
        
        precision='float32'
        
        with open(self.params_file, "rb") as f:
            # model length, number of symbols, valid/invalid sequences
            # and iterations
            self.L, self.num_symbols, self.N_valid, self.N_invalid, self.num_iter = (
                np.fromfile(f, "int32", 5)
            )

            # theta, regularization weights, and effective number of samples
            self.theta, self.lambda_h, self.lambda_J, self.lambda_group, self.N_eff = (
                np.fromfile(f, precision, 5)
            )

            # Read alphabet (make sure we get proper unicode rather than byte string)
            self.alphabet = np.fromfile(
                f, "S1", self.num_symbols
            ).astype("U1")

            # weights of individual sequences (after clustering)
            self.weights = np.fromfile(
                f, precision, self.N_valid + self.N_invalid
            )

            # target sequence and index mapping, again ensure unicode
            self.target_seq = np.fromfile(f, "S1", self.L).astype("U1")
            self.index_list = np.fromfile(f, "int32", self.L)

            # single site frequencies f_i and fields h_i
            self.f_i, = np.fromfile(
                f, dtype=(precision, (self.L, self.num_symbols)), count=1
            )

            self.h_i, = np.fromfile(
                f, dtype=(precision, (self.L, self.num_symbols)), count=1
            )

            # pair frequencies f_ij and pair couplings J_ij / J_ij
            self.f_ij = np.zeros(
                (self.L, self.L, self.num_symbols, self.num_symbols)
            )

            self.J_ij = np.zeros(
                (self.L, self.L, self.num_symbols, self.num_symbols)
            )

            for i in range(self.L - 1):
                for j in range(i + 1, self.L):
                    self.f_ij[i, j], = np.fromfile(
                        f, dtype=(precision, (self.num_symbols, self.num_symbols)),
                        count=1
                    )
                    self.f_ij[j, i] = self.f_ij[i, j].T

            for i in range(self.L - 1):
                for j in range(i + 1, self.L):
                    self.J_ij[i, j], = np.fromfile(
                        f, dtype=(precision, (self.num_symbols, self.num_symbols)),
                        count=1
                    )
                    self.J_ij[j, i] = self.J_ij[i, j].T
                    
                    
    def __map(self, indices, mapping):
        """
        Applies a mapping either to a single index, or to a list of indices

        Parameters
        ----------
        indices : Iterable of items to be mapped, or single item
        mapping: Dictionary containing mapping into new space

        Returns
        -------
        Iterable, or single item
            Items mapped into new space
        """
        if ((isinstance(indices, Iterable) and not isinstance(indices, str)) or
                (isinstance(indices, str) and len(indices) > 1)):
            return np.array(
                [mapping[i] for i in indices]
            )
        else:
            return mapping[indices]  
        
                    
    def Jij(self, i, j, A_i, A_j):
        i = self.__map(i, self.index_map)
        j = self.__map(j, self.index_map)
        A_i = self.__map(A_i, self.alphabet_map)
        
        #TODO check if this is ok
        if A_j not in ['.', '-']: 
            A_j = self.__map(A_j, self.alphabet_map)
            return self.J_ij[i,j,A_i, A_j]
        else: 
            return 0.0
    
    def hi(self, i, A_i): 
        i = self.__map(i, self.index_map)
        A_i = self.__map(A_i, self.alphabet_map)
        return self.h_i[i, A_i]
    
    def Ji(self, i, A_i, sequence): 
        Ji = 0.0
        for j, A_j in zip(self.index_list, sequence): 
            Ji += self.Jij(i, j, A_i, A_j)
        return Ji
    
    def _encode_homologs(self, sequences):
                
        #encode sequences 
        encoded_sequences = []
        for sequence in sequences:
            
            trimmed_sequence = [sequence[i-1] for i in self.index_list]

            encoded_sequence = []
            for i in self.index_list:
                A_i = sequence[i-1]
                if A_i not in ['.', '-']: 
                    encoded_sequence.append(self.hi(i, A_i) + 0.5 * self.Ji(i, A_i, trimmed_sequence))
                else: 
                    encoded_sequence.append(0.0)
            encoded_sequences.append(encoded_sequence)
            
        return np.array(encoded_sequences)

    def _encode_variants(self, sequences, y, variants):
        #get indexes of sequences that have all substitutions in index_list
        unique_indexes = []
        for index, substitutions in enumerate(variants): 
            substitutions = substitutions.split(',')
            pos_in_index_list = True
            for s in substitutions: 

                pos = int(s[1:-1])
                
                if pos-(self.start_pos-1) not in self.index_list: 
                    pos_in_index_list = False
                    break
            if pos_in_index_list: 
                unique_indexes.append(index)
                
        #encode sequences in unique indexes 
        unique_encoded_sequences = []
        unique_y = []
        for index in unique_indexes:
            sequence = sequences[index]
            trimmed_sequence = [sequence[i-1] for i in self.index_list]

            encoded_sequence = []
            for i in self.index_list:
                A_i = sequence[i-1]
 
                encoded_sequence.append(self.hi(i, A_i) + 0.5 * self.Ji(i, A_i, trimmed_sequence))
    
            unique_y.append(y[index])
            unique_encoded_sequences.append(encoded_sequence)
            
        return np.array(unique_encoded_sequences), np.array(unique_y), unique_indexes
    
    def _encode_wt(self):
        encoded_wt = np.zeros(self.target_seq.size, dtype=float)
        for idx, (i, A_i) in enumerate(zip(self.index_list, self.target_seq)):
            encoded_wt[idx] = self.hi(i, A_i) + 0.5 * self.Ji(i, A_i, self.target_seq)
        return encoded_wt
    
def save_encoding(data_dict, log_file, n_processes, output_dir):
    
    for key in data_dict.keys(): 
        
        fasta = data_dict[key]['fasta']
        start_pos = data_dict[key]['start_pos']
        csv = data_dict[key]['csv']
        a2m = data_dict[key]['a2m']
        params = data_dict[key]['params']
        
        wt = np.array(SeqIO.read(open(fasta), 'fasta'))
        #pk.dump(wt, open('../'+key+'_wt.pk', 'wb'))
        
        # with open(log_file, 'a') as writer:
        #     writer.write('\n>>> '+key+':\n')
        #     writer.write('Reading .csv\n')
        sequences, y, variants = read_variants(csv, fasta, start_pos=start_pos)
        #pk.dump(sequences, open('../'+key+'_sequences.pk', 'wb'))
        #pk.dump(y, open('../'+key+'_y.pk', 'wb'))
        #pk.dump(variants, open('../'+key+'_variants.pk', 'wb'))
        
        # with open(log_file, 'a') as writer:
        #     writer.write('Reading .a2m\n')
        homologs = read_a2m(a2m)
        
        # with open(log_file, 'a') as writer:
        #     writer.write('Generating unambiguous homologs\n')
        unambiguous_homologs = generate_unambiguous_homologs(homologs, n_processes=n_processes, mode='random')
        #pk.dump(unambiguous_homologs, open('../'+key+'_unambiguous_homologs.pk', 'wb'))
        
        # with open(log_file, 'a') as writer:
        #     writer.write('One hot encoding\n')
        # ohe = SequencesOneHotEncoder(wt, start_pos=start_pos)
        # Xl_ohe = ohe.encode(sequences)
        # Xu_ohe = ohe.encode(unambiguous_homologs)
        # pk.dump(Xl_ohe, open('../'+key+'_Xl_ohe.pk', 'wb'))
        # pk.dump(Xu_ohe, open('../'+key+'_Xu_ohe.pk', 'wb'))
        
        # with open(log_file, 'a') as writer:
        #     writer.write('DCA encoding\n')
        dcae = SequencesDCAEncoder(wt, start_pos, params)
        Xl_dcae, y_dcae, indexes = dcae.encode_variants(sequences, y, variants)
        Xu_dcae = dcae.encode_homologs(unambiguous_homologs)
        pk.dump(Xl_dcae, open(os.path.join(output_dir, key+'_X_merge.pkl', 'wb')))
        #pk.dump(y_dcae, open('../'+key+'_y_dcae.pk', 'wb'))
        #pk.dump(indexes, open('../'+key+'_indexes.pk', 'wb'))
        #pk.dump(Xu_dcae, open('../'+key+'_Xu_dcae.pk', 'wb'))

def get_start_pos(fasta_file):
    return int(SeqIO.read(fasta_file, "fasta").description.lower().split("start: ")[1].split(",")[0])
    
if __name__=="__main__":
    
    data_path = os.path.join('..', '..', 'data')

    data_dict = dict()

    #for every dir in data_path
    for subdir in os.listdir(data_path):
        jackhmmer_dir = os.path.join(data_path, subdir, 'jhmmer')
        
        fasta_file = os.path.join(data_path, subdir, subdir+'_wt.fasta')
        csv_file = os.path.join(data_path, subdir, subdir+'_encoded.csv')
        a2m_file = os.path.join(jackhmmer_dir, subdir+'.a2m')
        params_file = os.path.join(jackhmmer_dir, subdir+'.params')

        # If sto file exists, skip
        if not os.path.exists(a2m_file) or os.path.exists(os.path.join(data_path, subdir, subdir+'_X_merge.pkl')):
            print('Skipping '+subdir)
        else:
            data_dict[subdir] = {'fasta': fasta_file,
                            'start_pos': get_start_pos(fasta_file),
                            'csv': csv_file,
                            'a2m': a2m_file,
                            'params': params_file}
            print("Encoding for "+subdir)
            print("\tStarting position: "+str(get_start_pos(fasta_file)))
            log_file = 'encoding_log.log'
            save_encoding(data_dict, log_file, 50, os.path.join(data_path, subdir))
            
    
    #Comment and uncomment for changing datasets 
    """
    data_dict['avgfp'] = {'fasta': os.path.join(data_path, 'avgfp', 'avgfp.fasta'),
                          'start_pos': 1,
                          'csv': os.path.join(data_path, 'avgfp', 'avgfp_encoded.csv'), 
                          'a2m': os.path.join(data_path, 'avgfp', 'avgfp_jhmmer.a2m'), 
                          'params': os.path.join(data_path, 'avgfp', 'avgfp_plmc.params')}
    data_dict['bg_strsq'] = {'fasta': os.path.join(data_path, 'bg_strsq', 'bg_strsq.fasta'),
                             'start_pos': 2,
                             'csv': os.path.join(data_path, 'bg_strsq', 'BG_STRSQ_Abate2015_encoded.csv'), 
                             'a2m': os.path.join(data_path, 'bg_strsq', 'bg_strsq_jhmmer.a2m'), 
                             'params': os.path.join(data_path, 'bg_strsq', 'bg_strsq_plmc.params')}
    data_dict['blat_ecolx_1'] = {'fasta': os.path.join(data_path, 'blat_ecolx', 'blat_ecolx.fasta'),
                                 'start_pos': 1,
                                 'csv': os.path.join(data_path, 'blat_ecolx', 'BLAT_ECOLX_Ostermeier2014_encoded.csv'), 
                                 'a2m': os.path.join(data_path, 'blat_ecolx', 'blat_ecolx_jhmmer.a2m'), 
                                 'params': os.path.join(data_path, 'blat_ecolx', 'blat_ecolx_plmc.params')}
    data_dict['blat_ecolx_2'] = {'fasta': os.path.join(data_path, 'blat_ecolx', 'blat_ecolx.fasta'),
                                 'start_pos': 1,
                                 'csv': os.path.join(data_path, 'blat_ecolx', 'BLAT_ECOLX_Palzkill2012_encoded.csv'), 
                                 'a2m': os.path.join(data_path, 'blat_ecolx', 'blat_ecolx_jhmmer.a2m'), 
                                 'params': os.path.join(data_path, 'blat_ecolx', 'blat_ecolx_plmc.params')}
    data_dict['blat_ecolx_3'] = {'fasta': os.path.join(data_path, 'blat_ecolx', 'blat_ecolx.fasta'),
                                 'start_pos': 1,
                                 'csv': os.path.join(data_path, 'blat_ecolx', 'BLAT_ECOLX_Ranganathan2015_encoded.csv'), 
                                 'a2m': os.path.join(data_path, 'blat_ecolx', 'blat_ecolx_jhmmer.a2m'), 
                                 'params': os.path.join(data_path, 'blat_ecolx', 'blat_ecolx_plmc.params')}
    data_dict['blat_ecolx_4'] = {'fasta': os.path.join(data_path, 'blat_ecolx', 'blat_ecolx.fasta'),
                                 'start_pos': 1,
                                 'csv': os.path.join(data_path, 'blat_ecolx', 'BLAT_ECOLX_Tenaillon2013-singles_encoded.csv'), 
                                 'a2m': os.path.join(data_path, 'blat_ecolx', 'blat_ecolx_jhmmer.a2m'), 
                                 'params': os.path.join(data_path, 'blat_ecolx', 'blat_ecolx_plmc.params')}
    data_dict['brca1_human_1'] = {'fasta': os.path.join(data_path, 'brca1_human', 'brca1.fasta'),
                                  'start_pos': 2,
                                  'csv': os.path.join(data_path, 'brca1_human', 'BRCA1_HUMAN_Fields2015_e3_encoded.csv'), 
                                  'a2m': os.path.join(data_path, 'brca1_human', 'brca1_human_jhmmer.a2m'), 
                                  'params': os.path.join(data_path, 'brca1_human', 'brca1_human_plmc.params')}
    data_dict['brca1_human_2'] = {'fasta': os.path.join(data_path, 'brca1_human', 'brca1.fasta'),
                                  'start_pos': 2,
                                  'csv': os.path.join(data_path, 'brca1_human', 'BRCA1_HUMAN_Fields2015_y2h_encoded.csv'), 
                                  'a2m': os.path.join(data_path, 'brca1_human', 'brca1_human_jhmmer.a2m'), 
                                  'params': os.path.join(data_path, 'brca1_human', 'brca1_human_plmc.params')}
    data_dict['gal4_yeast'] = {'fasta': os.path.join(data_path, 'gal4_yeast', 'gal4_yeast.fasta'),
                               'start_pos': 2,
                               'csv': os.path.join(data_path, 'gal4_yeast', 'GAL4_YEAST_Shendure2015_encoded.csv'), 
                               'a2m': os.path.join(data_path, 'gal4_yeast', 'gal4_yeast_jhmmer.a2m'), 
                               'params': os.path.join(data_path, 'gal4_yeast', 'gal4_yeast_plmc.params')}
    data_dict['hg_flu'] = {'fasta': os.path.join(data_path, 'hg_flu', 'hg_flu.fasta'),
                           'start_pos': 2,
                           'csv': os.path.join(data_path, 'hg_flu', 'HG_FLU_Bloom2016_encoded.csv'), 
                           'a2m': os.path.join(data_path, 'hg_flu', 'hg_flu_jhmmer.a2m'), 
                           'params': os.path.join(data_path, 'hg_flu', 'hg_flu_plmc.params')}

    data_dict['hsp82_yeast'] = {'fasta': os.path.join(data_path, 'hsp82_yeast', 'hsp82_yeast.fasta'),
                                'start_pos': 2,
                                'csv': os.path.join(data_path, 'hsp82_yeast', 'HSP82_YEAST_Bolon2016_encoded.csv'), 
                                'a2m': os.path.join(data_path, 'hsp82_yeast', 'hsp82_yeast_jhmmer.a2m'), 
                                'params': os.path.join(data_path, 'hsp82_yeast', 'hsp82_yeast_plmc.params')}
                                
    data_dict['kka2_klepn'] = {'fasta': os.path.join(data_path, 'kka2_klepn', 'kka2_klepn.fasta'),
                               'start_pos': 1,
                               'csv': os.path.join(data_path, 'kka2_klepn', 'KKA2_KLEPN_Mikkelsen2014_encoded.csv'), 
                               'a2m': os.path.join(data_path, 'kka2_klepn', 'kka2_klepn_jhmmer.a2m'), 
                               'params': os.path.join(data_path, 'kka2_klepn', 'kka2_klepn_plmc.params')}
    data_dict['mth3_haeaestabilized'] = {'fasta': os.path.join(data_path, 'mth3_haeaestabilized', 'mth3.fasta'),
                                         'start_pos': 2,
                                         'csv': os.path.join(data_path, 'mth3_haeaestabilized', 'MTH3_HAEAESTABILIZED_Tawfik2015_encoded.csv'), 
                                         'a2m': os.path.join(data_path, 'mth3_haeaestabilized', 'mth3_haeaestabilized_jhmmer.a2m'), 
                                         'params': os.path.join(data_path, 'mth3_haeaestabilized', 'mth3_haeaestabilized_plmc.params')}
    data_dict['pabp_yeast_1'] = {'fasta': os.path.join(data_path, 'pabp_yeast', 'pabp_yeast.fasta'),
                                 'start_pos': 126,
                                 'csv': os.path.join(data_path, 'pabp_yeast', 'PABP_YEAST_Fields2013-singles_encoded.csv'), 
                                 'a2m': os.path.join(data_path, 'pabp_yeast', 'pabp_yeast_jhmmer.a2m'), 
                                 'params': os.path.join(data_path, 'pabp_yeast', 'pabp_yeast_plmc.params')}
    data_dict['pabp_yeast_2'] = {'fasta': os.path.join(data_path, 'pabp_yeast', 'pabp_yeast.fasta'),
                                 'start_pos': 126,
                                 'csv': os.path.join(data_path, 'pabp_yeast', 'PABP_YEAST_Fields2013-doubles_encoded.csv'), 
                                 'a2m': os.path.join(data_path, 'pabp_yeast', 'pabp_yeast_jhmmer.a2m'), 
                                 'params': os.path.join(data_path, 'pabp_yeast', 'pabp_yeast_plmc.params')}
    data_dict['polg_hcvjf'] = {'fasta': os.path.join(data_path, 'polg_hcvjf', 'polg_hcvjf.fasta'),
                               'start_pos': 1994,
                               'csv': os.path.join(data_path, 'polg_hcvjf', 'POLG_HCVJF_Sun2014_encoded.csv'), 
                               'a2m': os.path.join(data_path, 'polg_hcvjf', 'polg_hcvjf_jhmmer.a2m'), 
                               'params': os.path.join(data_path, 'polg_hcvjf', 'polg_hcvjf_plmc.params')}
    data_dict['rl401_yeast_1'] = {'fasta': os.path.join(data_path, 'rl401_yeast', 'rl401.fasta'),
                                   'start_pos': 2,
                                   'csv': os.path.join(data_path, 'rl401_yeast', 'RL401_YEAST_Bolon2013_encoded.csv'), 
                                   'a2m': os.path.join(data_path, 'rl401_yeast', 'rl401_yeast_jhmmer.a2m'), 
                                   'params': os.path.join(data_path, 'rl401_yeast', 'rl401_yeast_plmc.params')}
    data_dict['rl401_yeast_2'] = {'fasta': os.path.join(data_path, 'rl401_yeast', 'rl401.fasta'),
                                  'start_pos': 2,
                                  'csv': os.path.join(data_path, 'rl401_yeast', 'RL401_YEAST_Bolon2014_encoded.csv'), 
                                  'a2m': os.path.join(data_path, 'rl401_yeast', 'rl401_yeast_jhmmer.a2m'), 
                                  'params': os.path.join(data_path, 'rl401_yeast', 'rl401_yeast_plmc.params')}

    data_dict['ube4b_mouse'] = {'fasta': os.path.join(data_path, 'ube4b_mouse', 'ueb_mouse.fasta'),
                                'start_pos': 1072,
                                'csv': os.path.join(data_path, 'ube4b_mouse', 'UBE4B_MOUSE_Klevit2013-singles_encoded.csv'), 
                                'a2m': os.path.join(data_path, 'ube4b_mouse', 'ube4b_mouse_jhmmer.a2m'), 
                                'params': os.path.join(data_path, 'ube4b_mouse', 'ube4b_mouse_plmc.params')}
    data_dict['yap1_human'] = {'fasta': os.path.join(data_path, 'yap1_human', 'yap1_human.fasta'),
                               'start_pos': 170,
                               'csv': os.path.join(data_path, 'yap1_human', 'YAP1_HUMAN_Fields2012-singles_encoded.csv'), 
                               'a2m': os.path.join(data_path, 'yap1_human', 'yap1_human_jhmmer.a2m'), 
                               'params': os.path.join(data_path, 'yap1_human', 'yap1_human_plmc.params')}
    """

    
    