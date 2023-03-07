import numpy as np
import pandas as pd
import random
import multiprocessing
from collections import deque
from Bio import SeqIO

def read_variants(filename, wt_fasta, start_pos): 
    
    wt = np.array(SeqIO.read(open(wt_fasta), 'fasta'))
        
    df = pd.read_csv(filename, delimiter=';')
    variants = df['variant'].to_numpy()
    
    sequences = []
    for row in variants: 

        vs = row.split(',')
        sequence = wt.copy()
        
        for variant in vs: 
            old_amino = variant[0]
            pos = int(variant[1:-1])
            new_amino = variant[-1:]
            
            assert(old_amino == sequence[pos-start_pos])
            
            sequence[pos-start_pos] = new_amino
        sequences.append(sequence)
     
    return np.array(sequences), df['y'].to_numpy(), df['variant'].to_numpy()
    

def read_a2m(filename): 
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    #parse file to list of sequences 
    sequences =  []
    sequence = None
    for line in lines: 
        if '>' not in line: 
            if sequence == None: 
                sequence = line[:-1]
            else :
                sequence = sequence + line[:-1]
        elif sequence != None: 
            sequences.append(sequence)
            sequence = None
    
    homologs = []
    for sequence in sequences[1:]:
        sequence = sequence.upper()
        homologs.append(np.array(list(sequence)))
    
    return np.array(homologs)


def generate_unambiguous_homologs(homologs, n_processes=1, mode='all', random_seed=1234): 
    # modes= 'random', 'all'

    homologs_split = np.array_split(homologs, n_processes)
    manager = multiprocessing.Manager()
    result = manager.list()
    processes = []

    for h in homologs_split:
        p = multiprocessing.Process(target=_job, args=[h, result, mode])
        p.start()
        processes.append(p)
    
    for p in processes: 
        p.join()
    
    array_result = np.array([x for x in result])
    return array_result
        
    
def _job(homologs, result, mode, random_seed=1234): 
    
    random.seed(random_seed)
    queue = deque(homologs)
    while len(queue) != 0: 
        
        sequence = queue.pop()
        
        flag_symbol_found = False
        for i, amino in enumerate(sequence): 
            
            if flag_symbol_found == False: 
                posible_symbols = []

                if amino == 'X': 
                    posible_symbols = ['A', 'R', 'N', 'Q', 'E', 'G', 'H', 'I', 'L',
                                       'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
                elif amino == 'B':
                    posible_symbols = ['A', 'N']
                elif amino == 'J': 
                    posible_symbols = ['I', 'L']
                elif amino == 'Z': 
                    posible_symbols = ['Q', 'E']

                if len(posible_symbols) != 0: 
                    
                    flag_symbol_found = True
                    
                    if mode == 'random': 
                        posible_symbols = [random.choice(posible_symbols)]
                        
                    for symbol in posible_symbols: 

                        new_sequence = sequence.copy()
                        new_sequence[i] = symbol
                        queue.append(new_sequence)


        if flag_symbol_found == False:    
            result.append(sequence)
                