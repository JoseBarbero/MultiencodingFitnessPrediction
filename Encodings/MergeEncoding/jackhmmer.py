import os
from Bio import SeqIO
from sto2a2m import convert_sto2a2m

def run_jackhmmer(data_dict): 
    
    for key in data_dict.keys(): 
        fasta = data_dict[key]['fasta']
        start_pos = data_dict[key]['start_pos']
        sto = data_dict[key]['sto']
        a2m = data_dict[key]['a2m']
        params = data_dict[key]['params']

        wt = SeqIO.read(open(fasta), 'fasta')
        length = len(wt)
        incT = length*0.5
        
        #CALL JACKHMMER
        with open('jackhmmer_log.log', 'a') as writer:
            writer.write('>>> '+key+':\n')
            writer.write('Running jackhmmer\n')
        os.system('jackhmmer --incT '+str(incT)+' --cpu 64 --noali -A '+sto+' '+fasta+' '+uniref_path) 
        
        #CONVERT .sto TO .a2m
        with open('jackhmmer_log.log', 'a') as writer:
            writer.write('Converting .sto to .a2m\n')
        n_seqs,n_active_sites,n_sites=convert_sto2a2m(sto, 0.3, 0.5)
        le = 0.2*(n_active_sites-1)
        
        #CALL PLMC
        with open('jackhmmer_log.log', 'a') as writer:
            writer.write('Running plmc\n')
            writer.write('\n')
        os.system('../../../plmc/bin/plmc -o '+params+' -n 64 -le '+str(le)+' -m 3500 -g -f '+key+' '+a2m)

def get_start_pos(fasta_file):
    return int(SeqIO.read(fasta_file, "fasta").description.lower().split("start: ")[1].split(",")[0])

if __name__=="__main__":
    
    data_path = os.path.join('..', '..', 'data')
    uniref_path = '/home/jabarbero/Uniref/uniref100/uniref100.fasta'

    data_dict = dict()

    #for every dir in data_path
    for subdir in os.listdir(data_path):
        jackhmmer_dir = os.path.join(data_path, subdir, 'jhmmer')
        if not os.path.exists(jackhmmer_dir):
            os.mkdir(jackhmmer_dir)

        fasta_file = os.path.join(jackhmmer_dir, subdir+'_wt.fasta')
        sto_file = os.path.join(jackhmmer_dir, subdir+'.sto')
        a2m_file = os.path.join(jackhmmer_dir, subdir+'.a2m')
        params_file = os.path.join(jackhmmer_dir, subdir+'.params')

        # If sto file exists, skip
        if os.path.exists(sto_file):
            print('Skipping '+subdir)
        else:
            data_dict[subdir] = {'fasta': fasta_file,
                            'start_pos': get_start_pos(fasta_file),
                            'sto': sto_file,
                            'a2m': a2m_file,
                            'params': params_file}
            print("Running jackhmmer for "+subdir)
            print("\tStarting position: "+str(get_start_pos(fasta_file)))
            run_jackhmmer(data_dict)


    """
    data_dict['avgfp'] = {'fasta': os.path.join(data_path, 'avgfp', 'avgfp.fasta'), 
                          'start_pos': 1,
                          'sto': os.path.join(data_path, 'avgfp', 'avgfp_jhmmer.sto'), 
                          'a2m': os.path.join(data_path, 'avgfp', 'avgfp_jhmmer.a2m'), 
                          'params': os.path.join(data_path, 'avgfp', 'avgfp_plmc.params')}   
    data_dict['bg_strsq'] = {'fasta': os.path.join(data_path, 'bg_strsq', 'bg_strsq.fasta'), 
                             'start_pos': 2, 
                             'sto': os.path.join(data_path, 'bg_strsq', 'bg_strsq_jhmmer.sto'), 
                             'a2m': os.path.join(data_path, 'bg_strsq', 'bg_strsq_jhmmer.a2m'),
                             'params': os.path.join(data_path, 'bg_strsq', 'bg_strsq_plmc.params')}
    data_dict['blat_ecolx'] = {'fasta': os.path.join(data_path, 'blat_ecolx', 'blat_ecolx.fasta'), 
                               'start_pos': 1, 
                               'sto': os.path.join(data_path, 'blat_ecolx', 'blat_ecolx_jhmmer.sto'), 
                               'a2m': os.path.join(data_path, 'blat_ecolx', 'blat_ecolx_jhmmer.a2m'), 
                               'params': os.path.join(data_path, 'blat_ecolx', 'blat_ecolx_plmc.params')}
    data_dict['brca1_human'] = {'fasta': os.path.join(data_path, 'brca1_human', 'brca1.fasta'), 
                                'start_pos': 2,
                                'sto': os.path.join(data_path, 'brca1_human', 'brca1_human_jhmmer.sto'), 
                                'a2m': os.path.join(data_path, 'brca1_human', 'brca1_human_jhmmer.a2m'), 
                                'params': os.path.join(data_path, 'brca1_human', 'brca1_human_plmc.params')}
    data_dict['gal4_yeast'] = {'fasta': os.path.join(data_path, 'gal4_yeast', 'gal4_yeast.fasta'), 
                               'start_pos': 2,
                               'sto': os.path.join(data_path, 'gal4_yeast', 'gal4_yeast_jhmmer.sto'), 
                               'a2m': os.path.join(data_path, 'gal4_yeast', 'gal4_yeast_jhmmer.a2m'), 
                               'params': os.path.join(data_path, 'gal4_yeast', 'gal4_yeast_plmc.params')}
    data_dict['hg_flu'] = {'fasta': os.path.join(data_path, 'hg_flu', 'hg_flu.fasta'), 
                           'start_pos': 2,
                           'sto': os.path.join(data_path, 'hg_flu', 'hg_flu_jhmmer.sto'), 
                           'a2m': os.path.join(data_path, 'hg_flu', 'hg_flu_jhmmer.a2m'), 
                           'params': os.path.join(data_path, 'hg_flu', 'hg_flu_plmc.params')}
    data_dict['hsp82_yeast'] = {'fasta': os.path.join(data_path, 'hsp82_yeast', 'hsp82_yeast.fasta'), 
                                'start_pos': 2,
                                'sto': os.path.join(data_path, 'hsp82_yeast', 'hsp82_yeast_jhmmer.sto'), 
                                'a2m': os.path.join(data_path, 'hsp82_yeast', 'hsp82_yeast_jhmmer.a2m'), 
                                'params': os.path.join(data_path, 'hsp82_yeast', 'hsp82_yeast_plmc.params')}
    
    data_dict['kka2_klepn'] = {'fasta': os.path.join(data_path, 'kka2_klepn', 'kka2_klepn.fasta'), 
                               'start_pos': 1,
                               'sto': os.path.join(data_path, 'kka2_klepn', 'kka2_klepn_jhmmer.sto'), 
                               'a2m': os.path.join(data_path, 'kka2_klepnt', 'kka2_klepnt_jhmmer.a2m'), 
                               'params': os.path.join(data_path, 'kka2_klepnt', 'kka2_klepnt_plmc.params')}
    
    data_dict['mth3_haeaestabilized'] = {'fasta': os.path.join(data_path, 'mth3_haeaestabilized', 'mth3.fasta'), 
                                         'start_pos': 2,
                                         'sto': os.path.join(data_path, 'mth3_haeaestabilized', 'mth3_haeaestabilized_jhmmer.sto'), 
                                         'a2m': os.path.join(data_path, 'mth3_haeaestabilized', 'mth3_haeaestabilized_jhmmer.a2m'), 
                                         'params': os.path.join(data_path, 'mth3_haeaestabilized', 'mth3_haeaestabilized_plmc.params')}
                                        
    data_dict['pabp_yeast'] = {'fasta': os.path.join(data_path, 'pabp_yeast', 'pabp_yeast.fasta'), 
                                'start_pos': 126,
                                'sto': os.path.join(data_path, 'pabp_yeast', 'pabp_yeast_jhmmer.sto'), 
                                'a2m': os.path.join(data_path, 'pabp_yeast', 'pabp_yeast_jhmmer.a2m'), 
                                'params': os.path.join(data_path, 'pabp_yeast', 'pabp_yeast_plmc.params')}
    data_dict['polg_hcvjf'] = {'fasta': os.path.join(data_path, 'polg_hcvjf', 'polg_hcvjf.fasta'), 
                                'start_pos': 1994,
                                'sto': os.path.join(data_path, 'polg_hcvjf', 'polg_hcvjf_jhmmer.sto'), 
                                'a2m': os.path.join(data_path, 'polg_hcvjf', 'polg_hcvjf_jhmmer.a2m'), 
                                'params': os.path.join(data_path, 'polg_hcvjf', 'polg_hcvjf_plmc.params')}
    data_dict['rl401_yeast'] = {'fasta': os.path.join(data_path, 'rl401_yeast', 'rl401.fasta'), 
                                'start_pos': 2,
                                'sto': os.path.join(data_path, 'rl401_yeast', 'rl401_yeast_jhmmer.sto'), 
                                'a2m': os.path.join(data_path, 'rl401_yeast', 'rl401_yeast_jhmmer.a2m'), 
                                'params': os.path.join(data_path, 'rl401_yeast', 'rl401_yeast_plmc.params')}
    data_dict['ube4b_mouse'] = {'fasta': os.path.join(data_path, 'ube4b_mouse', 'ueb_mouse.fasta'), 
                                'start_pos': 1072,
                                'sto': os.path.join(data_path, 'ube4b_mouse', 'ube4b_mouse_jhmmer.sto'), 
                                'a2m': os.path.join(data_path, 'ube4b_mouse', 'ube4b_mouse_jhmmer.a2m'), 
                                'params': os.path.join(data_path, 'ube4b_mouse', 'ube4b_mouse_plmc.params')}
    data_dict['yap1_human'] = {'fasta': os.path.join(data_path, 'yap1_human', 'yap1_human.fasta'), 
                                'start_pos': 170,
                                'sto': os.path.join(data_path, 'yap1_human', 'yap1_human_jhmmer.sto'), 
                                'a2m': os.path.join(data_path, 'yap1_human', 'yap1_human_jhmmer.a2m'),
                                'params': os.path.join(data_path, 'yap1_human', 'yap1_human_plmc.params')}
    """
    
    run_jackhmmer(data_dict)