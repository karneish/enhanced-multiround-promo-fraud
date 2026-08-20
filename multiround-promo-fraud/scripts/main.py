import argparse
import sys
import os
sys.path.append('../src')

import datetime
import itertools
import re
import gc
import json

import torch
import dgl
import pandas as pd

from time import time
from experiment.supervised_multi import MultiroundExperiment
from utils.utils_const import DEFAULT_MAIN_CONFIG, DEFAULT_TRAIN_CONFIG, DEFAULT_ADVER_CONFIG, DEFAULT_MODEL_CONFIG,  DEFAULT_STRAT_CONFIG
from utils.utils_const import LOSS_DICT, BACKBONE_DICT

main_config = DEFAULT_MAIN_CONFIG.copy()
train_config = DEFAULT_TRAIN_CONFIG.copy()
model_config = DEFAULT_MODEL_CONFIG.copy()
strat_config = DEFAULT_STRAT_CONFIG.copy()
adver_config = DEFAULT_ADVER_CONFIG.copy()

def main(cname):
    # READ CONFIG
    with open(f'{cname}.json')  as f:
      config_file = json.loads(f.read())

    print(config_file)
    TRIAL_NUM = config_file['TRIAL_NUM']
    FAILURE_LIMIT = config_file['FAILURE_LIMIT']

    EXPERIMENT_DESC = config_file['EXPERIMENT_DESC']
    LIST_DSET = config_file['LIST_DSET']
    LIST_TRAIN_DSET = config_file['LIST_TRAIN_DSET']
    EXP_DICT = config_file['EXP_DICT']

    outer_dfs = []
    ts = datetime.datetime.now().strftime("%y%m%d%H%M%S")

    # Write txt log
    os.makedirs(f'../result/{cname}/{ts}', exist_ok=True)

    f = open(f'../result/{cname}/{ts}/meta.txt', "a")
    f.write(f'{EXPERIMENT_DESC}\n\n')
    f.write(f'MAIN CONFIG\n{str(main_config)}\n\n')
    f.write(f'TRAIN CONFIG\n{str(train_config)}\n\n')
    f.write(f'MODEL CONFIG\n{str(model_config)}\n\n')
    f.write(f'STRAT CONFIG\n{str(strat_config)}\n\n')
    f.write(f'ADVER CONFIG\n{str(adver_config)}\n\n')
    f.write(f'EXP DICT\n{str(EXP_DICT)}\n\n')
  
    f.close()

    ### DATASET STUFF
    for idx, dset in enumerate(LIST_DSET):
      dataset, _ = dgl.load_graphs(f'../dataset/{dset}')
      graph = dataset[0].long()
    
      if len(graph.ndata['label'].shape) > 1:
          graph.ndata['label'] = graph.ndata['label'].argmax(1)
          graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
          
      graph.ndata['feature'] = graph.ndata['feature'].float()
      train_config['dset_name'] = dset
    
      if LIST_TRAIN_DSET[idx] != 'NONE':
        dataset, _ = dgl.load_graphs(f'../dataset/{LIST_TRAIN_DSET[idx]}')
        train_graph = dataset[0].long()
    
        if len(train_graph.ndata['label'].shape) > 1:
            train_graph.ndata['label'] = train_graph.ndata['label'].argmax(1)
            train_graph.ndata['label'] = train_graph.ndata['label'].long().squeeze(-1)
        train_graph.ndata['feature'] = train_graph.ndata['feature'].float()
      else:
        train_graph = None
    
      ### ADJUST BUDGETS
      pos = (graph.ndata['label'] == 1).sum().item()
      neg = (graph.ndata['label'] == 0).sum().item()
    
      main_config['round_new_pos'] = int(0.05 * pos)
      main_config['round_new_neg'] = int(0.05 * neg)
      main_config['round_budget_pos'] = 0
      main_config['round_budget_neg'] = 0
    
      EXP_DICT['round_budget'] = [0]
      EXP_DICT['augment_pos_ratio'] = [pos / (pos + neg) * 0.125]
      EXP_DICT['augment_neg_ratio'] = [neg / (pos + neg) * 0.125]
    
      for key in ['num_epoch', 'num_round_epoch', 'early_stopping']:
        model_config[key] = train_config[key]
    
      ### MAIN LOOP
      keywords = EXP_DICT.keys()
      combinations = list(itertools.product(*EXP_DICT.values()))
    
      for combi in combinations:
        for key, val in (zip(keywords, combi)):
          # Special cases
          if key == 'round_budget':
            main_config['round_budget_pos'] = int(val * pos)
            main_config['round_budget_neg'] = int(val * neg)
          elif key == 'loss':
            train_config['loss'] = LOSS_DICT[val]
          elif key == 'boost_agg_backbone_name':
            model_config['boost_agg_backbone'] = BACKBONE_DICT[val]
          elif key == 'round_window':
            model_config['round_window'] = val
            strat_config['augment_round_split'] = val + 1
          elif key == 'dropedge_prob':
            strat_config['dropedge_prob'] = val
          
          # Else just copy
          else:
            for cnfg in [main_config, strat_config, train_config, model_config, adver_config]:
                if key in cnfg.keys():
                    cnfg[key] = val
    
            # Other hardcoded settings
          model_config['mlp_feats'] = model_config['h_feats']
    
        # Counter and container   
        dfs, exp, dataset, graph = [], [], [], []
        trial_counter, failure_counter = 0, 0
    
        start = time()
        while trial_counter < TRIAL_NUM:
          print(f"================")
          print(f"++++++++++++++++")
          print(f"TRIAL NUMBER {trial_counter}")
          print(f"++++++++++++++++")
          print(f"================")
          print(f'>> Mem status: {torch.cuda.mem_get_info() if torch.cuda.is_available() else "CPU"}')
            
          # REREAD GRAPH DATA
          print(f"  > Rereading graph data...")
          dataset, _ = dgl.load_graphs(f'../dataset/{dset}')
          graph = dataset[0].long()
    
          if len(graph.ndata['label'].shape) > 1:
              graph.ndata['label'] = graph.ndata['label'].argmax(1)
              graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)
          graph.ndata['feature'] = graph.ndata['feature'].float()
          train_config['dset_name'] = dset
    
          if LIST_TRAIN_DSET[idx] != 'NONE':
            dataset, _ = dgl.load_graphs(f'../dataset/{LIST_TRAIN_DSET[idx]}')
            train_graph = dataset[0].long()
    
            if len(train_graph.ndata['label'].shape) > 1:
                train_graph.ndata['label'] = train_graph.ndata['label'].argmax(1)
                train_graph.ndata['label'] = train_graph.ndata['label'].long().squeeze(-1)
            train_graph.ndata['feature'] = train_graph.ndata['feature'].float()
          else:
            train_graph = None
    
          print(f"  > Initializing multiround object...")
          exp = MultiroundExperiment(
            graph, train_graph=train_graph,
            main_config=main_config, model_config=model_config, strat_config=strat_config, 
            adver_config=adver_config, train_config=train_config
          )
          
          # Adversarial Round
          round_counter = 0
          round_flag = True
          while (round_counter < main_config['round_num']) and (round_flag):
            print(f"  > Starting round {round_counter}...")
            round_flag = exp.one_round_node(round_counter)
            round_counter = round_counter + 1
    
          # Check if round successful or need hard reset
          if round_flag:
            eval_df = pd.DataFrame(sum([r['log_single_eval'] for r in exp.rounds], []), columns=['round', 'eval_type', 'time', 'rec', 'prec', 'f1', 'auc', 'tp', 'fp', 'tn', 'fn']) # Evaluation log
            trainlog_df = pd.DataFrame([r['log_round'] for r in exp.rounds]) # Round training log
            log_df = pd.merge(left=eval_df, right=trainlog_df, on='round')
            
            # Other global log            
            log_df['trial'] = trial_counter
            log_df['num_nodes'] = graph.num_nodes()
            log_df['num_edges'] = graph.num_edges()
            
            dfs.append(log_df)
            trial_counter = trial_counter + 1
          else:
            failure_counter = failure_counter + 1
    
          if failure_counter > FAILURE_LIMIT:
            raise Exception('Too many failed experiments!')
          
          # Remove checkpoints
          exp.clean_temp_files()

          # Save embedding before cleanup if needed
          if main_config['save_embedding']:
            exp.model.set_graph(exp.dset['graph'], round_num=(main_config['round_num']-1), device=main_config['device'])
            embedding = exp.model.embed_nodes(exp.model.graph, exp.model.graph.ndata['feature'])
            os.makedirs('../result/Z-temp', exist_ok=True)
            torch.save(torch.linalg.vector_norm(embedding[:,int(embedding.shape[1] / 2):], ord=2, dim=1), f'../result/Z-temp/{ts}-{dset}-{suffix}-TEMP.pt')
            torch.save(torch.linalg.vector_norm(embedding[:,:int(embedding.shape[1] / 2)], ord=2, dim=1), f'../result/Z-temp/{ts}-{dset}-{suffix}-NONTEMP.pt')

          print(f'>> Freeing memory')
          # Remove other stuffs
          del dataset 
          del graph
          del train_graph
          del exp
          del _
          gc.collect()
          torch.cuda.empty_cache() 
          print(f'>> Freeing memory: {torch.cuda.mem_get_info() if torch.cuda.is_available() else "CPU"}')
    
        print(f'Experiment ended, experienced {failure_counter} failures')
        print(f'Elapsed experiment time {time() - start:.8f}s')
    
        # Save artifacts per config setting
        if main_config['save_df']:
          final_df = pd.concat(dfs)
          
          for cnfg in [main_config, strat_config, model_config, train_config, adver_config]:
            for key, value in cnfg.items():
                if key not in ['verbose']:
                    final_df[key] = str(value)
    
          final_df['timestamp'] = ts
          
          stripped = [re.sub(r"\W+", "", str(val))[:6] for val in combi]
          suffix = '-'.join(stripped)
          os.makedirs(f'../result/{cname}/{ts}', exist_ok=True)
          final_df.to_csv(f'../result/{cname}/{ts}/{dset}-{suffix}-E.csv')
          outer_dfs.append(final_df)
    
    # Save overall artifacts
    if main_config['save_df']:
      final_outer_df = pd.concat(outer_dfs)
      os.makedirs(f'../result/{cname}/{ts}', exist_ok=True)
      final_outer_df.to_csv(f'../result/{cname}/{ts}/combined_result.csv')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cname", help="Name of the json config file of the experiments")
    args = parser.parse_args()
    
    main(args.cname)