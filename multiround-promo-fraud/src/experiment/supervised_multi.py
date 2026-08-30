import time
import os
import copy
import dgl
import torch
import gc
import numpy as np

from collections import Counter
from numpy import random
from sklearn.model_selection import train_test_split

from adversary.choose.simple_choose import RandomChoose

from utils.utils_func import add_generated_nodes, remove_generated_nodes, subgraph_from_nodes, verPrint, get_best_f1, eval_and_print
from utils.utils_const import MODEL_DICT, AUGMENT_DICT, ADVER_CHOOSE_DICT, ADVER_MOD_DICT, ADDON_DICT
from utils.utils_const import DEFAULT_MAIN_CONFIG, DEFAULT_MODEL_CONFIG, DEFAULT_STRAT_CONFIG, DEFAULT_TRAIN_CONFIG, DEFAULT_ADVER_CONFIG
from utils.utils_const import TEMP_MODEL_SAVE_PATH

class MultiroundExperiment(object):
    #######################
    ### INITIALIZATIONS ###
    #######################

    # Class init
    def __init__(
            self, graph, train_graph=None,
            main_config=DEFAULT_MAIN_CONFIG,
            model_config=DEFAULT_MODEL_CONFIG,
            strat_config=DEFAULT_STRAT_CONFIG,
            adver_config=DEFAULT_ADVER_CONFIG, 
            train_config=DEFAULT_TRAIN_CONFIG
        ):
        
        # Copy some values in main config to all other configs
        for att in ['device']:
            for conf in [main_config, model_config, strat_config, adver_config, train_config]:
                conf[att] = main_config[att]
        model_config['etypes'] = ['none'] if graph.is_homogeneous else graph.etypes
        
        # Model checkpointing
        self.temp_model_path = f'{TEMP_MODEL_SAVE_PATH}_{time.time()}'

        # Initialize configs
        self.main_config = main_config
        self.model_config = model_config
        self.strat_config = strat_config
        self.adver_config = adver_config
        self.train_config = train_config
        self.verbose = main_config['verbose']

        # Initialize common working variables
        self.working_vars = {}
        self.current_round = 0

        verPrint(self.verbose, 2, f'''
=========================
STARTING {self.main_config['exp_type']} EXPERIMENT
=========================
MAIN CONFIG: {sorted(self.main_config.items())}
MODEL CONFIG: {sorted(self.model_config.items())}
STRAT CONFIG: {sorted(self.strat_config.items())}
TRAIN CONFIG: {sorted(self.train_config.items())}
ADVER CONFIG: {sorted(self.adver_config.items())}

INITIAL GRAPH: {graph.num_nodes()} {dict(Counter(graph.ndata['label'].tolist()))} Nodes, {graph.num_edges()} Edges
=========================
        ''')

        verPrint(self.verbose, 2, f'''>> Initializing graph...''')
        for g, txt in [(graph, 'MAIN GRAPH'), (train_graph, 'TRAINING GRAPH')]:
            if g != None: verPrint(self.verbose, 2, f'''{txt}: {g.num_nodes()} {dict(Counter(g.ndata['label'].tolist()))} Nodes, {g.num_edges()} Edges\n''')
            self.init_graph(g)

        verPrint(self.verbose, 2, f'''>> Initializing experiment type-specific modules:''')
        if self.main_config['exp_type'] == 'ADVER':
            self.dset = { 'graph': graph, 'train_graph':train_graph } # Init dataset
            self.rounds = [{'budgets_pos': torch.tensor([]), 'budgets_neg': torch.tensor([]), 'log_single_eval': [], 'log_round': {}, 'seeds_pos': [], 'seeds_neg': []} for i in range(self.main_config['round_num'])]
            
            verPrint(self.verbose, 2, f'''  >> Initializing adversarial modules...''')
            self.init_adversarial()
            self.neg_sampler = RandomChoose(replace=False, verbose=self.verbose) # Negative sampler

            # ASSIGN FUNCTIONS
            self.assign_train_graph = self.assign_train_graph_adver
            self.assign_eval_graph = self.assign_eval_graph_adver
            self.final_eval = self.final_eval_adver
            self.close_round = self.close_round_adver

        elif self.main_config['exp_type'] == 'ADVOR':
            self.dset = { 'graph': graph, 'train_graph':train_graph } # Init dataset
            self.rounds = [{'budgets_pos': torch.tensor([]), 'budgets_neg': torch.tensor([]), 'log_single_eval': [], 'log_round': {}, 'seeds_pos': [], 'seeds_neg': []} for i in range(self.main_config['round_num'])]
            
            verPrint(self.verbose, 2, f'''  >> Initializing adversarial modules...''')
            self.init_adversarial()
            self.neg_sampler = RandomChoose(replace=False, verbose=self.verbose) # Negative sampler

            # ASSIGN FUNCTIONS
            self.assign_train_graph = self.assign_train_graph_oracle
            self.assign_eval_graph = self.assign_eval_graph_oracle
            self.final_eval = self.final_eval_adver
            self.close_round = self.close_round_adver

        verPrint(self.verbose, 2, f'''>> Initializing task-specific modules:''')
        if self.main_config['task_type'] == 'NODE':
            pass # TODO
        elif self.main_config['task_type'] == 'GRAPH':
            pass # TODO: If the task type is graph detection, conversion should happen just before the training process?
            
        verPrint(self.verbose, 2, f'''>> Initializing classifier model...''')
        self.init_detector()

        verPrint(self.verbose, 2, f'''>> Initializing other strategies...''')
        self.init_strat()

        verPrint(self.verbose, 2, f'''
FINISH INITIALIZATION
=====================
        ''')

    # Initialize graph to have all the necessary ndatas
    def init_graph(self, graph):
        if graph != None:
            # Status for nodes
            graph.ndata['creation_round'] = torch.full([graph.num_nodes()], -100, dtype=torch.long) # Round where the node is first observable
            
            graph.ndata['predicted'] = torch.full([graph.num_nodes()], True, dtype=torch.bool) # Current prediction status of the node
            graph.ndata['true_predicted_round'] = torch.full([graph.num_nodes()], 100, dtype=torch.long) # Current prediction status of the node

            # Masks
            graph.ndata['train_mask'] = torch.zeros([graph.num_nodes()]).bool()
            graph.ndata['val_mask'] = torch.zeros([graph.num_nodes()]).bool()
            graph.ndata['test_mask'] = torch.zeros([graph.num_nodes()]).bool()

    # Initialize model
    def init_detector(self):
        # MODEL                        
        in_feats = self.dset['graph'].ndata['feature'].shape[1]
        num_classes = self.dset['graph'].ndata['label'].unique(return_counts=True)[0].shape[0]
        self.model = MODEL_DICT[self.model_config['model_name']](**(self.model_config | {'in_feats': in_feats, 'num_classes': num_classes, 'temp_model_path': self.temp_model_path}))

        # ADDONS
        self.addon_predictor = {
            'function': ADDON_DICT[self.model_config['addon_name']],
            'state': {}
        }
    
    # Initialize other algorithm/strategies like graph augmentation or pseudolabeling
    def init_strat(self):
        augment_keys = [key for key in self.strat_config.keys() if 'augment_' in key]
        self.augment = {
            'function': AUGMENT_DICT[self.strat_config['augment_name']],
            'params': { key:self.strat_config[key] for key in augment_keys }
        }

    # Initialize adversarial
    def init_adversarial(self):
        self.adver_choose = ADVER_CHOOSE_DICT[self.adver_config['adver_choose_name']](**self.adver_config)
        self.adver_mod = ADVER_MOD_DICT[self.adver_config['adver_mod_name']](**self.adver_config)
    
    ##############################
    ### HIGHEST LEVEL FUNCTIONS ###
    ##############################

    # Execute 1 round of node classification
    def one_round_node(self, round_num):
        verPrint(self.verbose, 2, f'''
+++++++++++++++++
STARTING ROUND {round_num}!
+++++++++++++++++
        ''')
    
        self.init_round(round_num)

        self.assign_train_graph(round_num)
        verPrint(self.verbose, 2, f'''ASSIGNED GRAPH FOR TRAINING: {self.dset['graph'].num_nodes()} {dict(Counter(self.dset['graph'].ndata['label'].tolist()))} Nodes, {self.dset['graph'].num_edges()} Edges\n''')
       
        if (len(self.train_config['round_train_list']) == 0) or (round_num in self.train_config['round_train_list']):
            self.model_round_train(round_num)

        # Log training data for this round
        self.rounds[round_num]['log_round'] = self.rounds[round_num]['log_round'] | {'round': f'round_{round_num}'} | self.model.get_latest_trainlog()

        self.assign_eval_graph(round_num)
        verPrint(self.verbose, 2, f'''\nASSIGNED GRAPH FOR EVALUATION: {self.dset['graph'].num_nodes()} {dict(Counter(self.dset['graph'].ndata['label'].tolist()))} Nodes, {self.dset['graph'].num_edges()} Edges''')
        
        # Check max retry, if not evaluate
        if self.working_vars['train_retry'] < self.train_config['train_max_retry']:

            # Predict
            self.rounds[round_num]['preds'], self.rounds[round_num]['probs'], self.rounds[round_num]['checks'] = self.model_round_predict(round_num)
            self.rounds[round_num]['label'] = self.dset['graph'].ndata['label'].bool()

            # Generate masks as needed
            true_predicted_mask = (self.dset['graph'].ndata['true_predicted_round'] == 100)
            
            tp_mask = torch.zeros([self.dset['graph'].num_nodes()]).bool()
            fp_mask = torch.zeros([self.dset['graph'].num_nodes()]).bool()
            tn_mask = torch.zeros([self.dset['graph'].num_nodes()]).bool()

            tp_mask[self.rounds[round_num]['checks'][0]] = True
            fp_mask[self.rounds[round_num]['checks'][1]] = True
            tn_mask[self.rounds[round_num]['checks'][2]] = True

            # Update prediction history, this is status based on TP and FP, used to track additional label for next round
            self.dset['graph'].ndata['predicted'][tp_mask] = True
            self.dset['graph'].ndata['predicted'][fp_mask] = True

            #self.dset['graph'].ndata['predicted'][self.rounds[round_num]['checks'][0]] = True
            #self.dset['graph'].ndata['predicted'][self.rounds[round_num]['checks'][1]] = True

            # Update prediction history, this is TRUE prediction status based on TP and TN, used for evaluating true prediction speed
            self.dset['graph'].ndata['true_predicted_round'][tp_mask & true_predicted_mask] = round_num
            self.dset['graph'].ndata['true_predicted_round'][tn_mask & true_predicted_mask] = round_num

            # Evaluate for logging
            self.one_round_node_eval(round_num)
            self.final_eval(round_num)
            self.close_round(round_num)

            return True # Round succeeded
        else:
            return False # Round failed, need to restart experiment
    
    ###############################
    ### GENERIC ROUND FUNCTIONS ### '
    ###############################

    def init_round(self, round_num):
        self.current_round = round_num

    def one_round_node_eval(self, round_num):
        labels = self.dset['graph'].ndata['label']


        verPrint(self.verbose, 2, '''
-------------------
STARTING EVALUATION
-------------------
''')
        
        _p, _cm = eval_and_print(self.verbose, labels, self.rounds[round_num]['preds'], self.rounds[round_num]['probs'], 'Dataset - Overall')
        self.rounds[round_num]['log_single_eval'].append((f'round_{round_num}', 'entire_graph', 0) + _p + _cm)

        train_mask = self.dset['graph'].ndata['train_mask']
        _p, _cm = eval_and_print(self.verbose, labels[train_mask], self.rounds[round_num]['preds'][train_mask], self.rounds[round_num]['probs'][train_mask], 'Dataset - Train')
        self.rounds[round_num]['log_single_eval'].append((f'round_{round_num}', 'train_set', 0) + _p + _cm)

        val_mask = self.dset['graph'].ndata['val_mask']
        _p, _cm = eval_and_print(self.verbose, labels[val_mask], self.rounds[round_num]['preds'][val_mask], self.rounds[round_num]['probs'][val_mask], 'Dataset - Val')
        self.rounds[round_num]['log_single_eval'].append((f'round_{round_num}', 'val_set', 0) + _p + _cm)

        test_mask = self.dset['graph'].ndata['test_mask']
        _p, _cm = eval_and_print(self.verbose, labels[test_mask], self.rounds[round_num]['preds'][test_mask], self.rounds[round_num]['probs'][test_mask], 'Dataset - Test')
        self.rounds[round_num]['log_single_eval'].append((f'round_{round_num}', 'test_set', 0) + _p + _cm)
        
        pstatus = [f'{str(p[0])}-{str(p[1])}' for p in list(zip(labels.tolist(), self.dset['graph'].ndata['predicted'].tolist()))]
        verPrint(self.verbose, 1, f'PREDICTION STATUS - {dict(Counter(pstatus))}')

        # Coverage
        prio_pool = ((self.dset['graph'].ndata['predicted'] == False) & (self.dset['graph'].ndata['label'] == 1)).nonzero().flatten()
        og_prio_pool = ((self.dset['graph'].ndata['predicted'] == False) & (self.dset['graph'].ndata['creation_round'] < 1) & (self.dset['graph'].ndata['label'] == 1)).nonzero().flatten()
        
        # Prediction speed
        speed_mask = (self.dset['graph'].ndata['true_predicted_round'] < 100) & (self.dset['graph'].ndata['label'] == 1)
        ceil_creation_round = (torch.maximum(torch.full_like(labels, 0), self.dset['graph'].ndata['creation_round']))
        pred_speed = (self.dset['graph'].ndata['true_predicted_round'] - ceil_creation_round)[speed_mask].float().mean().item()

        self.rounds[round_num]['log_round'] = self.rounds[round_num]['log_round'] | {'predicted_pos': len(prio_pool), 'predicted_pos_og': len(og_prio_pool), 'prediction_speed': pred_speed}

        verPrint(self.verbose, 2, f'    >> {len(prio_pool)} positive nodes left unpredicted...')
        verPrint(self.verbose, 2, f'    >> {len(og_prio_pool)} OG positive nodes left unpredicted...')

    ###########################################
    ### ADVERSARIAL SETTING ROUND FUNCTIONS ###
    ###########################################

    def assign_train_graph_adver(self, round_num):
        if round_num == 0:
            if self.dset['train_graph'] != None: # Precomputed train/test graph
                r0_test_idx = torch.arange(self.dset['train_graph'].num_nodes(), self.dset['graph'].num_nodes(), dtype=torch.long) # Get index
                train_graph = self.dset['train_graph'] # Get train graph
            else: # Randomized train/test graph
                r0_num = round(self.dset['graph'].num_nodes() * self.train_config['ratio_initial'])
                r0_test_idx = torch.tensor(random.choice(list(range(self.dset['graph'].num_nodes())), r0_num, replace=False), dtype=torch.long)
                
                # Create graph and get index
                train_graph, r0_test_idx = remove_generated_nodes(self.dset['graph'], r0_test_idx)

            # Swap            
            self.dset['temp_graph'] = self.dset['graph']
            self.dset['graph'] = train_graph

            full_mask = torch.ones([self.dset['temp_graph'].num_nodes()]).bool()
            full_mask[r0_test_idx] = False
            
            self.working_vars['r0_test_idx'] = r0_test_idx
            self.working_vars['r0_train_idx'] = full_mask.nonzero()
        else:
            # Cleanup temp graph to squeeze out memory
            if 'temp_graph' in self.dset.keys():
                del self.dset['temp_graph']
                gc.collect()
            
            pass # Graph used to evaluate last round immediately becomes training graph this round
    
    def split_train_test_adver(self, round):
        verPrint(self.verbose, 2, f"Alotting train-val-test split for round {round}")

        # Actual function code
        labels = self.dset['graph'].ndata['label']
        self.dset['graph'].ndata['train_mask'] = torch.zeros([len(labels)]).bool()
        self.dset['graph'].ndata['val_mask'] = torch.zeros([len(labels)]).bool()
        self.dset['graph'].ndata['test_mask'] = torch.zeros([len(labels)]).bool()

        # Build pool of training data
        if round == 0: 
            # For first round, just use the entire graph for training
            initial_pool, prediction_pool, budget_pool = [], [], []
            full_pool = torch.arange(len(labels), dtype=torch.long)
        else:
            # Otherwise, determine the rounds from which to generate the training data from
            init_round = 0 
            source_rounds = list(range(round))
         
            # Exctracting the data based on the pool
            initial_pool = (self.dset['graph'].ndata['creation_round'] < init_round).nonzero().flatten()
            prediction_pool = torch.cat([torch.cat(self.rounds[i]['checks'][:2], 0) for i in source_rounds])
            budget_pool = torch.cat([torch.cat([self.rounds[i]['budgets_pos'], self.rounds[i]['budgets_neg']], 0) for i in source_rounds])
            full_pool = torch.cat([initial_pool.to('cpu'), prediction_pool.to('cpu'), budget_pool], 0).unique().long()

        verPrint(self.verbose, 3, f"Initial pool: {len(initial_pool)}, Prediction pool: {len(prediction_pool)}, Budget pool: {len(budget_pool)}, Full pool: {len(full_pool)}")

        index = torch.arange(len(labels), dtype=torch.long)[full_pool]        
        nonindex = torch.ones_like(labels, dtype=bool)
        nonindex[full_pool] = False

        # Split after sanity check
        if (torch.sum(labels[index] == 0) < 2) or (torch.sum(labels[index] == 1) < 2):
            return None, None, None, None
        
        idx_train, idx_valid, y_train, y_valid = train_test_split(
            index, labels[index], stratify=labels[index],
            train_size = self.train_config['ratio_train'], random_state = self.train_config['random_state'], shuffle=True
        )
        
        # Set result in graph data
        self.dset['graph'].ndata['train_mask'][idx_train] = 1
        self.dset['graph'].ndata['val_mask'][idx_valid] = 1
        self.dset['graph'].ndata['test_mask'][nonindex] = 1
 
        verPrint(self.verbose, 3, f"Full graph size: {self.dset['graph'].num_nodes()}, Training: {len(idx_train)}, Val: {len(idx_valid)}, Test: {len(nonindex)}")        
        return idx_train, idx_valid, y_train, y_valid

    def assign_eval_graph_adver(self, round_num):
        if round_num == 0:
            # Swap back to original graph
            self.dset['temp_graph'].ndata['train_mask'][self.working_vars['r0_train_idx']] = self.dset['graph'].ndata['train_mask'][:len(self.working_vars['r0_train_idx'])].unsqueeze(1)
            self.dset['temp_graph'].ndata['val_mask'][self.working_vars['r0_train_idx']] = self.dset['graph'].ndata['val_mask'][:len(self.working_vars['r0_train_idx'])].unsqueeze(1)
            self.dset['temp_graph'].ndata['test_mask'][self.working_vars['r0_train_idx']] = self.dset['graph'].ndata['test_mask'][:len(self.working_vars['r0_train_idx'])].unsqueeze(1)

            self.dset['graph'] = self.dset['temp_graph']

            self.dset['graph'].ndata['creation_round'][self.working_vars['r0_test_idx']] = 0
            self.dset['graph'].ndata['test_mask'][self.working_vars['r0_test_idx']] = True
            self.dset['graph'].ndata['predicted'][self.working_vars['r0_test_idx']] = False

            adv_seed, adv_new_ids = [], []
            neg_seed, neg_new_ids = [], []
        else:
            verPrint(self.verbose, 2, f'Generating additional positive data from adversary...')
            new_adv_nodes, new_adv_edges, adv_seed, adv_new_ids = self.adversary_round_generate()
            add_generated_nodes(self.dset['graph'], (new_adv_nodes, new_adv_edges), self.current_round)
            
            # Negatives
            verPrint(self.verbose, 2, f'Generating additional negative data by duplicating random nodes...')
            new_neg_nodes, new_neg_edges, neg_seed, neg_new_ids = self.round_generate_negatives()
            add_generated_nodes(self.dset['graph'], (new_neg_nodes, new_neg_edges), self.current_round)

        self.rounds[round_num]['seeds_pos'] = (adv_seed, adv_new_ids)
        self.rounds[round_num]['seeds_neg'] = (neg_seed, neg_new_ids)

    def close_round_adver(self, round_num):
        # Get budget for next round
        verPrint(self.verbose, 2, f'Selecting data as budgeted ground truth for next round...')
        round_pos_budgets, round_neg_budgets = self.get_round_budget(round_num)
        self.rounds[round_num]['budgets_pos'] = round_pos_budgets
        self.rounds[round_num]['budgets_neg'] = round_neg_budgets

    def final_eval_adver(self, round_num):
        labels = self.dset['graph'].ndata['label']

        if (round_num > 0) and (self.verbose >= 2):
            
            verPrint(self.verbose, 2, '-----------------------\nPREDICTION RESULT - ROUNDS')
            for i in range(round_num + 1):
                i_round_mask = (self.dset['graph'].ndata['creation_round'] == i).nonzero().flatten()
                _p, _cm = eval_and_print(self.verbose, labels[i_round_mask], self.rounds[round_num]['preds'][i_round_mask], self.rounds[round_num]['probs'][i_round_mask], f'Dataset - Round {i}')
                self.rounds[round_num]['log_single_eval'].append((f'round_{round_num}', f'round_{i}_nodes', 0) + _p + _cm )

            verPrint(self.verbose, 2, '-----------------------\nPREDICTION RESULT - SEEDS')

            adv_seed = self.rounds[round_num]['seeds_pos'][0]
            neg_seed = self.rounds[round_num]['seeds_neg'][0]

            _p, _cm = eval_and_print(self.verbose, labels[torch.cat([adv_seed, neg_seed], 0)], self.rounds[round_num]['preds'][torch.cat([adv_seed, neg_seed], 0)], self.rounds[round_num]['probs'][torch.cat([adv_seed, neg_seed], 0)], 'Seeds - Current')
            self.rounds[round_num]['log_single_eval'].append((f'round_{round_num}', f'seed_current_pred', 0) + _p + _cm)

            _p, _cm = eval_and_print(self.verbose, labels[torch.cat([adv_seed, neg_seed], 0)], self.rounds[round_num-1]['preds'][torch.cat([adv_seed, neg_seed], 0)], self.rounds[round_num]['probs'][torch.cat([adv_seed, neg_seed], 0)], 'Seeds - Prev')
            self.rounds[round_num]['log_single_eval'].append((f'round_{round_num}', f'seed_prev_pred', 0) + _p + _cm)

    ##################################################
    ### ORACLE ADVERSARIAL SETTING ROUND FUNCTIONS ###
    ##################################################

    def assign_train_graph_oracle(self, round_num):
        if round_num == 0:
            if self.dset['train_graph'] != None: # Precomputed train/test graph
                r0_test_idx = torch.arange(self.dset['train_graph'].num_nodes(), self.dset['graph'].num_nodes(), dtype=torch.long) # Get index
                train_graph = self.dset['train_graph'] # Get train graph
            else: # Randomized train/test graph
                r0_num = round(self.dset['graph'].num_nodes() * self.train_config['ratio_initial'])
                r0_test_idx = torch.tensor(random.choice(list(range(self.dset['graph'].num_nodes())), r0_num, replace=False), dtype=torch.long)
                
            # DIFFERENCE IS HERE, TEST SPLIT REMAINS THE SAME BUT ENTIRE GRAPH USED FOR TRAINING
            train_graph = copy.deepcopy(self.dset['graph'])
            
            # Swap     
            self.dset['temp_graph'] = self.dset['graph']
            self.dset['graph'] = train_graph

            full_mask = torch.ones([self.dset['temp_graph'].num_nodes()]).bool()
            full_mask[r0_test_idx] = False
            
            self.working_vars['r0_test_idx'] = r0_test_idx
            self.working_vars['r0_train_idx'] = full_mask.nonzero()

            adv_seed, adv_new_ids = [], []
            neg_seed, neg_new_ids = [], []
        else:
            # IF NOT GENERATE NEW DATA
            verPrint(self.verbose, 2, f'Generating additional positive data from adversary...')
            new_adv_nodes, new_adv_edges, adv_seed, adv_new_ids = self.adversary_round_generate()
            add_generated_nodes(self.dset['graph'], (new_adv_nodes, new_adv_edges), self.current_round)
            
            # Negatives
            verPrint(self.verbose, 2, f'Generating additional negative data by duplicating random nodes...')
            new_neg_nodes, new_neg_edges, neg_seed, neg_new_ids = self.round_generate_negatives()
            add_generated_nodes(self.dset['graph'], (new_neg_nodes, new_neg_edges), self.current_round)

        # Log seeds
        self.rounds[round_num]['seeds_pos'] = (adv_seed, adv_new_ids)
        self.rounds[round_num]['seeds_neg'] = (neg_seed, neg_new_ids)
    
    def split_train_test_oracle(self, round):
        verPrint(self.verbose, 2, f"Alotting train-val-test split for round {round}")

        # Actual function code
        labels = self.dset['graph'].ndata['label']
        self.dset['graph'].ndata['train_mask'] = torch.zeros([len(labels)]).bool()
        self.dset['graph'].ndata['val_mask'] = torch.zeros([len(labels)]).bool()
        self.dset['graph'].ndata['test_mask'] = torch.zeros([len(labels)]).bool()

        # Build pool of training data
        if round == 0: 
            # For first round, get from working vars since the graph includes train now
            full_pool = self.working_vars['r0_train_idx'].clone()
        else:
            # Otherwise, determine the rounds from which to generate the training data from
            init_round = 0 
            source_rounds = list(range(round))
         
            # Exctracting the data based on the pool
            initial_pool = (self.dset['graph'].ndata['creation_round'] < init_round).nonzero().flatten()
            prediction_pool = torch.cat([torch.cat(self.rounds[i]['checks'][:2], 0) for i in source_rounds])
            budget_pool = torch.cat([torch.cat([self.rounds[i]['budgets_pos'], self.rounds[i]['budgets_neg']], 0) for i in source_rounds])
            full_pool = torch.cat([initial_pool.to('cpu'), prediction_pool.to('cpu'), budget_pool], 0).unique().long()

            verPrint(self.verbose, 3, f"Initial pool: {len(initial_pool)}, Prediction pool: {len(prediction_pool)}, Budget pool: {len(budget_pool)}, Full pool: {len(full_pool)}")

        index = torch.arange(len(labels), dtype=torch.long)[full_pool]        
        nonindex = torch.ones_like(labels, dtype=bool)
        nonindex[full_pool] = False
        self.dset['graph'].ndata['test_mask'][nonindex] = 1

        # Split after sanity check
        if (torch.sum(labels[index] == 0) < 2) or (torch.sum(labels[index] == 1) < 2):
            return None, None, None, None
        
        # In full oracle mode, all nodes are available for training
        if self.main_config['full_oracle'] == True:
            index = torch.arange(len(labels), dtype=torch.long)
        
        idx_train, idx_valid, y_train, y_valid = train_test_split(
            index, labels[index], stratify=labels[index],
            train_size = self.train_config['ratio_train'], random_state = self.train_config['random_state'], shuffle=True
        )
        
        # Set result in graph data
        self.dset['graph'].ndata['train_mask'][idx_train] = 1
        self.dset['graph'].ndata['val_mask'][idx_valid] = 1

        verPrint(self.verbose, 3, f"Full graph size: {self.dset['graph'].num_nodes()}, Training: {len(idx_train)}, Val: {len(idx_valid)}, Test: {len(nonindex)}")        
        return idx_train, idx_valid, y_train, y_valid

    def assign_eval_graph_oracle(self, round_num):
        if round_num == 0:
            # Swap back to original graph
            self.dset['temp_graph'].ndata['train_mask'][self.working_vars['r0_train_idx']] = self.dset['graph'].ndata['train_mask'][:len(self.working_vars['r0_train_idx'])].unsqueeze(1)
            self.dset['temp_graph'].ndata['val_mask'][self.working_vars['r0_train_idx']] = self.dset['graph'].ndata['val_mask'][:len(self.working_vars['r0_train_idx'])].unsqueeze(1)
            self.dset['temp_graph'].ndata['test_mask'][self.working_vars['r0_train_idx']] = self.dset['graph'].ndata['test_mask'][:len(self.working_vars['r0_train_idx'])].unsqueeze(1)

            self.dset['graph'] = self.dset['temp_graph']

            self.dset['graph'].ndata['creation_round'][self.working_vars['r0_test_idx']] = 0
            self.dset['graph'].ndata['test_mask'][self.working_vars['r0_test_idx']] = True
            self.dset['graph'].ndata['predicted'][self.working_vars['r0_test_idx']] = False
        else:
            # If not eval graph is same with train graph
            pass

    ####################################
    ### GRAPH/DATA RELATED FUNCTIONS ###
    ####################################
    
    # Get budgeted ground truth for the round
    def get_round_budget(self, round_num):
        # POSITIVES
        all_new_positives = ((self.dset['graph'].ndata['creation_round'] >= 0) & (self.dset['graph'].ndata['label'] == 1)).nonzero().flatten().tolist()
        predicted_new_positives = torch.cat([self.rounds[i]['checks'][0] for i in list(range(round_num + 1))], 0).tolist()
        budget_new_positives = torch.cat([self.rounds[i]['budgets_pos'] for i in list(range(round_num + 1))], 0).tolist()
        
        positive_budget_pool = list(set(all_new_positives) - set(predicted_new_positives) - set(budget_new_positives))
        round_budget = min([len(positive_budget_pool), self.main_config['round_budget_pos']])
        positive_budgets = torch.tensor(random.choice(positive_budget_pool, round_budget, replace=False))         
        
        # NEGATIVES
        base_negatives = ((self.dset['graph'].ndata['creation_round'] < 0) & (self.dset['graph'].ndata['label'] == 0)).nonzero().flatten().tolist()
        predicted_new_negatives = torch.cat([self.rounds[i]['checks'][1] for i in list(range(round_num + 1))], 0).tolist()

        negative_budget_pool = list(set(base_negatives).union(predicted_new_negatives))
        negative_budgets = torch.tensor(random.choice(negative_budget_pool, self.main_config['round_budget_neg'], replace=False))

        return positive_budgets, negative_budgets

    #######################
    ### MODEL FUNCTIONS ###
    #######################
    
    # Model training on single round
    def model_round_train(self, round_num):
        self.working_vars['train_retry'] = 0
        self.working_vars['train_finish'] = False

        # Check if model resets every round
        if self.train_config['round_reset_model']:
            self.init_detector()

        # Model checkpointing
        while True:
            try:
                self.model.save_model(f'{self.temp_model_path}_round')
                break
            except RuntimeError as e:
                    print("CHECKPOINTING ERROR", e)

        while not self.working_vars['train_finish']:
            # Load model
            self.model.load_model(f'{self.temp_model_path}_round')

            # Additional dataset splitting based on experiment type
            if self.main_config['exp_type'] == 'ADVER': # In adversarial experiment, splitting needs to be done at-round instead of in the beginning
                verPrint(self.verbose, 2, '>> Splitting to train-validation set...')
                (idx_train, idx_valid, y_train, y_valid) = self.split_train_test_adver(round_num)
                if idx_train == None:
                    verPrint(self.verbose, 1, 'Not enough data, continuing round without training')
                    return
            
            elif self.main_config['exp_type'] == 'ADVOR': # In adversarial experiment, splitting needs to be done at-round instead of in the beginning and there's difference for oracle
                verPrint(self.verbose, 2, '>> Splitting to train-validation set...')
                (idx_train, idx_valid, y_train, y_valid) = self.split_train_test_oracle(round_num)
                if idx_train == None:
                    verPrint(self.verbose, 1, 'Not enough data, continuing round without training')
                    return
            
            self.print_data_split(self.dset['graph'], '    >> INITIAL')

            # Clone training graph, augment, pseudolabel
            self.model.release_graph()
            train_graph = copy.deepcopy(self.dset['graph']).to(self.train_config['device'])
            self.model.set_graph(train_graph, round_num, self.train_config['device'])

            self.model.augment_graph(augment_strat=self.augment['function'], round_num=round_num, **self.augment['params'])
            self.print_data_split(self.model.graph, '    >> AUGMENTED')

            # Recalculate Training parameters
            labels = self.model.graph.ndata['label']
            self.train_config['ce_weight'] = (1-labels[self.model.graph.ndata['train_mask']]).sum().item() / labels[self.model.graph.ndata['train_mask']].sum().item()
            verPrint(self.verbose, 2, f"    >> Updated cross-entropy weight to {self.train_config['ce_weight']}...")

            # Train!
            time_start = time.time()
            verPrint(self.verbose, 2, '''
-----------------------
STARTING MODEL TRAININGps_l
-----------------------
''')

            if self.model_config['model_name'] in ['XGB', 'XGB-SP', 'ADAPTIVE']:
                res = self.model_train_classic(round_num)
            else:
                res = self.model_train_nn(
                    num_epoch=self.train_config['num_epoch'] if round_num == 0 else self.train_config['num_round_epoch'], 
                    early_stopping=self.train_config['early_stopping'],
                    stuck_stopping=self.train_config['stuck_stopping']
                )
            
            verPrint(self.verbose, 2, '>> Ending training!\n')
            time_end = time.time()

            # Release training graph
            self.model.release_graph() 

            if res != None: # Training eval
                final_trec, final_tpre, final_tmf1, final_tauc, final_tp, final_fp, final_tn, final_fn = res        
                self.rounds[self.current_round]['log_single_eval'].append((f'round_{self.current_round}', f'val_set_best', (time_end - time_start), final_trec, final_tpre, final_tmf1, final_tauc, final_tp, final_fp, final_tn, final_fn))

                verPrint(self.verbose, 1, '-----------------------')
                verPrint(self.verbose, 1, f'TIME COST: {str(time_end - time_start)} s')
                verPrint(self.verbose, 1, f'Best Val: REC {final_trec*100:.2f} PRE {final_tpre*100:.2f} MF1 {final_tmf1*100:.2f} AUC {final_tauc*100:.2f} TP {final_tp} FP {final_fp} TN {final_tn} FN {final_fn}')
                verPrint(self.verbose, 1, '-----------------------')
                self.working_vars['train_finish'] = True
            else: # Model is stuck and entire round needs to be restarted
                self.working_vars['train_finish'] = self.working_vars['train_retry'] >= self.train_config['train_max_retry']
                self.working_vars['train_retry'] = self.working_vars['train_retry'] + 1

    # Train procedure for conventional-variant models with a single fit call
    def model_train_classic(self, round_num):
        labels = self.model.graph.ndata['label']
        
        # Train
        _, _ = self.model.train(self.model.graph, self.train_config['ce_weight'], round_num)

        # Predict
        self.logits, _, _ = self.model(self.model.graph, self.model.graph.ndata['feature'])
        probs = self.logits
        f1, thres = get_best_f1(labels[self.model.graph.ndata['val_mask']], probs[self.model.graph.ndata['val_mask']])
        preds = torch.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1

        # Eval
        _p, _cm  = eval_and_print(0, labels[self.model.graph.ndata['val_mask']], preds[self.model.graph.ndata['val_mask']], probs[self.model.graph.ndata['val_mask']][:, 1], f'Validation')
        trec, tpre, tmf1, tauc = _p
        tp, fp, tn, fn = _cm
        
        return trec, tpre, tmf1, tauc, tp, fp, tn, fn       
        
    # Train procedure for GNN-variant models with epochs
    def model_train_nn(self, num_epoch=100, early_stopping=10, stuck_stopping=5):
        # Load everything to GPU
        self.model = self.model.to(self.train_config['device'])
        self.model.graph = self.model.graph.to(self.train_config['device'])
        self.model._set_device(self.train_config['device'])

        # Initialize evaluation vars
        best_f1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0.
        final_tp, final_fp, final_tn, final_fn = 0, 0, 0, 0

        # Initialize epoch flags
        epoch_counter, stagnant_counter, stuck_counter = 0, 0, 0
        stop_training = False

        # Initialize training module
        self.optimizer = self.train_config['optimizer'](self.model.parameters(), lr=self.train_config['learning_rate'])
        self.loss = self.train_config['loss']   
        
        # Get required tensors
        features = self.model.graph.ndata['feature']
        labels = self.model.graph.ndata['label']
        train_mask = self.model.graph.ndata['train_mask']
        val_mask = self.model.graph.ndata['val_mask']

        # Custom parameters for certain models
        rl_idx = torch.nonzero(self.model.graph.ndata['train_mask'] & labels, as_tuple=False).squeeze(1).to(self.train_config['device'])
        
        # TRAIN
        while not stop_training:     
            self.model.train()
            self.logits = torch.zeros([len(labels), 2]).to(self.train_config['device'])

            # Forward
            logits, test_loss, val_loss = self.model(self.model.graph, features, **{'epoch': epoch_counter, 'ce_weight': self.train_config['ce_weight']})
            self.logits = logits

            if test_loss == None:
                # No internal loss function returned by model, just normally calculate using logits and chosen loss
                epoch_train_loss = self.loss(logits[train_mask], labels[train_mask], weight=torch.tensor([1., self.train_config['ce_weight']]).to(self.train_config['device']))
                epoch_val_loss = self.loss(logits[val_mask], labels[val_mask], weight=torch.tensor([1., self.train_config['ce_weight']]).to(self.train_config['device']))
            else:
                # Model returns some kind of loss
                epoch_train_loss = test_loss
                epoch_val_loss = val_loss

            # Backward
            self.optimizer.zero_grad()
            epoch_train_loss.backward()
            self.optimizer.step()

            # Additional stuff after backward pass if any
            #self.model.postBackprop(**{ 'graph': self.model.graph, 'epoch': epoch_counter, 'rl_idx': rl_idx })     
            
            # EVALUATE
            self.model.eval()

            # Predict
            probs = self.logits.softmax(1)
            f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
            preds = torch.zeros_like(labels)
            preds[probs[:, 1] > thres] = 1

            # Training Eval
            t_p, t_cm  = eval_and_print(0, labels[train_mask], preds[train_mask], probs[train_mask][:, 1], f'Epoch {epoch_counter}')
            trec, tpre, tmf1, tauc = t_p
            ttp, tfp, ttn, tfn = t_cm

            # Validation Eval
            _p, _cm  = eval_and_print(0, labels[val_mask], preds[val_mask], probs[val_mask][:, 1], f'Epoch {epoch_counter}')
            rec, pre, mf1, auc = _p
            tp, fp, tn, fn = _cm
   
            # Check F1 improvement
            if f1 < best_f1:
                stagnant_counter = stagnant_counter + 1
            else:
                # Model checkpointing
                while True:
                    try:
                        print("Saving model")
                        self.model.save_model(f'{self.temp_model_path}_epoch')
                        break
                    except RuntimeError as e:
                            print("CHECKPOINTING ERROR", e)

                best_f1 = f1
                final_trec, final_tpre, final_tmf1, final_tauc = rec, pre, mf1, auc
                final_tp, final_fp, final_tn, final_fn = tp, fp, tn, fn
                stagnant_counter = 0
            
            # Check model getting stuck
            stuck_thres = self.train_config['stuck_threshold']
            if ((tp < stuck_thres) and (fp < stuck_thres)) or ((tn < stuck_thres) and (fn < stuck_thres)):
                stuck_counter = stuck_counter + 1
            else:
                stuck_counter = 0

            # Update epoch counter
            epoch_counter = epoch_counter + 1

            # Print and log
            verPrint(self.verbose, 3, f'Epoch {epoch_counter}, loss: {epoch_train_loss:.3f}-{epoch_val_loss:.3f}, mf1: {tmf1:.3f}-{f1:.3f}, conf: {ttp} {tfp} {ttn} {tfn}|{tp} {fp} {tn} {fn}, (best {best_f1:.3f}) | mem {torch.cuda.mem_get_info()}')

            # Check epoch exit condition
            stop_training = (epoch_counter >= num_epoch) or (stagnant_counter >= early_stopping) or (stuck_counter >= stuck_stopping)
            

        if stuck_counter >= stuck_stopping:
            verPrint(self.train_config['verbose'], 2, '>> Model stuck!')
            return None
        else:            
            verPrint(self.train_config['verbose'], 2, '>> Reached final epoch. Loading best model...')
            self.model.load_model(f'{self.temp_model_path}_epoch')
            verPrint(self.train_config['verbose'], 2, '>> Best model loaded!')

            return final_trec, final_tpre, final_tmf1, final_tauc, final_tp, final_fp, final_tn, final_fn

    # Predict for entire dataset and return prediction; used to update round prediction data
    def model_round_predict(self, round_num):

        # Model Prediction
        self.model.release_graph()
        fresh_graph = copy.deepcopy(self.dset['graph'])
        self.model.set_graph(fresh_graph, round_num, self.train_config['device']) # Set back entire graph
        self.model.eval()
        
        labels = self.model.graph.ndata['label']
        logits, _, _ = self.model(self.model.graph, self.model.graph.ndata['feature'])
        probs = logits.softmax(1)
        _, thres = get_best_f1(labels, probs)
       
        preds = torch.zeros_like(self.model.graph.ndata['label'])
        preds[probs[:, 1] > thres] = 1
        
        # Additional non-model module prediction
        addon_preds, addon_state, _ = self.addon_predictor['function'](self.model, round_num, state=self.addon_predictor['state'], **self.model_config)
        self.addon_predictor['state'] = addon_state
        
        if addon_preds != None:
            final_preds = preds.bool() | addon_preds.bool()
        else:
            final_preds = preds.bool()

        self.model.release_graph() # Release to free memory
        
        # Aggregate prediction - PROBS not edited for logging true model probs
        
        # Classify prediction
        tp = ((labels == preds) & (preds == 1)).nonzero().flatten()
        fp = ((labels != preds) & (preds == 1)).nonzero().flatten()
        tn = ((labels == preds) & (preds == 0)).nonzero().flatten()
        fn = ((labels != preds) & (preds == 0)).nonzero().flatten()

        return final_preds, probs[:, 1], [tp, fp, tn, fn]
        
            
    ###########################
    ### ADVERSARY FUNCTIONS ###
    ###########################

    # TODO: Training phase for adversary
    def adversary_round_train(self):
        return

    # Adversarial generation of new fraud data
    def adversary_round_generate(self):
        new_node_feats, new_edge_feats, seed_ids, new_ids = self.adver_choose.generate_seeds(self.dset['graph'], n_instances=self.main_config['round_new_pos'], label=1, return_id=True) # Generate
        
        # Give the modifier context about the round and how well the previous
        # model was fooled by the seeds (used by the INTELLIGENT generator).
        adver_kwargs = {'round_num': self.current_round}
        prev_probs = self.rounds[self.current_round - 1].get('probs') if self.current_round > 0 else None
        if (prev_probs is not None) and (len(prev_probs) > 0) and (seed_ids.numel() > 0):
            adver_kwargs['missed_probs'] = prev_probs[seed_ids].cpu()

        new_node_feats, new_edge_feats, seed_ids, new_ids = self.adver_mod.modify_seeds(self.dset['graph'], new_node_feats, new_edge_feats, seed_ids, new_ids, **adver_kwargs) # Modify

        # Log diagnostics from the intelligent fraud generator (if present)
        if hasattr(self.adver_mod, 'get_generation_stats') and callable(self.adver_mod.get_generation_stats):
            gen_stats = self.adver_mod.get_generation_stats() or {}
            if gen_stats:
                self.rounds[self.current_round]['log_round'] = self.rounds[self.current_round]['log_round'] | gen_stats

        return new_node_feats, new_edge_feats, seed_ids, new_ids
    
    # Generate negative instances
    def round_generate_negatives(self):
        return self.neg_sampler.generate_seeds(self.dset['graph'], n_instances=self.main_config['round_new_neg'], label=0, return_id=True)
    
    ######################
    ### MISC FUNCTIONS ###
    ######################
    def print_data_split(self, graph, text):
       
        idx_train = graph.ndata['train_mask'].nonzero().flatten()
        idx_valid = graph.ndata['val_mask'].nonzero().flatten()
        idx_test = graph.ndata['test_mask'].nonzero().flatten()
        
        y_train = graph.ndata[f'label'][idx_train]
        y_valid = graph.ndata[f'label'][idx_valid]
        y_test = graph.ndata[f'label'][idx_test]
        
        verPrint(self.verbose, 2, f'{text} DATA SPLIT: {len(idx_train)} train rows ({dict(Counter(y_train.tolist()))}) | {len(idx_valid)} val rows ({dict(Counter(y_valid.tolist()))}) | {len(idx_test)} test rows ({dict(Counter(y_test.tolist()))})')

    def clean_temp_files(self):
        for suffix in ['_epoch.py', '_hybrid_epoch.pt', '_round-M.json', '_round-T.json', '_round.json', '_round.pt',
                        '_adaptive_epoch.pt']:
            if os.path.exists(f"{self.temp_model_path}{suffix}"):
                os.remove(f"{self.temp_model_path}{suffix}")