import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

def read_df(foldername, grouper = []):
    round_df = pd.concat([
        pd.read_csv(f'../result/{foldername}/{fname}') for fname in list(filter(lambda x: ('-E.csv' in x), os.listdir(f"../result/{foldername}/")))
    ])

    round_df['round_int'] = round_df.apply(lambda x: int(x['round'][6:]) + 1, axis=1)
    round_df['eval_type'] = np.where((round_df['round'] == 'round_0') & (round_df['eval_type'] == 'test_set'), 'round_0_nodes', round_df['eval_type'])
    round_df['agg_backbone'] = round_df['boost_agg_backbone'].astype(str).apply(lambda x: x.split(".")[-1][:-2])

    round_df['num_samples'] = round_df['tp'] + round_df['fp'] + round_df['tn'] + round_df['fn'] 
    round_df['fprate'] = (round_df['fp'] / (round_df['fp'] + round_df['tn'])).fillna(0)
    
    
    round_df['tpr'] = (round_df['tp'] / (round_df['tp'] + round_df['fn'])).fillna(0)
    round_df['fnr'] = (round_df['fn'] / (round_df['tp'] + round_df['fn'])).fillna(0)
    
    round_df['fpr'] = (round_df['fp'] / (round_df['fp'] + round_df['tn'])).fillna(0)
    round_df['tnr'] = (round_df['tn'] / (round_df['fp'] + round_df['tn'])).fillna(0)

    for colname in grouper + ['round_window']:
        round_df[colname] = round_df[colname].astype(str)

    return round_df


def get_metric_df(df, grouper, metric='f1', aggs=['mean', 'std']):
    values = []
    for name, group in df.groupby(grouper + ['trial']):
        if len(group) > 10:
            single_df = group.query('eval_type == "round_0_nodes"').copy()
            r0_start, r0_end = single_df[metric].values[0], single_df[metric].values[-1]
            r0_gap = r0_end - r0_start
        
            single_df = group[group.apply(lambda x: x['round'] in x['eval_type'], axis=1)].copy()
            rsub_mean = single_df[metric].values[1:].mean()

            values.append([*name, r0_start, r0_end, r0_gap, rsub_mean])
        else:
            raise Exception('Grouping not atomic!')

    result = pd.DataFrame(values, columns=grouper + ['trial', 'r0_start', 'r0_end', 'r0_gap', 'rsub_mean']).drop(columns=['trial']).groupby(grouper).agg(aggs).reset_index()
    return result

def get_cover_df(df, grouper, aggs=['mean', 'std']):
    values = []
    for name, group in df.groupby(grouper + ['trial']):
        if len(group) > 10:
            maxround = group['round'].max()
            single_df = group.query(f'(round =="{str(maxround)}")').copy()
            single_df['pos_ratio'] = single_df['predicted_pos'] / (single_df['round_new_pos'] * 19)
            single_df['og_pos_ratio'] = single_df['predicted_pos_og'] / (single_df['round_new_pos'] * 10)
            
            vals = single_df[['pos_ratio', 'og_pos_ratio', 'prediction_speed']].values[0]

            values.append([*name, *vals])
        else:
            raise Exception('Grouping not atomic!')

    result = pd.DataFrame(values, columns=grouper + ['trial', 'pos_ratio', 'og_pos_ratio', 'prediction_speed']).drop(columns=['trial']).groupby(grouper).agg(aggs).reset_index()
    return result

def get_loss_df(df, grouper, aggs=['mean', 'std']):
    values = []

    cols = [c for c in df.columns if (('loss' in c) and ('type' not in c) and ('sample' not in c) and ('gamma' not in c) and (c != 'loss'))]
    for name, group in df.groupby(grouper + ['trial']):
        if len(group) > 10:
            maxround = group['round'].max()
            single_df = group.query(f'(round =="{str(maxround)}")').copy()

            for c in cols:
                single_df[c] = single_df[c].astype('float')

            vals = single_df[cols].values[0]
            values.append([*name, *vals])
        else:
            raise Exception('Grouping not atomic!')
    
    result = pd.DataFrame(values, columns=grouper + ['trial'] + cols).drop(columns=['trial']).groupby(grouper).agg(aggs).reset_index()
    return result

def print_round_graph(df, grouper, metric='f1', ylim=(0.45, 1), texts=True, hue_order=None):
    if len(grouper) > 3:
        supergroup = '_'.join(grouper[:-2])
        df[supergroup]  = ''

        for i, g in enumerate(grouper[:-2]):
            df[supergroup] = df[supergroup] + df[g]

        grouper = [supergroup] + grouper[-2:]

    out_out_colname = grouper[0]
    out_out_col_vals = df[out_out_colname].unique()
    for out_out_val in out_out_col_vals:
        print(f"\n==={out_out_val}===".upper())
        temp_temp_df = df.query(f'{out_out_colname} == "{out_out_val}"')

        out_colname = grouper[1]
        out_col_vals = temp_temp_df[out_colname].unique()

        col = int(np.ceil(np.sqrt(len(out_col_vals))))
        row = int(np.ceil(len(out_col_vals) / col))

        fig = plt.figure(figsize=(col*10, row*5))
        for idx, out_val in enumerate(out_col_vals):
            temp_df = temp_temp_df.query(f'{out_colname} == "{out_val}"')
            temp_df = temp_df.replace([np.inf, -np.inf], np.nan)

            colname = grouper[2]
            col_vals = temp_df[colname].unique()
            dfs = []

            for val in col_vals:
                dfs.append(temp_df.groupby([colname, 'round_int'])[[metric, 'auc']].mean().reset_index().query(f'{colname} == "{val}"')[['round_int', metric]].values)

            ax = plt.subplot(row, col, idx + 1)
            p1 = sb.lineplot(
                data=temp_df, x='round_int', y=metric, 
                hue=colname, hue_order=hue_order, linewidth=2, err_style='bars', 
                dashes=False, marker='o', markersize=10)
            plt.legend()

            if texts:
                for vals in dfs:
                    for i in vals:
                            ax.text(i[0], i[1],f'{i[1]:.2f}', size=10, horizontalalignment='center', verticalalignment='center')

            plt.ylabel(f"{metric.upper()} Score" if idx % col == 0 else '', fontweight='bold')
            plt.xlabel("Round #" if idx // col == (row - 1) else '', fontweight='bold')        
            
            plt.xticks(np.arange(1, 6, 1), fontsize=12)
            plt.title(out_val, fontweight='bold')
            plt.yticks(fontsize=12)
            plt.ylim(ylim)
            plt.grid()

        plt.show()