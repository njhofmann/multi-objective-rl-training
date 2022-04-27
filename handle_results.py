import argparse as ap
import pathlib as pl
import pickle as p
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml as y
import yaml.loader as yl


def plot_data_type_results(data_type, data_frame, y_label, title, group_cols, save_dirc, simple_moving_avg, sort_col, file_name_start):
    group_results = []
    for key, group in data_frame.groupby(group_cols):
        if not isinstance(key, tuple):
            key = tuple([key])
        label = '_'.join([f'{c}[{key[i]}]' for i, c in enumerate(group_cols)])
        group = group[[sort_col, data_type]]

        key_group_vals = {col: key[i] for i, col in enumerate(group_cols)}  # for dataframe

        if simple_moving_avg > 1:
            sma_groups = []
            unique_epochs = list(reversed(group['epoch'].unique()))
            for i, epoch in enumerate(unique_epochs):
                if i < len(unique_epochs) - simple_moving_avg:
                    epochs_in_group = [unique_epochs[i + j] for j in range(simple_moving_avg)]
                    group_data = group[group['epoch'].isin(epochs_in_group)].copy()
                    group_data['epoch'] = epoch
                    sma_groups.append(group_data)
            group = pd.concat(sma_groups)
            # group = (group.groupby(sort_col, as_index=False).mean()  
            #               .rolling(simple_moving_avg).mean()
            #               .dropna())
            
            # for row in group.iterrows():
            #     row = row[1]
            #     group_results.append({sort_col: row[sort_col], f'{data_type}_avg': row[data_type], **key_group_vals})
        sns.lineplot(x=sort_col, y=data_type, data=group, label=label)

    file_name = f'{file_name_start}_{data_type}'
    if simple_moving_avg > 0:
        title += f' SMA {simple_moving_avg}'
        file_name += f'_sma[{simple_moving_avg}]'

    # move legend outside plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.title(title)
    plt.xlabel(f'Training {sort_col.capitalize()}')
    plt.ylabel(y_label)

    save_path = str(save_dirc.joinpath(file_name))
    plt.savefig(f'{save_path}.png', bbox_inches='tight')
    plt.close()

    if simple_moving_avg > 0:
        (pd.DataFrame(group_results).sort_values(by=[sort_col, f'{data_type}_avg'], ascending=False)
                                    .to_csv(f'{save_path}.csv', index=False))


def plot_loss_results(result_dircs, save_dirc, merge_cols, simple_moving_avg):
    results = []
    for result_dirc in result_dircs.iterdir():
        result_df = pd.read_csv(str(result_dirc / 'train_loss_results.csv'))

        with (result_dirc / 'params.yaml').open('r') as f:
            result_info = y.load(f, Loader=yl.SafeLoader)
        
        result_df = result_df.pct_change().dropna()
        result_df.reset_index(inplace=True)
        result_df = result_df.rename(columns={'index': 'updates'})

        for col in merge_cols:
            if col not in result_info:
                raise ValueError(f'merge column {col} not in info for {results_dirc}')
            result_df[col] = str(result_info[col])

        results.append(result_df)
   
    results = pd.concat(results).reset_index()
    for data_type, y_label, title in ('actor', 'Actor Loss', 'Percent Change in Actor Loss'), \
                                     ('length', 'Episode Duration', 'Average Episode Duration'):
        if data_type in results.columns:
            plot_data_type_results(data_type, results, y_label, title, merge_cols, save_dirc, simple_moving_avg, 'updates', 'train')


def plot_eval_results(result_dircs, save_dirc, merge_cols, simple_moving_avg):
    results = []
    for result_dirc in result_dircs.iterdir():
        result_df = pd.read_csv(str(result_dirc / 'eval_results.csv'), index_col='epoch')

        with (result_dirc / 'params.yaml').open('r') as f:
            result_info = y.load(f, Loader=yl.SafeLoader)

        for col in merge_cols:
            if col not in result_info:
                raise ValueError(f'merge column {col} not in info for {results_dirc}')
            result_df[col] = str(result_info[col])
            
        results.append(result_df)
   
    results = pd.concat(results).reset_index()
    for data_type, y_label, title in ('cum_rwd', 'Episode Reward', 'Average Episode Reward'), \
                                     ('len', 'Episode Duration', 'Average Episode Duration'):
        if data_type in results.columns:
            plot_data_type_results(data_type, results, y_label, title, merge_cols, save_dirc, simple_moving_avg, 'epoch', 'eval')
        

def get_args():
    parser = ap.ArgumentParser()
    parser.add_argument('--results_dirc', '-r', type=str, required=True, help='directory to results files from')
    parser.add_argument('--group_cols', '-g', nargs='+', default='method', help='experiment info to group results by')
    parser.add_argument('--sma', default=0, type=int, help='# of previous timesteps to use in performing a simple moving average')
    parser.add_argument('--eval', '-e', action='store_true', help='process evaluation results')
    parser.add_argument('--train', '-t', action='store_true', help='process training results')
    parser.add_argument('--losses', '-l', action='store_true', help='process training losses')    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    results_dirc = pl.Path(args.results_dirc)
    save_dirc = pl.Path('plots') / results_dirc.name
    save_dirc.mkdir(exist_ok=True, parents=True)
    
    if args.eval:
        plot_eval_results(results_dirc, save_dirc, args.group_cols, args.sma)
    
    if args.train:
        pass

    if args.losses:
        plot_loss_results(results_dirc, save_dirc, args.group_cols, args.sma)