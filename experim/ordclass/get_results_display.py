import os
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import seaborn as sns

from experim.ordclass import CONFIG

FILE_RESULTS_PERFORM = 'Results_Analysis-Performance summary.json'
FILE_RESULTS_BAYES = 'Results_Analysis-Bayesian comparison.json'

RESULTS_ORDER = ['DDR-Classif', 'UTKF-Classif', 'BACH-Classif', 'BACH-Segment']

FIG_SIZE = (11, 9)

NUMBER_ROUND = 3
NUMBER_FORMAT = '.1%'

EXPERIMENTS = {
    0: {'name': 'regr-l1', 'label': r'Regr-$L_{1}$', 'color': 'tab:orange', 'marker': 'o'},
    1: {'name': 'regr-l2', 'label': r'Regr-$L_{2}$', 'color': 'tab:orange', 'marker': 's'},
    2: {'name': 'nom-mce', 'label': r'MCE', 'color': 'tab:green', 'marker': 'o'},
    3: {'name': 'nom-mfoc', 'label': r'MFoc', 'color': 'tab:green', 'marker': 's'},
    4: {'name': 'ur-mce', 'label': r'UR$\beta$-MCE', 'color': 'tab:blue', 'marker': 'o'},
    5: {'name': 'ur-mfoc', 'label': r'UR$\beta$-MFoc', 'color': 'tab:blue', 'marker': 's'},
    6: {'name': 'owk-l1', 'label': r'OWK-$L_{1}$', 'color': 'tab:purple', 'marker': 'o'},
    7: {'name': 'owk-l2', 'label': r'OWK-$L_{2}$', 'color': 'tab:purple', 'marker': 's'},
    8: {'name': 'obd-ce_f', 'label': r'$\bf{OBD\text{-}CE}$[F]', 'color': 'tab:red', 'marker': 'o'},
    9: {'name': 'obd-ce_h', 'label': r'$\bf{OBD\text{-}CE}$[H]', 'color': 'tab:red', 'marker': 'D'},
    10: {'name': 'obd-foc_f', 'label': r'$\bf{OBD\text{-}Foc}$[F]', 'color': 'tab:red', 'marker': 's'},
    11: {'name': 'obd-foc_h', 'label': r'$\bf{OBD\text{-}Foc}$[H]', 'color': 'tab:red', 'marker': 'P'}
}

COLORMAP_NET = 'RdYlGn'
COLORMAP_DRAW = 'Blues'

PERFORM_METRICS = {
    'avg_f1': {'dir': 'max', 'label': r'$maF_{1}$', 'color': 'dodgerblue'},
    'mcc': {'dir': 'max', 'label': r'$MCC$', 'color': 'deepskyblue'},
    'avg_mae': {'dir': 'min', 'label': r'$maNMAE$', 'color': 'yellowgreen'},
    'avg_rmse': {'dir': 'min', 'label': r'$maNRMSE$', 'color': 'greenyellow'},
    'ckappa': {'dir': 'max', 'label': r'$QWK$', 'color': 'olivedrab'},
    'avg_auroc': {'dir': 'max', 'label': r'$maAUROC$', 'color': 'darkorchid'},
    'avg_auprc': {'dir': 'max', 'label': r'$maAUPRC$', 'color': 'mediumorchid'},
    'brier': {'dir': 'min', 'label': r'$Brier$', 'color': 'hotpink'},
    'rps': {'dir': 'min', 'label': r'$RPS$', 'color': 'pink'}
}


if __name__ == '__main__':
    exprm_names = []
    exprm_labels = []
    for exprm_idx_, exprm_dict_ in EXPERIMENTS.items():
        exprm_names.append(exprm_dict_['name'])
        exprm_labels.append(exprm_dict_['label'])
    num_exprm = len(exprm_names)

    metric_names = []
    metric_labels = []
    for metric_, metric_dict_ in PERFORM_METRICS.items():
        metric_names.append(metric_)
        metric_labels.append(metric_dict_['label'])
    num_metric = len(metric_names)

    # print tables
    with open(os.path.join(CONFIG.PATH_RESULTS, FILE_RESULTS_PERFORM), 'r') as f_perform:
        perform_json = json.load(f_perform)

    for metric_, perform_dict_ in perform_json.items():
        perform_avg_ = perform_dict_['avg']
        perform_std_ = perform_dict_['std']
        df_perform_avg_ = pd.DataFrame(perform_avg_).astype(float)
        df_perform_std_ = pd.DataFrame(perform_std_).astype(float)

        df_perform_avg_ = df_perform_avg_.reindex(RESULTS_ORDER)
        df_perform_std_ = df_perform_std_.reindex(RESULTS_ORDER)

        df_perform_avg_.columns = exprm_names
        df_perform_std_.columns = exprm_names

        # prepare LaTeX print
        df_perform_avg_str_ = df_perform_avg_.map(lambda x: '-' if np.isnan(x) else f'{x:.3f}')
        df_perform_std_str_ = df_perform_std_.map(lambda x: '-' if np.isnan(x) else f'{x:.3f}')

        df_perform_comb_str_ = df_perform_avg_str_ + r'\textsubscript{' + df_perform_std_str_ + '}'
        # df_perform_comb_str_ = df_perform_comb_str_.to_latex(escape=False)
        with pd.option_context('display.max_columns', None, 'display.expand_frame_repr', False):
            print(metric_, flush=True)
            print(df_perform_comb_str_, flush=True)

        # round and get ranks
        if PERFORM_METRICS[metric_]['dir'] == 'max':
            rank_ascend = False
        elif PERFORM_METRICS[metric_]['dir'] == 'min':
            rank_ascend = True
        else:
            raise ValueError

        df_perform_avg_round_ = df_perform_avg_.round(NUMBER_ROUND)
        df_perform_avg_rank_ = df_perform_avg_round_.rank(axis='columns', method='dense', ascending=rank_ascend)
        df_perform_avg_rank_ = df_perform_avg_rank_.dropna(axis='columns', how='all').astype(np.uint8)
        with pd.option_context('display.max_columns', None, 'display.expand_frame_repr', False):
            print(metric_, flush=True)
            print(df_perform_avg_rank_, flush=True)

    # plot figures for Bayesian comparison tables
    with open(os.path.join(CONFIG.PATH_RESULTS, FILE_RESULTS_BAYES), 'r') as f_bayes:
        bayes_json = json.load(f_bayes)

    for metric_, bayes_dict_ in bayes_json.items():
        # prepare data
        bayes_win_ = bayes_dict_['win']
        bayes_draw_ = bayes_dict_['draw']
        bayes_loss_ = bayes_dict_['loss']
        df_bayes_win_ = pd.DataFrame(bayes_win_).astype(float)
        df_bayes_draw_ = pd.DataFrame(bayes_draw_).astype(float)
        df_bayes_loss_ = pd.DataFrame(bayes_loss_).astype(float)

        df_bayes_win_.columns = exprm_names
        df_bayes_draw_.columns = exprm_names
        df_bayes_loss_.columns = exprm_names
        df_bayes_win_.index = exprm_names
        df_bayes_draw_.index = exprm_names
        df_bayes_loss_.index = exprm_names

        df_bayes_win_ = df_bayes_win_.add_prefix('challenger@', axis='index')
        df_bayes_draw_ = df_bayes_draw_.add_prefix('challenger@', axis='index')
        df_bayes_loss_ = df_bayes_loss_.add_prefix('challenger@', axis='index')
        df_bayes_win_ = df_bayes_win_.add_prefix('baseline@', axis='columns')
        df_bayes_draw_ = df_bayes_draw_.add_prefix('baseline@', axis='columns')
        df_bayes_loss_ = df_bayes_loss_.add_prefix('baseline@', axis='columns')

        df_bayes_net_ = df_bayes_win_ - df_bayes_loss_

        # plot
        fig_bayes_, ax_bayes_ = plt.subplots(figsize=FIG_SIZE)

        # masking
        mask_upper = np.triu(np.ones_like(df_bayes_net_.to_numpy(), dtype=bool))
        mask_lower = np.tril(np.ones_like(df_bayes_net_.to_numpy(), dtype=bool))

        # plot draws on the lower triangle
        sns.heatmap(df_bayes_draw_.to_numpy().T,
                    mask=mask_upper,
                    vmin=0.0,
                    vmax=1.0,
                    square=True,
                    fmt=NUMBER_FORMAT,
                    cmap=COLORMAP_DRAW,
                    cbar_kws={'label': r'P($ROPE$)=P(Challenger$\approx$Baseline)',
                              'format': FuncFormatter(lambda x, pos: '{:.0%}'.format(x))},
                    annot=True,
                    annot_kws={'fontsize': 9},
                    ax=ax_bayes_)

        # plot wins - losses on the upper triangle
        sns.heatmap(df_bayes_net_.to_numpy().T,
                    mask=mask_lower,
                    vmin=-1.0,
                    vmax=+1.0,
                    square=True,
                    fmt=NUMBER_FORMAT,
                    cmap=COLORMAP_NET,
                    cbar_kws={'label': r'$\Delta$P=P(Challenger>Baseline)-P(Challenger<Baseline)',
                              'format': FuncFormatter(lambda x, pos: '{:.0%}'.format(x))},
                    annot=True,
                    annot_kws={'fontsize': 9},
                    xticklabels=exprm_labels,
                    yticklabels=exprm_labels,
                    ax=ax_bayes_)

        ax_bayes_.set_xticklabels(ax_bayes_.get_xticklabels(), rotation=45, fontsize=10)
        ax_bayes_.set_yticklabels(ax_bayes_.get_yticklabels(), rotation=0, fontsize=10)

        # add thinner separation lines between families of loss functions
        ax_bayes_.axvline(x=2, color='gray', linewidth=1.25)
        ax_bayes_.axhline(y=2, color='gray', linewidth=1.25)
        ax_bayes_.axvline(x=4, color='gray', linewidth=1.25)
        ax_bayes_.axhline(y=4, color='gray', linewidth=1.25)
        ax_bayes_.axvline(x=6, color='gray', linewidth=1.25)
        ax_bayes_.axhline(y=6, color='gray', linewidth=1.25)

        # add thicker separation lines between literature (8) and proposed (4)
        ax_bayes_.axvline(x=8, color='black', linewidth=2.00)
        ax_bayes_.axhline(y=8, color='black', linewidth=2.00)

        # tickers
        ax_bayes_.tick_params(axis='both', which='major',
                              bottom=False, labelbottom=False,
                              top=True, labeltop=True,
                              left=True, labelleft=True,
                              right=False, labelright=False)

        # labels
        ax_bayes_.set_title(PERFORM_METRICS[metric_]['label'], fontsize=20, loc='left')
        ax_bayes_.set_xlabel(r'$\it{Loss\,fn}$: $\bf{Challenger}$', fontsize=12)
        ax_bayes_.set_ylabel(r'$\it{Loss\,fn}$: $\bf{Baseline}$', fontsize=12)
        ax_bayes_.xaxis.set_label_position('top')
        ax_bayes_.yaxis.set_label_position('left')

        # add a subtle dividing line on the diagonal
        ax_bayes_.plot([0, num_exprm], [0, num_exprm], color='grey', linewidth=1, linestyle='--')

        plt.tight_layout()
        plt.show(block=True)

    # all-round performance comparison
    l_bayes_net_agg = []
    for metric_, bayes_dict_ in bayes_json.items():
        # prepare data
        bayes_win_ = bayes_dict_['win']
        # bayes_draw_ = bayes_dict_['draw']
        bayes_loss_ = bayes_dict_['loss']
        df_bayes_win_ = pd.DataFrame(bayes_win_).astype(float)
        # df_bayes_draw_ = pd.DataFrame(bayes_draw_).astype(float)
        df_bayes_loss_ = pd.DataFrame(bayes_loss_).astype(float)

        df_bayes_win_.columns = exprm_names
        # df_bayes_draw_.columns = exprm_names
        df_bayes_loss_.columns = exprm_names
        df_bayes_win_.index = exprm_names
        # df_bayes_draw_.index = exprm_names
        df_bayes_loss_.index = exprm_names

        df_bayes_win_ = df_bayes_win_.add_prefix('challenger@', axis='index')
        # df_bayes_draw_ = df_bayes_draw_.add_prefix('challenger@', axis='index')
        df_bayes_loss_ = df_bayes_loss_.add_prefix('challenger@', axis='index')
        df_bayes_win_ = df_bayes_win_.add_prefix('baseline@', axis='columns')
        # df_bayes_draw_ = df_bayes_draw_.add_prefix('baseline@', axis='columns')
        df_bayes_loss_ = df_bayes_loss_.add_prefix('baseline@', axis='columns')

        df_bayes_net_ = df_bayes_win_ - df_bayes_loss_

        # aggregate across baselines: mean
        df_bayes_net_agg_ = df_bayes_net_.mean(axis='columns', skipna=True)
        l_bayes_net_agg.append(df_bayes_net_agg_)

    df_bayes_net_agg_all = pd.concat(l_bayes_net_agg, axis='columns').astype(float)
    df_bayes_net_agg_all.columns = metric_names

    df_bayes_net_agg_sel = df_bayes_net_agg_all.dropna(axis='columns', how='any')

    # aggregate across metrics: sum
    df_bayes_net_agg_all['agg'] = df_bayes_net_agg_all.sum(axis='columns', skipna=True)
    df_bayes_net_agg_sel['agg'] = df_bayes_net_agg_sel.sum(axis='columns', skipna=True)
    df_bayes_net_agg_all = df_bayes_net_agg_all.sort_values('agg', ascending=True)
    df_bayes_net_agg_sel = df_bayes_net_agg_sel.sort_values('agg', ascending=True)

    # stacked plot for all-round comparison
    fig_stack_all, ax_stack_all = plt.subplots(figsize=FIG_SIZE)
    y_pos_all = np.arange(num_exprm)

    # track where the next bar should start (positives stack right, negatives stack left)
    left_pos_all = np.zeros(num_exprm)
    left_neg_all = np.zeros(num_exprm)

    # iterative stacking
    for metric_, metric_dict_ in PERFORM_METRICS.items():
        metric_vals_ = df_bayes_net_agg_all[metric_]

        # scale values: divide by the count of valid metrics for that experiment
        # metric_vals_ = metric_vals_ / metric_counts

        mask_pos_ = (metric_vals_ >= 0)
        mask_neg_ = (metric_vals_ < 0)

        # draw positive segments
        widths_pos = np.where(mask_pos_, metric_vals_, 0)
        ax_stack_all.barh(y_pos_all, widths_pos,
                          left=left_pos_all,
                          color=metric_dict_['color'],
                          label=metric_dict_['label'])

        # update offsets for the next positive segment
        left_pos_all += widths_pos

        # draw negative segments
        widths_neg = np.where(mask_neg_, metric_vals_, 0)
        ax_stack_all.barh(y_pos_all, widths_neg,
                          left=left_neg_all,
                          color=metric_dict_['color'])

        # update offsets for the next negative segment
        left_neg_all += widths_neg

    # extra visual elements
    exprm_labels_sort = []
    for exprm_name_full_ in df_bayes_net_agg_all.index:
        exprm_name_short_ = exprm_name_full_.replace('challenger@', '')
        for exprm_idx_, exprm_dict_ in EXPERIMENTS.items():
            if exprm_dict_['name'] == exprm_name_short_:
                # short name to pretty name
                label_ = exprm_dict_['label']
                exprm_labels_sort.append(label_)
                break

    ax_stack_all.axvline(0, color='black', linewidth=1.5, linestyle='--')
    ax_stack_all.set_yticks(y_pos_all)
    ax_stack_all.set_yticklabels(exprm_labels_sort)

    ax_stack_all.set_xlabel(r'Cumulative Net Dominance $\sum \Delta P$', fontsize=11)
    ax_stack_all.set_title('Bayesian Hierarchical evaluation: Cumulative Net Probabilities', fontsize=14)

    # de-duplicate legend (caused by splitting positive/negative draws)
    handles_stack_all, labels_stack_all = ax_stack_all.get_legend_handles_labels()
    dict_label_stack_all = dict(zip(labels_stack_all, handles_stack_all))
    ax_stack_all.legend(dict_label_stack_all.values(),
                        dict_label_stack_all.keys(),
                        title='Performance metrics',
                        bbox_to_anchor=(1.02, 1),
                        loc='upper left')

    plt.tight_layout()
    plt.show(block=True)

    # stacked plot for all-round comparison
    fig_stack_sel, ax_stack_sel = plt.subplots(figsize=FIG_SIZE)
    y_pos_sel = np.arange(num_exprm)

    # track where the next bar should start (positives stack right, negatives stack left)
    left_pos_sel = np.zeros(num_exprm)
    left_neg_sel = np.zeros(num_exprm)

    # iterative stacking
    for metric_, metric_dict_ in PERFORM_METRICS.items():
        if metric_ not in df_bayes_net_agg_sel.columns:
            continue
        metric_vals_ = df_bayes_net_agg_sel[metric_]

        # scale values: divide by the count of valid metrics for that experiment
        # metric_vals_ = metric_vals_ / metric_counts

        mask_pos_ = (metric_vals_ >= 0)
        mask_neg_ = (metric_vals_ < 0)

        # draw positive segments
        widths_pos = np.where(mask_pos_, metric_vals_, 0)
        ax_stack_sel.barh(y_pos_sel, widths_pos,
                          left=left_pos_sel,
                          color=metric_dict_['color'],
                          label=metric_dict_['label'])

        # update offsets for the next positive segment
        left_pos_sel += widths_pos

        # draw negative segments
        widths_neg = np.where(mask_neg_, metric_vals_, 0)
        ax_stack_sel.barh(y_pos_sel, widths_neg,
                          left=left_neg_sel,
                          color=metric_dict_['color'])

        # update offsets for the next negative segment
        left_neg_sel += widths_neg

    # extra visual elements
    exprm_labels_sort = []
    for exprm_name_full_ in df_bayes_net_agg_sel.index:
        exprm_name_short_ = exprm_name_full_.replace('challenger@', '')
        for exprm_idx_, exprm_dict_ in EXPERIMENTS.items():
            if exprm_dict_['name'] == exprm_name_short_:
                # short name to pretty name
                label_ = exprm_dict_['label']
                exprm_labels_sort.append(label_)
                break

    ax_stack_sel.axvline(0, color='black', linewidth=1.5, linestyle='--')
    ax_stack_sel.set_yticks(y_pos_sel)
    ax_stack_sel.set_yticklabels(exprm_labels_sort)

    ax_stack_sel.set_xlabel(r'Cumulative Net Dominance $\sum \Delta P$', fontsize=11)
    ax_stack_sel.set_title('Bayesian Hierarchical evaluation: Cumulative Net Probabilities', fontsize=14)

    # de-duplicate legend (caused by splitting positive/negative draws)
    handles_stack_sel, labels_stack_sel = ax_stack_sel.get_legend_handles_labels()
    dict_label_stack_sel = dict(zip(labels_stack_sel, handles_stack_sel))
    ax_stack_sel.legend(dict_label_stack_sel.values(),
                        dict_label_stack_sel.keys(),
                        title='Performance metrics',
                        bbox_to_anchor=(1.02, 1),
                        loc='upper left')

    plt.tight_layout()
    plt.show(block=True)

    print('Done!')
