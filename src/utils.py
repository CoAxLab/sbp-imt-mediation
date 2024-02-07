#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here some functions to gather the results.
They are used later when generating the plots.

"""

import numpy as np
import pandas as pd


def get_results(file_npz):
    y_trues = np.load(file_npz, allow_pickle=True)['y_trues']
    y_preds = np.load(file_npz, allow_pickle=True)['y_preds']

    seed_ids = np.repeat([1,2,3,4,5], 10)
    res_df = pd.DataFrame({'pred': np.concatenate(y_preds),
                           'true': np.concatenate(y_trues),
                           'fold' :np.concatenate([[1+ ii]*len(res) for ii, res in enumerate(y_trues)]),
                           'seed' :np.concatenate([[seed_ids[ii]]*len(res) for ii, res in enumerate(y_trues)])}
                         )

    return res_df

def get_results_R2(file_npz):
    from sklearn.metrics import r2_score
    y_trues = np.load(file_npz, allow_pickle=True)['y_trues']
    y_preds = np.load(file_npz, allow_pickle=True)['y_preds']
    
    res_df = pd.DataFrame({'R2': [r2_score(t,p) for t,p in  zip(y_trues, y_preds)],
                           'fold' : [1+ ii for ii, _ in enumerate(y_trues)],
                           'repeat': np.repeat([1,2,3,4,5], 10)})

    return res_df
    
def generate_res_df(preds_df):
    """This function shows all performance metrics together"""
    
    from sklearn.metrics import r2_score, mean_absolute_error
    from scipy.stats import pearsonr
    
    from pathlib import Path
    import sys
    sys.path.append(Path(__file__).resolve().parent.as_posix())
    from metrics import r2_med
    
    preds_grouped_df = preds_df.groupby(['seed', 'model'])
    n_points = preds_grouped_df['fold'].count().iloc[0]
    seeds = [1, 2, 3, 4, 5]
    
    r2_seed_df = []
    effects_seed_df = []
    for seed in seeds:

        xy_t = preds_grouped_df.get_group((seed, 'xy'))['true'].to_numpy()
        xy_p = preds_grouped_df.get_group((seed, 'xy'))['pred'].to_numpy()
        r2_xy = r2_score(xy_t, xy_p)
        corr_xy = pearsonr(xy_t, xy_p)[0]
        mae_xy = mean_absolute_error(xy_t, xy_p)

        xm_t = preds_grouped_df.get_group((seed, 'xm'))['true'].to_numpy()
        xm_p = preds_grouped_df.get_group((seed, 'xm'))['pred'].to_numpy()
        r2_xm = r2_score(xm_t, xm_p)
        corr_xm = pearsonr(xm_t, xm_p)[0]
        mae_xm = mean_absolute_error(xm_t, xm_p)
        
        my_t = preds_grouped_df.get_group((seed, 'my'))['true'].to_numpy()
        my_p = preds_grouped_df.get_group((seed, 'my'))['pred'].to_numpy()
        r2_my = r2_score(my_t, my_p)
        corr_my = pearsonr(my_t, my_p)[0]
        mae_my = mean_absolute_error(my_t, my_p)
        
        xmy_t = preds_grouped_df.get_group((seed, 'xmy'))['true'].to_numpy()
        xmy_p = preds_grouped_df.get_group((seed, 'xmy'))['pred'].to_numpy()
        r2_xmy = r2_score(xmy_t, xmy_p)
        corr_xmy = pearsonr(xmy_t, xmy_p)[0]
        mae_xmy = mean_absolute_error(xmy_t, xmy_p)

        data = (xy_t, xy_p, my_t, my_p, xmy_t, xmy_p)
        res_med = r2_med(xy_t, xy_p, my_t, my_p, xmy_t, xmy_p)
        
        effects_df = pd.DataFrame({'xy': [r2_xy, mae_xy, corr_xy],
                                 'xm':  [r2_xm,  mae_xm, corr_xm],
                                 'my': [r2_my,  mae_my, corr_my],  
                                 'xmy': [r2_xmy,  mae_xmy, corr_xmy], 
                                })
        effects_df['seed'] = seed
        effects_df['measure'] = ["R2", "MAE", "r"]
        effects_seed_df.append(effects_df)
        
        r2_df = pd.DataFrame({'xy': [r2_xy],
                                 'xm':  [r2_xm],
                                 'my': [r2_my],  
                                 'xmy': [r2_xmy], 
                                 'med': [res_med],
                                })
        r2_df['seed'] = seed
        r2_seed_df.append(r2_df)
    return pd.concat(effects_seed_df), pd.concat(r2_seed_df)
