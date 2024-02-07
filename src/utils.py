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
