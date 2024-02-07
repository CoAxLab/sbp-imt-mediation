#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 14:05:27 2022

@author: javi
"""

# IMPORTS
import numpy as np
import pandas as pd
import argparse
import sys
import os
import json
from pathlib import Path
from os.path import join as opj
from datetime import datetime

from nilearn.input_data import NiftiMasker

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import StratifiedKFold
from my_sklearn_tools.pca_regressors import LassoPCR, ElasticNetPCR, RidgePCR
#from my_sklearn_tools.model_selection import StratifiedKFoldReg

from joblib import Parallel, delayed

project_dir = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(project_dir)

from src.input_data import load_data
from src.harmonization import norm_standardize_my, harmonize_cohorts

from src.models import L2MediationModel

def fit_mediation(i_boot, 
                  X, m, y, 
                  study,labels,
                  model, 
                  output_dir):
    
    from sklearn.utils import resample

    X_boot, y_boot, m_boot, study_boot = resample(X, y, m, study,
                                                  random_state=i_boot,
                                                  stratify=labels)
    y_boot_ss = norm_standardize_my(y_boot, study_boot)
    m_boot_ss = norm_standardize_my(m_boot, study_boot)
    X_boot_combat = harmonize_cohorts(X_boot, study_boot)
    
    del X_boot, y_boot, m_boot, study_boot
      
    try:
        
        model.fit(X=X_boot_combat, y=y_boot_ss, m=m_boot_ss)
        
        c_boot = model.weights_['xy']
        a_boot = model.weights_['xm']
        
        
        b_boot =  model.weights_['xmy'][-1]
        cp_boot = model.weights_['xmy'][:-1]

        # Save results for each bootstrap
        np.savez_compressed(opj
                            (output_dir,
                             f'weights_boot-{str(i_boot+1).zfill(3)}.npz'),
                            a=a_boot, b=b_boot, cprime=cp_boot, c=c_boot
                            )
        return True
    except np.linalg.LinAlgError as e:
        return False

def main():

    parser = argparse.ArgumentParser(description='Run a particular experiment')
    parser.add_argument('--target',
                        dest="target",
                        type=str,
                        required=True,
                        help='Which variable as a target (e.g. mavgimt)')
    parser.add_argument('--mediator',
                        dest="mediator",
                        type=str,
                        required=True,
                        help='Which variable as a mediator '
                        '(e.g. map_auc_g_both)')
    parser.add_argument('--task',
                        dest="task",
                        type=str,
                        required=True,
                        choices=['both', 'stroop', 'msit'],
                        help='Which task used as input')
    # parser.add_argument('--model',
    #                     type=str,
    #                     required=True,
    #                     choices=['ridge', 'lasso', 'elasticnet'],
    #                     help='Which penalized PC Regression model to run')
    parser.add_argument('--n_boots',
                        type=int,
                        default=5000,
                        help="Number of bootstraps"
                        )
    parser.add_argument('--output_dir',
                        type=str,
                        help="Name for the output directory")

    opts = parser.parse_args()
    
    y_var = opts.target  # e.g "mavg_bulf_ccaf"
    m_var = opts.mediator  # e.g "map_auc_g_both"
    run_task = opts.task  # e.g "both"

    if opts.output_dir:
        output_dir = opj(project_dir, opts.output_dir)
    else:
        res_name = f"Y-{y_var}_M-{m_var}_task-{run_task}"
        output_dir = opj(project_dir, "results", res_name,
                         "bootstrapping", opts.model)
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    

    print(f"Running experiment with Y: {y_var}, M: {m_var}, "
          f"using {run_task} contrast maps, "
          f"with model: Ridge") #{opts.model}")

    # We are returning IDS, for later controlling predictions for covariates
    X, y, M, study, ids = load_data(y_var, m_var, run_task, return_ids=True)

    # Create digits for according to quartile
    y_digits = np.digitize(y, np.quantile(y, np.arange(0, 1, 0.25)))
    m_digits = np.digitize(M, np.quantile(M, np.arange(0, 1, 0.25)))

    # Stratify along m, y and stutdy.
    labels = pd.DataFrame(m_digits.astype(str)) + \
        "_" + pd.DataFrame(y_digits.astype(str)) + \
        "_" + pd.DataFrame(study)
    labels = LabelEncoder().fit_transform(labels.to_numpy().flatten())

    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    mediation_model = L2MediationModel(my=False, # We don't need this path
                                       cv = cv_inner, 
                                       alphas=1000,
                                       ridge_kws = {'max_iter': int(1e6)},
                                       n_jobs=-1)

    n_boots = opts.n_boots  # e.g. 1000
#    n_boots = 5
    #i_boot = 0
    batch_size=10
    
    # if os.path.exists(opj(output_dir, "status.json")) is False:
    #     boots_ok = 0
    #     init_boot = 0
        
    #     with open(opj(output_dir, "status.json"), 'w') as file:
    #         json.dump({'boots_ok' : boots_ok, 
    #                    'init_boot': init_boot}, 
    #                   file)
    # else:
    #     with open(opj(output_dir, "status.json"), 'r') as file:
    #         data_status = json.load(file)
    #         boots_ok = data_status['boots_ok']
    #         init_boot = data_status['init_boot']

    if os.path.exists(opj(output_dir, "status.json")) is False:
        boots_ok = 0
        i_boot = 0
        
        with open(opj(output_dir, "status.json"), 'w') as file:
            json.dump({'boots_ok' : boots_ok, 
                       'i_boot': i_boot}, 
                      file)
    else:
        with open(opj(output_dir, "status.json"), 'r') as file:
            data_status = json.load(file)
            boots_ok = data_status['boots_ok']
            i_boot = data_status['i_boot']

    while (boots_ok < n_boots):
        print(f"i_boot: {i_boot}, boots_ok: {boots_ok}")
        # completed = Parallel(n_jobs=2)(delayed(fit_mediation)(i_boot,
        #                                                       X,
        #                                                       M,
        #                                                       y,
        #                                                       study,
        #                                                       labels,
        #                                                       mediation_model,
        #                                                       output_dir
        #                                                       ) for i_boot \
        #                                in range(init_boot, init_boot+batch_size))
            
        completed = fit_mediation(i_boot, X, M, y, study, labels,
                                   mediation_model, output_dir)
        if completed:
            boots_ok +=1
        i_boot +=1
        
        if i_boot % batch_size == 0:
            with open(opj(output_dir, "status.json"), 'w') as file:
                json.dump({'boots_ok' : boots_ok, 
                           'i_boot': i_boot}, 
                          file)
                
        # boots_ok += sum(completed)
        # init_boot += batch_size
        # print("done", boots_ok, "status", init_boot)
        
        # with open(opj(output_dir, "status.json"), 'w') as file:
        #     json.dump({'boots_ok' : boots_ok, 
        #                'init_boot': init_boot}, 
        #               file)




if __name__ == "__main__":
    sys.exit(main())
