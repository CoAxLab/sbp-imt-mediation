#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script just runs predicitons filtering by average 
framewise displacement when loading the data.


"""

# IMPORTS
import numpy as np
import pandas as pd
import argparse
import sys
from pathlib import Path
from os.path import join as opj

from scipy.stats import ks_2samp

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
# from my_sklearn_tools.model_selection import (StratifiedKFoldReg,
#                                               RepeatedStratifiedKFoldReg)

project_dir = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(project_dir)
from src.input_data import load_data
from src.harmonization import norm_standardize_my, harmonize_cohorts
from src.models import L1MediationModel, L2MediationModel


def main():

    parser = argparse.ArgumentParser(description='Run a particular experiment filtering by average framewise displacement')
    parser.add_argument('--target',
                        dest="target",
                        type=str,
                        default="mavg_bulbf_ccaf",
                        help='Which variable as a target (default: mavg_bulbf_ccaf)')
    parser.add_argument('--mediator',
                        dest="mediator",
                        type=str,
                        default="sbp_auc_g_both",
                        help='Which variable as a mediator '
                        '(default: sbp_auc_g_both)')
    parser.add_argument('--task',
                        dest="task",
                        type=str,
                        default="both",
                        choices=['both', 'stroop', 'msit'],
                        help='Which task used as input (default: both)')
    parser.add_argument('--model',
                        type=str,
                        default="ridge",
                        choices=['ridge', 'lasso'],
                        help='Which penalized PC Regression model to run (default: ridge)')
    parser.add_argument('--fwd',
                        type=float,
                        default=0.5,
                        help='Amount of framewise displacement above which filtering subjectss')
    parser.add_argument('--output_dir',
                        type=str,
                        help="Name for the output directory")

    opts = parser.parse_args()

    # Choose models
    if opts.model == 'ridge':
        Model = L2MediationModel
        model_kws = {'ridge_kws': {'max_iter': int(1e6)}, 'alphas': 1000}
    else:
        Model = L1MediationModel
        model_kws = {'lasso_kws': {'max_iter': int(1e6)}, 'n_alphas': 1000}

    y_var = opts.target  # e.g "mavgimt"
    m_var = opts.mediator  # e.g "sbp_auc_g_both"
    run_task = opts.task  # e.g "both"
    fwd = opts.fwd

    if opts.output_dir:
        output_dir = opj(project_dir, opts.output_dir)
    else:
        res_name = f"Y-{y_var}_M-{m_var}_task-{run_task}"
        output_dir = opj(project_dir, "results", res_name,
                         "sensitivity_analysis", opts.model)
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    print(f"Running sensitivity analysis with Y: {y_var}, M: {m_var}, "
          f"using {run_task} contrast maps, "
          f"with model: {opts.model} and fwd: {fwd}")

    # We are returning IDS, for later controlling predictions for covariates
    X, y, M, study, ids = load_data(y_var, m_var, run_task, return_ids=True, fwd=fwd)

    # Create digits for according to quartile
    y_digits = np.digitize(y, np.quantile(y, np.arange(0, 1, 1/3)))
    m_digits = np.digitize(M, np.quantile(M, np.arange(0, 1, 1/3)))

    # Stratify along m, y and stutdy.
    labels = pd.DataFrame(m_digits.astype(str)) + \
        "_" + pd.DataFrame(y_digits.astype(str)) + \
        "_" + pd.DataFrame(study)
    labels = LabelEncoder().fit_transform(labels.to_numpy().flatten())

    # Cross-validation procedures:
    # 5 Times 10-Fold CV as outer CV, 5-Fold for inner.
    cv_outer = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, 
                                       random_state=1234)
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    # Dictionary to store the results
    res_xm = {'y_trues': [], 'y_preds': [], 'coefs': []}
    res_my = {'y_trues': [], 'y_preds': []}

    res_xy = {'y_trues': [], 'y_preds': [], 'coefs': []}
    res_xmy = {'y_trues': [], 'y_preds': [], 'coefs': []}

    fold_ids = {'train_ids': [], 'test_ids': []}

    ifold = 1
    for train_index, test_index in cv_outer.split(np.zeros(len(labels)),
                                                  labels):

        print(f"FOLD: {ifold}")
        print("----------------")
        print("----------------")

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        m_train, m_test = M[train_index], M[test_index]
        study_train, study_test = study[train_index], study[test_index]

        # Save subject IDs in each partition
        fold_ids['train_ids'].append(ids[train_index])
        fold_ids['test_ids'].append(ids[test_index])

        # Check stratification
        print("Checking partition...")
        print("-----------------------")
        print("No differentces in M? ", ks_2samp(m_train, m_test)[1] > 0.05)
        print("No differentces in y? ", ks_2samp(y_train, y_test)[1] > 0.05)

        print("Checking Y...")
        print("-----------------------")
        y_train_ss, y_test_ss = norm_standardize_my(y_train, study_train,
                                                    y_test, study_test)

        print("-----------------------")

        print("Checking M...")
        m_train_ss, m_test_ss = norm_standardize_my(m_train, study_train,
                                                    m_test, study_test)

        print("-----------------------")

        X_train_combat, X_test_combat = harmonize_cohorts(X_train, study_train,
                                                          X_test, study_test)

        # This is to ensure that I didn't screw up and used no standardised/
        # harmonised data.
        del X_train, X_test, y_train, y_test, m_train, m_test

        print("-----------------------")

        mediation_model = Model(cv=cv_inner,
                                n_jobs=-1,
                                **model_kws
                                )
        
        mediation_model.fit(X=X_train_combat, y=y_train_ss, m=m_train_ss)
        pred_dict = mediation_model.predict(X=X_test_combat, m=m_test_ss)
        
        res_xy['y_trues'].append(y_test_ss)
        res_xy['y_preds'].append(pred_dict['xy'])
        res_xy['coefs'].append(mediation_model.coef_['xy'])

        res_xm['y_trues'].append(m_test_ss)
        res_xm['y_preds'].append(pred_dict['xm'])
        res_xm['coefs'] = mediation_model.coef_['xm']

        res_my['y_trues'].append(y_test_ss)
        res_my['y_preds'].append(pred_dict['my'])

        res_xmy['y_trues'].append(y_test_ss)
        res_xmy['y_preds'].append(pred_dict['xmy'])
        res_xmy['coefs'].append(mediation_model.coef_['xmy'])

        ifold += 1

    # Save results
    np.savez(opj(output_dir, 'model_xy.npz'), **res_xy)
    np.savez(opj(output_dir, 'model_xm.npz'), **res_xm)
    np.savez(opj(output_dir, 'model_my.npz'), **res_my)
    np.savez(opj(output_dir, 'model_xmy.npz'), **res_xmy)
    np.savez(opj(output_dir, 'fold_subject_ids.npz'), **fold_ids)


if __name__ == "__main__":
    sys.exit(main())
