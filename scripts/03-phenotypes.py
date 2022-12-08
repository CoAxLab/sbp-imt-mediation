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
from pathlib import Path
from os.path import join as opj
from jolib import dump

from sklearn.preprocessing import LabelEncoder

from my_sklearn_tools.model_selection import StratifiedKFoldReg

project_dir = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(project_dir)
from src.input_data import load_data
from src.harmonization import norm_standardize_my, harmonize_cohorts
from src.models import (L1Model_XY, L2Model_XY, L1L2Model_XY,
                        L1Model_XMY, L2Model_XMY, L1L2Model_XMY)


def main():

    parser = argparse.ArgumentParser(description='Run a particular experiment')
    parser.add_argument('--target',
                        dest="target",
                        type=str,
                        default="mavg_bulbf_ccaf",
                        help='Which variable as a target (e.g. mavg_bulbf_ccaf)')
    parser.add_argument('--mediator',
                        dest="mediator",
                        type=str,
                        default="map_auc_g_both",
                        help='Which variable as a mediator '
                        '(e.g. map_auc_g_both)')
    parser.add_argument('--task',
                        dest="task",
                        type=str,
                        default="both",
                        choices=['both', 'stroop', 'msit'],
                        help='Which task used as input')
    parser.add_argument('--model',
                        type=str,
                        default="ridge",
                        choices=['ridge', 'lasso', 'elasticnet'],
                        help='Which penalized PC Regression model to run')
    parser.add_argument('--output_dir',
                        type=str,
                        help="Name for the output directory")

    opts = parser.parse_args()

    # Choose models
    if opts.model == 'ridge':
        Model = L2Model_XY
        Model_XMY = L2Model_XMY
        model_kws = {'ridge_kws': {'max_iter': int(1e6)}, 'alphas': 1000}
    elif opts.model == 'lasso':
        Model = L1Model_XY
        Model_XMY = L1Model_XMY
        model_kws = {'lasso_kws': {'max_iter': int(1e6)}, 'n_alphas': 1000}
    else:
        Model = L1L2Model_XY
        Model_XMY = L1L2Model_XMY
        model_kws = {'elasticnet_kws': {'max_iter': int(1e6)},
                     'n_alphas': 1000}

    y_var = opts.target  # e.g "mavgimt"
    m_var = opts.mediator  # e.g "map_auc_g_both"
    run_task = opts.task  # e.g "both"

    if opts.output_dir:
        output_dir = opj(project_dir, opts.output_dir)
    else:
        res_name = f"Y-{y_var}_M-{m_var}_task-{run_task}"
        output_dir = opj(project_dir, "results", res_name,
                         "phenotypes", opts.model)
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    print(f"Running experiment with Y: {y_var}, M: {m_var}, "
          f"using {run_task} contrast maps, "
          f"with model: {opts.model}")

    X, y, M, study = load_data(y_var, m_var, run_task)

    # Create digits for according to quartile
    y_digits = np.digitize(y, np.quantile(y, np.arange(0, 1, 0.25)))
    m_digits = np.digitize(M, np.quantile(M, np.arange(0, 1, 0.25)))

    # Stratify along m, y and stutdy.
    labels = pd.DataFrame(m_digits.astype(str)) + \
        "_" + pd.DataFrame(y_digits.astype(str)) + \
        "_" + pd.DataFrame(study)
    labels = LabelEncoder().fit_transform(labels.to_numpy().flatten())

    cv_inner = StratifiedKFoldReg(n_splits=5, shuffle=True,
                                  random_state=123)

    y_ss = norm_standardize_my(y, study)
    print("-----------------------")

    print("Checking M...")
    m_ss = norm_standardize_my(M, study)

    print("-----------------------")

    X_combat = harmonize_cohorts(X, study)
    print("-----------------------")

    del X, y, M

    print("Doing X ---> Y")
    print("-----------------------")
    model_xy = Model(cv=cv_inner,
                     n_jobs=-1,
                     **model_kws
                     )
    model_xy.fit(X_combat, y_ss)

    print("Doing X ---> M")
    print("-----------------------")
    model_xm = Model(cv=cv_inner,
                     n_jobs=-1,
                     **model_kws
                     )

    model_xm.fit(X_combat, m_ss)

    print("Doing X + M ---> Y")
    print("-----------------------")
    model_xmy = Model_XMY(cv=cv_inner,
                          n_jobs=-1,
                          **model_kws
                          )
    XM_clean = np.column_stack((X_combat, m_ss))
    model_xmy.fit(XM_clean, y_ss)

    # Save data for encoding weights computation
    np.savez_compressed(opj(output_dir, 'data.npz'),
                        X=X_combat, y=y_ss, m=m_ss)

    # Save fitted models
    dump(model_xy, opj(output_dir, 'model_xy.joblib'))
    dump(model_xm, opj(output_dir, 'model_xm.joblib'))
    dump(model_xmy, opj(output_dir, 'model_xmy.joblib'))


if __name__ == "__main__":
    sys.exit(main())
