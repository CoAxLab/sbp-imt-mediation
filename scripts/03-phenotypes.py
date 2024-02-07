#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This script just saves the model fitted on the entire data.
This is later used to calculate and plot the phenotypes.

"""

# IMPORTS
import numpy as np
import pandas as pd
import argparse
import sys
from pathlib import Path
from os.path import join as opj
from joblib import dump

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
#from my_sklearn_tools.model_selection import StratifiedKFoldReg

project_dir = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(project_dir)
from src.input_data import load_data
from src.harmonization import norm_standardize_my, harmonize_cohorts

from src.models import L2MediationModel, L1MediationModel

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
                        default="sbp_auc_g_both",
                        help='Which variable as a mediator '
                        '(e.g. sbp_auc_g_both)')
    parser.add_argument('--task',
                        dest="task",
                        type=str,
                        default="both",
                        choices=['both', 'stroop', 'msit'],
                        help='Which task used as input')
    parser.add_argument('--model',
                        type=str,
                        default="ridge",
                        choices=['ridge', 'lasso'],
                        help='Which penalized PC Regression model to run')
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

    y_var = opts.target  # e.g "mavg_bulbf_ccaf"
    m_var = opts.mediator  # e.g "sbp_auc_g_both"
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

    cv_inner = StratifiedKFold(n_splits=5, shuffle=True,
                                  random_state=123)

    y_ss = norm_standardize_my(y, study)
    print("-----------------------")

    print("Checking M...")
    m_ss = norm_standardize_my(M, study)

    print("-----------------------")

    X_combat = harmonize_cohorts(X, study)
    print("-----------------------")

    del X, y, M

    print("Fitting mediation model")
    print("-----------------------")
    mediation_model = Model(cv=cv_inner, n_jobs=-1, **model_kws)
    mediation_model.fit(X=X_combat, y=y_ss, m=m_ss)


    # Save data for encoding weights computation
    np.savez_compressed(opj(output_dir, 'data.npz'),
                        X=X_combat, y=y_ss, m=m_ss)

    # Save fitted models
    dump(mediation_model, opj(output_dir, 'mediation_model.joblib'))

if __name__ == "__main__":
    sys.exit(main())
