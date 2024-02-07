#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the functions to load the data.
"""

import numpy as np
import pandas as pd
from os.path import join as opj
from pathlib import Path


def load_data(y_var, m_var, run_task, mask_img=None, return_ids=False, fwd=None):

    from nilearn.input_data import NiftiMasker

    project_dir = Path(__file__).resolve().parent.parent.as_posix()

    if mask_img is None:
        mask_img = opj(project_dir, "data/resliced_grey25grey25.nii")

    nifti_masker = NiftiMasker(mask_img=mask_img)

    project_data = pd.read_csv(
        opj(project_dir, "data/final_data_2023-01-25_w10cvd.csv")
        )
    if fwd:
        try:
            fwd = float(fwd)
        except ValueError as e:
            raise("Error reading the amount of framewise displacement")

        print("filtering subjects with fwd greater"
              f" than {fwd} in either Stroop or MSIT")

        cond_fwd_stroop = project_data.fwd_stroop > fwd
        cond_fwd_msit = project_data.fwd_msit > fwd
        cond_fwd = cond_fwd_stroop | cond_fwd_msit
        print(f"{sum(cond_fwd)} subjects have fwd > {fwd}")
        project_data = project_data.loc[~cond_fwd, :]

    run_data = project_data.loc[:, ['ID', 'STUDY', y_var,  m_var]].dropna()

    print(f"For this scenario, we have {run_data.shape[0]} observations")

    # Construct dataset
    study = run_data.STUDY.to_numpy()
    y = run_data.loc[:, y_var].to_numpy()
    M = run_data.loc[:, m_var].to_numpy()
    ids = run_data.loc[:, 'ID'].to_numpy()

    stroop_files = []
    msit_files = []
    for _, row in run_data.iterrows():

        if row.STUDY == "PIP":
            stroop_files.append(
                opj(project_dir,
                    f"data/PIP/First_Level/{row.ID[1:]}/Stroop/smoothed_con_0003.nii")
                )
            msit_files.append(
                opj(project_dir,
                    f"data/PIP/First_Level/{row.ID[1:]}/MSIT/smoothed_con_0003.nii")
                )
        else:
            stroop_files.append(
                opj(project_dir,
                    f"data/NOAH/First_Level/{row.ID}/Stroop/smoothed_con_0003.nii")
                )
            msit_files.append(
                opj(project_dir,
                    f"data/NOAH/First_Level/{row.ID}/MSIT/smoothed_con_0003.nii")
                )

    if run_task == "both":
        X_stroop = nifti_masker.fit_transform(stroop_files)
        X_msit = nifti_masker.fit_transform(msit_files)
        X = np.mean([X_stroop, X_msit], axis=0)
    elif run_task == "stroop":
        X = nifti_masker.fit_transform(stroop_files)
    elif run_task == "msit":
        X = nifti_masker.fit_transform(msit_files)

    print('X:', X.shape, 'y:', y.shape, 'M:', M.shape)

    if return_ids:
        out = (X, y, M, study, ids)
    else:
        out = (X, y, M, study)

    return out
