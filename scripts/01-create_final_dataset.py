#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script creates the project data by using only those subjects with 
available fMRI data.

"""

import pandas as pd
from os.path import join as opj
from glob import glob
from pathlib import Path
from datetime import datetime

project_dir = Path(__file__).resolve().parent.parent.as_posix()

project_data = pd.read_csv(opj(project_dir, "data/NOAH_PIP_w_reactivities_AUC_Dec22.csv")

pip_stroop_ids = [
    folder.split("/")[-3] for folder in
    glob(opj(project_dir,
             "data/PIP/First_Level/*/Stroop/smoothed_con_0003.nii"))
    ]

pip_msit_ids = [
    folder.split("/")[-3] for folder in
    glob(opj(project_dir,
             "data/PIP/First_Level/*/MSIT/smoothed_con_0003.nii"))
    ]


pip_ids = list(set(pip_stroop_ids) & set(pip_msit_ids))
pip_ids = ['P' + str(x) for x in pip_ids]

noah_stroop_ids = [
    folder.split("/")[-3] for folder in
    glob(opj(project_dir,
             "data/NOAH/First_Level/*/Stroop/smoothed_con_0003.nii"))
    ]
noah_msit_ids = [
    folder.split("/")[-3] for folder in
    glob(opj(project_dir,
             "data/NOAH/First_Level/*/Stroop/smoothed_con_0003.nii"))
         ]

noah_ids = list(set(noah_stroop_ids) & set(noah_msit_ids))

project_data = pd.merge(
    pd.DataFrame({'ID': sorted(pip_ids) + sorted(noah_ids)}),
    project_data, on='ID')

outfile = f"final_data_{datetime.today().strftime('%Y-%m-%d')}.csv"
project_data.to_csv(opj(project_dir, "data", outfile), index=False)
