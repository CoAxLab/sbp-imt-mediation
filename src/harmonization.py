#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the functions used to harmonize the data.
"""

import numpy as np


def norm_standardize_my(var_train, study_train,
                        var_test=None, study_test=None,
                        verbose=False):
    """Harmonize by standardization by STUDY. For M and Y"""

    from sklearn.preprocessing import PowerTransformer
    from scipy.stats import ks_2samp, ttest_ind, bartlett

    method = 'box-cox'
    if np.any(var_train < 0):
        method = 'yeo-johnson'

    pw_pip = PowerTransformer(method=method, standardize=True)
    pw_noah = PowerTransformer(method=method, standardize=True)

    is_pip_train = study_train == "PIP"
    is_noah_train = study_train == "NOAH"

    if verbose:
        print("before zscore (KS test), p:",
              ks_2samp(var_train[is_pip_train], var_train[is_noah_train])[1])
        print("before zscore (t-test), p:",
              ttest_ind(var_train[is_pip_train], var_train[is_noah_train])[1])
        print("before zscore (bartlett), p:",
              bartlett(var_train[is_pip_train], var_train[is_noah_train])[1])

    pw_pip.fit(var_train[is_pip_train][:, None])
    pw_noah.fit(var_train[is_noah_train][:, None])

    var_train_ss = pw_pip.transform(var_train[:, None]).flatten()*is_pip_train\
        + pw_noah.transform(var_train[:, None]).flatten()*is_noah_train

    if verbose:
        print("after zscore (KS test) p:",
              ks_2samp(var_train_ss[is_pip_train],
                       var_train_ss[is_noah_train])[1]
              )
        print("after zscore (t-test), p:",
              ttest_ind(var_train_ss[is_pip_train],
                        var_train_ss[is_noah_train])[1]
              )
        print("after zscore (bartlett), p:",
              bartlett(var_train_ss[is_pip_train],
                       var_train_ss[is_noah_train])[1]
              )

    out = var_train_ss

    if var_test is not None and study_test is not None:
        is_pip_test = study_test == "PIP"
        is_noah_test = study_test == "NOAH"

        var_test_ss = pw_pip.transform(
            var_test[:, None]).flatten()*is_pip_test \
            + pw_noah.transform(var_test[:, None]).flatten()*is_noah_test

        out = (var_train_ss, var_test_ss)

    return out


def harmonize_cohorts(X_train, study_train, X_test=None, study_test=None):
    """Harmonize by Combat by STUDY. For X"""
    from pycombat import Combat
    from sklearn.preprocessing import LabelBinarizer

    if X_train.shape[1] < 2:
        raise ValueError("Combat needs at least two features")

    le = LabelBinarizer()
    pycombat = Combat()

    c_train = le.fit_transform(study_train).flatten()

    # harmonise X with Comabat
    X_train_combat = pycombat.fit_transform(X_train, c_train)

    out = X_train_combat

    if X_test is not None and study_test is not None:

        c_test = le.transform(study_test).flatten()
        X_test_combat = pycombat.transform(X_test, c_test)

        out = (X_train_combat, X_test_combat)

    return out
