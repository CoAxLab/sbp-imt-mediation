import numpy as np

def norm_standardize_my(var_train, var_test, study_train, study_test,
                        verbose=True):

    from sklearn.preprocessing import PowerTransformer
    from scipy.stats import ks_2samp, ttest_ind, bartlett

    method='box-cox'
    if np.any(var_train<0) or np.any(var_test<0):
        method='yeo-johnson'

    pw_pip = PowerTransformer(method=method, standardize=True)
    pw_noah = PowerTransformer(method=method, standardize=True)

    is_pip_train = study_train == "PIP"
    is_noah_train = study_train == "NOAH"

    is_pip_test = study_test == "PIP"
    is_noah_test = study_test == "NOAH"

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

    var_test_ss = pw_pip.transform(var_test[:, None]).flatten()*is_pip_test \
        + pw_noah.transform(var_test[:, None]).flatten()*is_noah_test

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

    return var_train_ss, var_test_ss


def harmonize_cohorts(X_train, X_test, study_train, study_test):

    from pycombat import Combat
    from sklearn.preprocessing import LabelBinarizer

    if X_train.shape[1] < 2:
        raise ValueError("Combat needs at least two features")

    le = LabelBinarizer()
    pycombat = Combat()

    c_train = le.fit_transform(study_train).flatten()
    c_test = le.transform(study_test).flatten()

    # harmonise X with Comabat
    X_train_combat = pycombat.fit_transform(X_train, c_train)
    X_test_combat = pycombat.transform(X_test, c_test)

    return X_train_combat, X_test_combat