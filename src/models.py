#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file contains the two classess for running 
a mediation framework. Each class basically runs the 
four paths of the mediation framework. They only differ
in that one uses Ridge and the other, Lasso.

"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.compose import ColumnTransformer
from sklearn.metrics import get_scorer
from sklearn.linear_model._coordinate_descent import _alpha_grid


from my_sklearn_tools.pca_regressors import LassoPCR, ElasticNetPCR, RidgePCR
from my_sklearn_tools.model_selection import check_cv

class BaseModel_XY():

    def _get_pca(self):
    
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        scaler = StandardScaler(with_std=True, with_mean=False)
                
        return make_pipeline(*super()._get_pca().named_steps.values(), 
                             scaler)
    
    def get_weights(self):

        check_is_fitted(self)
        
        # Get scales. This is just to return weights to their original
        # scales. We can't use the inversetransform of SS, because that
        # would the mean, and the mean should only affects the intercept.
        ss_scale = self.best_estimator_.\
            named_steps['standardscaler-2'].scale_

        # Here it's just essentially multiplying by the mixing V matrix,
        # given that we start from the demeaned data...
        pca_inv_transf = self.best_estimator_.\
            named_steps['pca'].inverse_transform
        vt_1_inv_transf = self.best_estimator_.\
            named_steps['variancethreshold-1'].inverse_transform
        vt_2_inv_transf = self.best_estimator_.\
            named_steps['variancethreshold-2'].inverse_transform

        beta = self.best_estimator_[-1].coef_

        if beta.ndim == 1:
            beta = beta[None, :]
            
        # Return unscaled beta coefficients
        beta_ss = beta/ss_scale
        
        # We are computing the weights for the centered or scaled data,
        # that's why we don't transform with the StandardScaler step
        w = vt_1_inv_transf(pca_inv_transf(vt_2_inv_transf(beta_ss)))

        # Return weights to original units if we had scaled the data before the PCA..
        if self.scale:
            w = w/self.best_estimator_.named_steps['standardscaler-1'].scale_

        if w.shape[0] == 1:
            w = w[0, :]
        return w


class L2Model_XY(BaseModel_XY, RidgePCR):
    pass

class L1Model_XY(BaseModel_XY, LassoPCR):
    pass

class L1L2Model_XY(BaseModel_XY, ElasticNetPCR):
    pass

class BaseModel_XMY():

    def _get_pca(self):

        # Here it comes the new transformation of the input data before
        # running the regression model. Since this class is intended for
        # X+M->Y, we just want to PCA transform X. This can be easily
        # accomplished using ColumnTransformer from sklearn

        from sklearn.preprocessing import StandardScaler
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import make_pipeline

        # This is the usual scaling before the regression model. It's
        # important because X and M will have likely different units
        # and ranges.
        scale = StandardScaler()

        # Grab the original transformation from the parent Class. Then,
        # by using slice(None, -1) we are saying that this tranformation
        # will be applied only on X.

        pca_transform = super()._get_pca()
        transformer = ColumnTransformer(
            transformers=[("pc_transf", pca_transform, slice(None, -1))],
            remainder="passthrough"
        )
        return make_pipeline(transformer, scale)

    def get_weights(self):

        from sklearn.utils.validation import check_is_fitted

        check_is_fitted(self)

        # Get scales. This is just to return weights to their original
        # scales. We can't use the inversetransform of SS, because that
        # would the mean, and the mean should only affects the intercept.
        ss_scale = self.best_estimator_.\
            named_steps['standardscaler'].scale_

        pca_inv_transf = self.best_estimator_.\
            named_steps['columntransformer'].\
            named_transformers_['pc_transf'].\
            named_steps['pca'].inverse_transform

        vt_1_inv_transf = self.best_estimator_.\
            named_steps['columntransformer'].\
            named_transformers_['pc_transf'].\
            named_steps['variancethreshold-1'].inverse_transform

        vt_2_inv_transf = self.best_estimator_.\
            named_steps['columntransformer'].\
            named_transformers_['pc_transf'].\
            named_steps['variancethreshold-2'].inverse_transform

        # Beta coefficients in PC and M space.
        beta = self.best_estimator_[-1].coef_

        if beta.ndim == 1:
            beta = beta[None, :]

        # Return unscaled beta coefficients
        beta_ss = beta/ss_scale
        beta_pc_ss = beta_ss[:, :-1]
        w_m_ss = beta_ss[:, -1]

        # We are computing the weights for the centered or scaled data,
        # that's why we don't transform with the StandardScaler step
        w_pca_ss = vt_1_inv_transf(
            pca_inv_transf(
                vt_2_inv_transf(beta_pc_ss)
                )
            )
        w = np.column_stack((w_pca_ss, w_m_ss))

        if w.shape[0] == 1:
            w = w[0, :]
        return w


class L2Model_XMY(BaseModel_XMY, RidgePCR):
    pass

class L1Model_XMY(BaseModel_XMY, LassoPCR):
    pass

class L1L2Model_XMY(BaseModel_XMY, ElasticNetPCR):
    pass

def get_xmy(Model):

    class Model_XMY(Model):

        # This is the full model class. We can inherit from our original models and
        # redefine how the transformation operation, given that we want the PCA
        # to be applied only to X. We also need to redefine the weights computation
        # on the original space.

            def _get_pca(self):

                # Here it comes the new transformation of the input data before
                # running the regression model. Since this class is intended for
                # X+M->Y, we just want to PCA transform X. This can be easily
                # accomplished using ColumnTransformer from sklearn

                from sklearn.preprocessing import StandardScaler
                from sklearn.compose import ColumnTransformer
                from sklearn.pipeline import make_pipeline

                # This is the usual scaling before the regression model. It's
                # important because X and M will have likely different units
                # and ranges.
                scale = StandardScaler()

                # Grab the original transformation from the parent Class. Then,
                # by using slice(None, -1) we are saying that this tranformation
                # will be applied only on X.

                pca_transform = super()._get_pca()
                transformer = ColumnTransformer(
                    transformers=[("pc_transf", pca_transform, slice(None, -1))],
                    remainder="passthrough"
                )
                return make_pipeline(transformer, scale)

            def get_weights(self):

                from sklearn.utils.validation import check_is_fitted

                check_is_fitted(self)

                # Get scales. This is just to return weights to their original
                # scales. We can't use the inversetransform of SS, because that
                # would the mean, and the mean should only affects the intercept.
                ss_scale = self.best_estimator_.\
                    named_steps['standardscaler'].scale_

                pca_inv_transf = self.best_estimator_.\
                    named_steps['columntransformer'].\
                    named_transformers_['pc_transf'].\
                    named_steps['pca'].inverse_transform

                vt_1_inv_transf = self.best_estimator_.\
                    named_steps['columntransformer'].\
                    named_transformers_['pc_transf'].\
                    named_steps['variancethreshold-1'].inverse_transform

                vt_2_inv_transf = self.best_estimator_.\
                    named_steps['columntransformer'].\
                    named_transformers_['pc_transf'].\
                    named_steps['variancethreshold-2'].inverse_transform

                # Beta coefficients in PC and M space.
                beta = self.best_estimator_[-1].coef_

                if beta.ndim == 1:
                    beta = beta[None, :]

                # Return unscaled beta coefficients
                beta_ss = beta/ss_scale
                beta_pc_ss = beta_ss[:, :-1]
                w_m_ss = beta_ss[:, -1]

                # We are computing the weights for the centered or scaled data,
                # that's why we don't transform with the StandardScaler step
                w_pca_ss = vt_1_inv_transf(
                    pca_inv_transf(
                        vt_2_inv_transf(beta_pc_ss)
                        )
                    )
                w = np.column_stack((w_pca_ss, w_m_ss))

                if w.shape[0] == 1:
                    w = w[0, :]
                return w

    return Model_XMY

class BaseMediationModel():
    
    def transform_X(self):

        vt_1 = VarianceThreshold()
        vt_2 = VarianceThreshold(threshold=1e-20)
        ss_1 = StandardScaler(with_std=self.scale)
        ss_2 = StandardScaler(with_mean=False, with_std=True)

        if self.pca_kws:
            pca = PCA(**self.pca_kws)
        else:
            pca = PCA()

        return make_pipeline(vt_1, ss_1, pca, vt_2, ss_2)

    def transform_XMY(self):

        # This is the usual scaling before the regression model. It's
        # important because X and M will have likely different units
        # and ranges.
        ss_2 = StandardScaler()

        # Grab the original transformation from the parent Class. Then,
        # by using slice(None, -1) we are saying that this tranformation
        # will be applied only on X.

        pca_transform = self.transform_X()[:-1]
        transformer = ColumnTransformer(
            transformers=[("pc_transf", pca_transform, slice(None, -1))],
            remainder="passthrough"
        )
        return make_pipeline(transformer, ss_2)
    
    def predict(self, *, X, m):

        check_is_fitted(self)
        
        XM = np.column_stack((X, m))
        
        preds = dict()
        
        if self.xy:
            preds['xy'] = self.best_estimator_['xy'].predict(X)
        if self.xy:
            preds['xm'] = self.best_estimator_['xm'].predict(X)    
        if self.xy:
            preds['my'] = self.best_estimator_['my'].predict(m[:, np.newaxis])    
        if self.xy:
            preds['xmy'] = self.best_estimator_['xmy'].predict(XM)

        return preds


class L2MediationModel(BaseEstimator, BaseMediationModel):

    def __init__(self,
                 xy=True,
                 xm=True,
                 my=True,
                 xmy=True,
                 scale = False,
                 cv=None,
                 alphas=100,
                 pca_kws=None,
                 ridge_kws=None,
                 scoring='neg_mean_squared_error',
                 n_jobs=None,
                 verbose=0
                 ):

        self.xy = xy
        self.xm = xm
        self.my = my
        self.xmy = xmy
        self.scale = scale
        self.cv = cv
        self.alphas = alphas
        self.pca_kws = pca_kws
        self.ridge_kws = ridge_kws
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, *, X, y, m):

        # Checkings here
        X, y = check_X_y(X, y)
        _, m = check_X_y(X, m)
        
        # Create digits for according to quartile
        y_digits = np.digitize(y, np.quantile(y, np.arange(0, 1, 0.25)))
        m_digits = np.digitize(m, np.quantile(m, np.arange(0, 1, 0.25)))

        # Stratify along m, y and stutdy.
        labels = pd.DataFrame(m_digits.astype(str)) + \
            "_" + pd.DataFrame(y_digits.astype(str)) 
        labels = LabelEncoder().fit_transform(labels.to_numpy().flatten())
        
        cv = check_cv(self.cv, labels, classifier=True)
        
        splits = list(cv.split(np.zeros(len(labels)), labels))

        X_transformer = self.transform_X()
        XMY_transformer = self.transform_XMY()

        if isinstance(self.alphas, int):
            alphas = np.logspace(-4, 4, self.alphas)
        else:
            alphas = self.alphas
        self.alphas_ = alphas

        if self.ridge_kws is None:
            ridge = Ridge()
        else:
            ridge = Ridge(**self.ridge_kws)

        ridge_kws = ridge.get_params()
        ridge_kws.pop('alpha')
        self.ridge_kws = ridge_kws

        estimators = [Ridge(alpha=alpha, **ridge_kws) for alpha in
                      self.alphas_]

        self.scorer_ = get_scorer(self.scoring)

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        
        # Generate list of cv scores
        list_scores_cv = parallel(
            delayed(self._cv_optimize)(
                estimators.copy(), X, y, m,
                train, val,
                clone(XMY_transformer),
                self.scorer_
                ) for train, val in splits)

        scores_cv_dict = {'xy': [], 'xm': [], 'xmy': []}

        for res_dict in list_scores_cv:
            scores_cv_dict['xy'].append(res_dict['xy'])
            scores_cv_dict['xm'].append(res_dict['xm'])
            scores_cv_dict['xmy'].append(res_dict['xmy'])

        scores_cv_dict['xy'] = np.column_stack(scores_cv_dict['xy'])
        scores_cv_dict['xm'] = np.column_stack(scores_cv_dict['xm'])
        scores_cv_dict['xmy'] = np.column_stack(scores_cv_dict['xmy'])
        self.scores_cv_ = scores_cv_dict

        # Compute mean scores in each model
        scores_cv_mean_dict = dict()
        for key, value in scores_cv_dict.items():
            scores_cv_mean_dict[key] = np.mean(value, axis=1)
            
        alpha_opt =  dict()
        for key, value in scores_cv_mean_dict.items():
            if getattr(self, key):
                alpha_opt[key] = self.alphas_[np.argmax(value)]
            else:
                alpha_opt[key] = None
        self.alpha_ = alpha_opt

        best_estimator_dict = dict()
        if self.xy:
            ridge_opt = Ridge(alpha=alpha_opt['xy'], **ridge_kws)
            pip_opt = clone(X_transformer)
            pip_opt.steps.append(make_pipeline(ridge_opt).steps[0])
            pip_opt.fit(X, y)
            best_estimator_dict['xy'] = pip_opt

        if self.xm:
            ridge_opt = Ridge(alpha=alpha_opt['xm'], **ridge_kws)
            pip_opt = clone(X_transformer)
            pip_opt.steps.append(make_pipeline(ridge_opt).steps[0])
            pip_opt.fit(X, m)
            best_estimator_dict['xm'] = pip_opt
                
        if self.my:
            linReg = LinearRegression()
            linReg.fit(m[:, np.newaxis], y)
            best_estimator_dict['my'] = linReg

        if self.xmy:
            ridge_opt = Ridge(alpha=alpha_opt['xmy'], **ridge_kws)
            pip_opt = clone(XMY_transformer)
            pip_opt.steps.append(make_pipeline(ridge_opt).steps[0])
            pip_opt.fit(np.column_stack((X,m)), y)
            best_estimator_dict['xmy'] = pip_opt

        self.best_estimator_ = best_estimator_dict
        
        intercept_dict = dict()
        coef_dict = dict()
        for model, estim in best_estimator_dict.items():
            if model != "my":
                estim = estim[-1] # Grab only Ridge part
            intercept_dict[model] = estim.intercept_
            coef_dict[model] = estim.coef_
        self.intercept_ = intercept_dict
        self.coef_ = coef_dict
        
        weights_dict = dict()
        for model, estim in best_estimator_dict.items():
            if model == "my": 
                continue
            elif model == "xmy":
                weights_dict[model] = get_weights_XMY(estim)
            else:
                weights_dict[model] = get_weights(estim)
        
        # weights_dict = dict()
        # weights_dict['xy'] = get_weights(best_estimator_dict['xy'])
        # weights_dict['xm'] = get_weights(best_estimator_dict['xm'])
        # weights_dict['xmy'] = get_weights_XMY(best_estimator_dict['xmy'])
        self.weights_ = weights_dict
        
        return self
  

    def _cv_optimize(self,
                     cv_estims,
                     X,
                     y,
                     m,
                     train,
                     val,
                     transf,
                     score):

        X_train, X_val = X[train], X[val]
        y_train, y_val = y[train], y[val]
        m_train, m_val = m[train], m[val]

        XM_train = np.column_stack((X_train, m_train))
        XM_val = np.column_stack((X_val, m_val))

        XM_train_trans = transf.fit_transform(XM_train)
        XM_val_trans = transf.transform(XM_val)

        # Generate list of estimators to fit
        if self.xy:
            fit_xy = [clone(estim).fit(XM_train_trans[:,:-1], y_train) 
                      for estim in cv_estims]
            scores_xy = [score(estim, XM_val_trans[:,:-1], y_val)
                         for estim in fit_xy]
        else:
            scores_xy = [None]
    
        if self.xm:
            fit_xm = [clone(estim).fit(XM_train_trans[:,:-1], m_train)
                      for estim in cv_estims]
            scores_xm = [score(estim, XM_val_trans[:,:-1], m_val)
                         for estim in fit_xm]
        else:
            scores_xm = [None]

        if self.xmy:
            fit_xmy = [clone(estim).fit(XM_train_trans, y_train)
                       for estim in cv_estims]
            scores_xmy = [score(estim, XM_val_trans, y_val)
                          for estim in fit_xmy]
        else:
            scores_xmy = [None]
    
        scores = {'xy': scores_xy, 'xm': scores_xm, 'xmy': scores_xmy}
        return scores

class L1MediationModel(BaseEstimator, BaseMediationModel):

    def __init__(self,
                 xy=True,
                 xm=True,
                 my=True,
                 xmy=True,
                 scale = False,
                 cv=None,
                 n_alphas=100,
                 alphas=None,
                 eps=0.001,
                 pca_kws=None,
                 lasso_kws=None,
                 scoring='neg_mean_squared_error',
                 n_jobs=None,
                 verbose=0
                 ):

        self.xy = xy
        self.xm = xm
        self.my = my
        self.xmy = xmy
        self.scale = scale
        self.cv = cv
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.eps = eps
        self.pca_kws = pca_kws
        self.lasso_kws = lasso_kws
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, *, X, y, m):

        # Checkings here
        X, y = check_X_y(X, y)
        _, m = check_X_y(X, m)
        
        # Create digits for according to quartile
        y_digits = np.digitize(y, np.quantile(y, np.arange(0, 1, 0.25)))
        m_digits = np.digitize(m, np.quantile(m, np.arange(0, 1, 0.25)))

        # Stratify along m, y and stutdy.
        labels = pd.DataFrame(m_digits.astype(str)) + \
            "_" + pd.DataFrame(y_digits.astype(str)) 
        labels = LabelEncoder().fit_transform(labels.to_numpy().flatten())
        
        cv = check_cv(self.cv, labels, classifier=True)
        
        splits = list(cv.split(np.zeros(len(labels)), labels))

        X_transformer = self.transform_X()
        XMY_transformer = self.transform_XMY()
        
        if self.lasso_kws is None:
            lasso = Lasso()
        else:
            lasso = Lasso(**self.lasso_kws)
        lasso_kws = lasso.get_params()
        lasso_kws.pop('alpha')
        self.lasso_kws = lasso_kws
        
        alphas_dict = dict()
        if self.alphas is None:
            X_transf = X_transformer.fit_transform(X)
            
            if self.xy:
                alphas_dict['xy'] = _alpha_grid(
                    X_transf, y, eps=self.eps, n_alphas=self.n_alphas,
                    fit_intercept=lasso_kws['fit_intercept'],
                    copy_X=lasso_kws['copy_X'])
            if self.xm:
                alphas_dict['xm'] = _alpha_grid(
                    X_transf, m, eps=self.eps, n_alphas=self.n_alphas,
                    fit_intercept=lasso_kws['fit_intercept'],
                    copy_X=lasso_kws['copy_X'])
            if self.xmy:
                X_transf = XMY_transformer.fit_transform(
                    np.column_stack((X, m)))
                alphas_dict['xmy'] = _alpha_grid(
                    X_transf, y, eps=self.eps, n_alphas=self.n_alphas,
                    fit_intercept=lasso_kws['fit_intercept'],
                    copy_X=lasso_kws['copy_X'])
        else:
            alphas = np.sort(self.alphas)[::-1]
            if self.xy:
                alphas_dict['xy'] = alphas
            if self.xm:
                alphas_dict['xm'] = alphas
            if self.xmy:
                alphas_dict['xmy'] = alphas

        self.alphas_ = alphas_dict
        
        estimators_dict = dict()
        for model, alphas in alphas_dict.items():
            estimators_dict[model] = [Lasso(alpha=alpha, **lasso_kws) 
                                      for alpha in alphas]

        self.scorer_ = get_scorer(self.scoring)

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        
        # Generate list of cv scores
        list_scores_cv = parallel(
            delayed(self._cv_optimize)(
                estimators_dict.copy(), X, y, m,
                train, val,
                clone(XMY_transformer),
                self.scorer_
                ) for train, val in splits)

        scores_cv_dict = {'xy': [], 'xm': [], 'xmy': []}

        for res_dict in list_scores_cv:
            scores_cv_dict['xy'].append(res_dict['xy'])
            scores_cv_dict['xm'].append(res_dict['xm'])
            scores_cv_dict['xmy'].append(res_dict['xmy'])

        scores_cv_dict['xy'] = np.column_stack(scores_cv_dict['xy'])
        scores_cv_dict['xm'] = np.column_stack(scores_cv_dict['xm'])
        scores_cv_dict['xmy'] = np.column_stack(scores_cv_dict['xmy'])
        self.scores_cv_ = scores_cv_dict
        
        # Compute mean scores in each model
        scores_cv_mean_dict = dict()
        for key, value in scores_cv_dict.items():
            scores_cv_mean_dict[key] = np.mean(value, axis=1)
            
        alpha_opt =  dict()
        for key, value in scores_cv_mean_dict.items():
            if getattr(self, key):
                alpha_opt[key] = self.alphas_[key][np.argmax(value)]
            else:
                alpha_opt[key] = None
        self.alpha_ = alpha_opt

        best_estimator_dict = dict()
        if self.xy:
            lasso_opt = Lasso(alpha=alpha_opt['xy'], **lasso_kws)
            pip_opt = clone(X_transformer)
            pip_opt.steps.append(make_pipeline(lasso_opt).steps[0])
            pip_opt.fit(X, y)
            best_estimator_dict['xy'] = pip_opt

        if self.xm:
            lasso_opt = Lasso(alpha=alpha_opt['xm'], **lasso_kws)
            pip_opt = clone(X_transformer)
            pip_opt.steps.append(make_pipeline(lasso_opt).steps[0])
            pip_opt.fit(X, m)
            best_estimator_dict['xm'] = pip_opt
                
        if self.my:
            linReg = LinearRegression()
            linReg.fit(m[:, np.newaxis], y)
            best_estimator_dict['my'] = linReg

        if self.xmy:
            lasso_opt = Lasso(alpha=alpha_opt['xmy'], **lasso_kws)
            pip_opt = clone(XMY_transformer)
            pip_opt.steps.append(make_pipeline(lasso_opt).steps[0])
            pip_opt.fit(np.column_stack((X,m)), y)
            best_estimator_dict['xmy'] = pip_opt

        self.best_estimator_ = best_estimator_dict
        
        intercept_dict = dict()
        coef_dict = dict()
        for model, estim in best_estimator_dict.items():
            if model != "my":
                estim = estim[-1] # Grab only Lasso part
            intercept_dict[model] = estim.intercept_
            coef_dict[model] = estim.coef_
        self.intercept_ = intercept_dict
        self.coef_ = coef_dict
        
        weights_dict = dict()
        for model, estim in best_estimator_dict.items():
            if model == "my": 
                continue
            elif model == "xmy":
                weights_dict[model] = get_weights_XMY(estim)
            else:
                weights_dict[model] = get_weights(estim)
            
        self.weights_ = weights_dict
        
        return self
    
    def _cv_optimize(self,
                     cv_estims,
                     X,
                     y,
                     m,
                     train,
                     val,
                     transf,
                     score):

        X_train, X_val = X[train], X[val]
        y_train, y_val = y[train], y[val]
        m_train, m_val = m[train], m[val]

        XM_train = np.column_stack((X_train, m_train))
        XM_val = np.column_stack((X_val, m_val))

        XM_train_trans = transf.fit_transform(XM_train)
        XM_val_trans = transf.transform(XM_val)

        # Generate list of estimators to fit
        if self.xy:
            estims = cv_estims['xy']
            fit_xy = [clone(estim).fit(XM_train_trans[:,:-1], y_train) 
                      for estim in estims]
            scores_xy = [score(estim, XM_val_trans[:,:-1], y_val)
                         for estim in fit_xy]
        else:
            scores_xy = [None]
    
        if self.xm:
            estims = cv_estims['xm']
            fit_xm = [clone(estim).fit(XM_train_trans[:,:-1], m_train)
                      for estim in estims]
            scores_xm = [score(estim, XM_val_trans[:,:-1], m_val)
                         for estim in fit_xm]
        else:
            scores_xm = [None]

        if self.xmy:
            estims = cv_estims['xmy']
            fit_xmy = [clone(estim).fit(XM_train_trans, y_train)
                       for estim in estims]
            scores_xmy = [score(estim, XM_val_trans, y_val)
                          for estim in fit_xmy]
        else:
            scores_xmy = [None]
    
        scores = {'xy': scores_xy, 'xm': scores_xm, 'xmy': scores_xmy}
        return scores


def get_weights(estimator):

    # Get scales. This is just to return weights to their original
    # scales. We can't use the inversetransform of SS, because that
    # would the mean, and the mean should only affects the intercept.
    ss_scale = estimator.named_steps['standardscaler-2'].scale_

    # Here it's just essentially multiplying by the mixing V matrix,
    # given that we start from the demeaned data...
    pca_inv_transf = estimator.named_steps['pca'].inverse_transform
    vt_1_inv_transf = estimator.\
        named_steps['variancethreshold-1'].inverse_transform
    vt_2_inv_transf = estimator.\
        named_steps['variancethreshold-2'].inverse_transform

    beta = estimator[-1].coef_

    if beta.ndim == 1:
        beta = beta[None, :]
        
    # Return unscaled beta coefficients
    beta_ss = beta/ss_scale
    
    # We are computing the weights for the centered or scaled data,
    # that's why we don't transform with the StandardScaler step
    w = vt_1_inv_transf(pca_inv_transf(vt_2_inv_transf(beta_ss)))

    # # Return weights to original units if we had scaled the data before the PCA..
    # if scale:
    #     w = w/estimator.named_steps['standardscaler-1'].scale_

    if w.shape[0] == 1:
        w = w[0, :]
    return w

def get_weights_XMY(estimator):

    # Get scales. This is just to return weights to their original
    # scales. We can't use the inversetransform of SS, because that
    # would the mean, and the mean should only affects the intercept.
    ss_scale = estimator.named_steps['standardscaler'].scale_

    pca_inv_transf = estimator.\
        named_steps['columntransformer'].\
        named_transformers_['pc_transf'].\
        named_steps['pca'].inverse_transform
            
    vt_1_inv_transf = estimator.\
        named_steps['columntransformer'].\
        named_transformers_['pc_transf'].\
        named_steps['variancethreshold-1'].inverse_transform

    vt_2_inv_transf = estimator.\
        named_steps['columntransformer'].\
        named_transformers_['pc_transf'].\
        named_steps['variancethreshold-2'].inverse_transform

    # Beta coefficients in PC and M space.
    beta = estimator[-1].coef_

    if beta.ndim == 1:
        beta = beta[None, :]

    # Return unscaled beta coefficients
    beta_ss = beta/ss_scale
    beta_pc_ss = beta_ss[:, :-1]
    w_m_ss = beta_ss[:, -1]

    # We are computing the weights for the centered or scaled data,
    # that's why we don't transform with the StandardScaler step
    w_pca_ss = vt_1_inv_transf(
        pca_inv_transf(
            vt_2_inv_transf(beta_pc_ss)
            )
        )
    w = np.column_stack((w_pca_ss, w_m_ss))

    if w.shape[0] == 1:
        w = w[0, :]
    return w

def check_scoring(scoring):
    from sklearn.metrics import r2_score, mean_squared_error

    if scoring == 'neg_mean_squared_error':
        def neg_mse(y_true, y_pred):
            return -mean_squared_error(y_true, y_pred)
        score = neg_mse
    elif scoring == 'r2_score':
        score = r2_score
    else:
        ValueError("scoring must be either 'neg_mean_squared_error'"
                   "or 'r2_score")
    return score
