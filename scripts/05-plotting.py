# IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import os
import argparse
import sys
import re
from glob import glob
from pathlib import Path
from os.path import join as opj
from datetime import datetime
import matplotlib.colors as mcolors
from joblib import load
from statsmodels.stats.multitest import multipletests

from scipy.stats import pearsonr,ttest_1samp
from sklearn.metrics import r2_score
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector

from nilearn import plotting
from nilearn.masking import unmask, apply_mask
from nilearn.datasets import load_mni152_template
from nilearn.image import resample_to_img

project_dir = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(project_dir)
from src.input_data import load_data
from src.visualization import boxviolin_w_points
from src.utils import get_results_R2, get_results
from src.models import (L1Model_XY, L2Model_XY, L1L2Model_XY,
                        L1Model_XMY, L2Model_XMY, L1L2Model_XMY)


def main():

    parser = argparse.ArgumentParser(description='Generate plots for a particular scenario')
    parser.add_argument('--target',
                        dest="target",
                        type=str,
                        #required=True,
                        default = "mavg_bulbf_ccaf",
                        help='Which variable as a target (e.g. mavg_bulbf_ccaf)')
    parser.add_argument('--mediator',
                        dest="mediator",
                        type=str,
                        #required=True,
                        default = "map_auc_g_both",
                        help='Which variable as a mediator '
                        '(e.g. map_auc_g_both)')
    parser.add_argument('--task',
                        dest="task",
                        type=str,
                        #required=True,
                        default="both",
                        choices=['both', 'stroop', 'msit'],
                        help='Which task used')
    parser.add_argument('--model',
                        type=str,
                        #required=True,
                        default="ridge",
                        choices=['ridge', 'lasso', 'elasticnet'],
                        help='Which penalized PC Regression model')
    parser.add_argument('--phenotypes',
                        action='store_true',
                        help="Plot phenotypes")
    parser.add_argument('--output_dir',
                        type=str,
                        help="Name for the output directory")
    opts = parser.parse_args()

    y_var = opts.target  # e.g "mavgimt"
    m_var = opts.mediator  # e.g "map_auc_g_both"
    run_task = opts.task  # e.g "both"

    if opts.output_dir:
        output_dir = opj(project_dir, opts.output_dir)
    else:
        res_name = f"Y-{y_var}_M-{m_var}_task-{run_task}"
        output_dir = opj(project_dir, "results", res_name, "plots", opts.model)
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    predictions_dir = opj(project_dir, "results",
                          f"Y-{y_var}_M-{m_var}_task-{run_task}",
                          "predictions",
                          f"{opts.model}")
    if os.path.exists(predictions_dir) is False:
        sys.exit("Predictions for this scenario do not exist")

    ############## PLOT PREDICTIONS ##############

    this_output_dir = opj(output_dir, "performances")
    Path(this_output_dir).mkdir(exist_ok=True, parents=True)

    list_res_df_r2 = []
    list_preds_df = []
    cases = glob(opj(predictions_dir, "model*.npz"))
    for case in cases:
        model = re.findall(opj(predictions_dir, 'model_(.*).npz'), case)[0]
        res_file = opj(predictions_dir, f"model_{model}.npz")

        res_df_r2 = get_results_R2(res_file)
        res_df_r2['model'] = model
        list_res_df_r2.append(res_df_r2)

        res_pred_df = get_results(res_file)
        res_pred_df['model'] = model
        list_preds_df.append(res_pred_df)

    list_res_df_r2 = pd.concat(list_res_df_r2)
    list_res_df_r2.model = list_res_df_r2.model.replace(to_replace=['xy', 'xm', 'my', 'xmy'],
                                                        value=['1', '2', '3','4'])

    list_preds_df = pd.concat(list_preds_df)

    # Violin plots
    fig, ax = plt.subplots(figsize=(12,7))
    boxviolin_w_points(x='model', y='R2', data=list_res_df_r2)
    ax.hlines(y=0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linestyles='dashed', colors='k')
    ax.tick_params(axis='x', rotation=0, labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.set_yticks([0.0, 0.05, 0.1, 0.15,0.2])
    ax.set_xlabel("")
    ax.set_xticklabels(['X->Y', 'X->M', 'Y->M','X + M->Y'])
    ax.set_ylabel("Coefficient of Determination", size=30)
    ax.set_ylim([-0.1, 0.25])
    ax.set_title(f"Y: {y_var}, M: {m_var}, \n X: Incongruent vs Congruent ({run_task})" , size=35)
    fig.savefig(opj(this_output_dir, "fig_r2_violins.png"), dpi=600,  bbox_inches='tight')
    fig.savefig(opj(this_output_dir, "fig_r2_violins.svg"), dpi=600,  bbox_inches='tight')
    fig.savefig(opj(this_output_dir, "fig_r2_violins.pdf"), dpi=600,  bbox_inches='tight')
    plt.close()

    for model_id in list_preds_df.model.unique():
        # Scatter plots
        with sns.axes_style("whitegrid", {"axes.edgecolor": 'k'}):

            fig, axs = plt.subplots(nrows=5, figsize=(7,15), sharex=True)
            for ax, repeat_id in zip(axs, range(1, 6)):
                cond = (list_preds_df.model == model_id) & (list_preds_df.seed == repeat_id)
                res_df = list_preds_df.loc[cond,:]
                r, p = pearsonr(res_df.true, res_df.pred)
                r2 =  r2_score(res_df.true, res_df.pred)

                bayes_factor = importr("BayesFactor")
                bf = bayes_factor.extractBF(bayes_factor.correlationBF(FloatVector(res_df.true.to_numpy()),
                                                                       FloatVector(res_df.pred.to_numpy())),
                                            onlybf=True)
                if bf[0]>150:
                    bf="BF10>150"
                else:
                    bf = np.round(bf[0], 2)
                    bf=f"BF10={bf}"

                legend = f"R²={r2:.3f}, r={r:.3f}, p={p:.2e}, {bf}"

                props = dict(boxstyle='round', edgecolor='k', facecolor='white', alpha=1)
                ax.text(0.15, 0.15, legend, transform=ax.transAxes, fontsize=14, 
                        verticalalignment='top', bbox=props)

                sns.regplot(x='true', y='pred', data=res_df, label=legend,
                            scatter_kws={'s':10}, line_kws={'linewidth':1, 'color':'red'}, 
                            ax=ax)

                ax.set_xlabel("")
                ax.set_ylabel("Predicted", fontsize=15)

            ax.set_xlabel("Observed", fontsize=15)
            fig.savefig(opj(this_output_dir, f"fig_preds_scatter_model_{model_id}.png"), dpi=600,  bbox_inches='tight')
            fig.savefig(opj(this_output_dir, f"fig_preds_scatter_model_{model_id}.svg"), dpi=600,  bbox_inches='tight')
            fig.savefig(opj(this_output_dir, f"fig_preds_scatter_model_{model_id}.pdf"), dpi=600,  bbox_inches='tight')
            plt.close()

    ############## PLOT Phenotypes ##############
    if opts.phenotypes:
        phenotypes_dir = opj(project_dir,
                             "results",
                             f"Y-{y_var}_M-{m_var}_task-{run_task}",
                             "phenotypes",
                             f"{opts.model}"
                            )
        if os.path.exists(phenotypes_dir) is False:
            sys.exist("phenotypes have not been computed")

        this_output_dir = opj(output_dir, "phenotypes")
        Path(this_output_dir).mkdir(exist_ok=True, parents=True)

        mask_img = opj(project_dir, "data/resliced_grey25grey25.nii")
        bg_img = resample_to_img(load_mni152_template(resolution=2), mask_img)

        X = np.load(opj(phenotypes_dir, "data.npz"))['X']
        X_demean = X - X.mean(0)

        m = np.load(opj(phenotypes_dir, "data.npz"))['m']
        m_demean = m - m.mean()

        XM_demean = np.column_stack((X_demean, m_demean))

        xy_model = load(opj(phenotypes_dir, "model_xy.joblib"))
        w_xy = xy_model.get_weights()
        enc_w_xy = X_demean.T @ (X_demean @ w_xy)

        xm_model = load(opj(phenotypes_dir, "model_xm.joblib"))
        w_xm = xm_model.get_weights()
        enc_w_xm = X_demean.T @ (X_demean @ w_xm)

        xmy_model = load(opj(phenotypes_dir, "model_xmy.joblib"))
        w_xmy = xmy_model.get_weights()
        enc_w_xmy = XM_demean.T @ (XM_demean @ w_xmy)

        plotting.plot_img(unmask(w_xm, mask_img),
                          cmap=plotting.cm.cyan_orange_r,
                          black_bg=False,
                          output_file = opj(this_output_dir, "apath_decoding.png"),
                          colorbar=False, #title="c' path ",
                          draw_cross=False,
                          bg_img=bg_img,
                          threshold=1e-15,
                          norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=w_xm.min(), vmax=w_xm.max()),
                          display_mode='tiled', cut_coords=(0,0,0))

        plotting.plot_img(unmask(enc_w_xm, mask_img),
                          output_file = opj(this_output_dir, "apath_encoding.png"),
                          cmap=plotting.cm.cyan_orange_r,
                          black_bg=False,
                          colorbar=False, #title="a' path ",
                          draw_cross=False,
                          bg_img=bg_img,
                          threshold=1e-15,
                          norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=enc_w_xm.min(), vmax=enc_w_xm.max()),
                          display_mode='tiled', cut_coords=(0,0,0))

        plotting.plot_img(unmask(w_xy, mask_img),
                          cmap=plotting.cm.cyan_orange_r,
                          black_bg=False,
                          output_file = opj(this_output_dir, "cpath_decoding.png"),
                          colorbar=False, #title="c' path ",
                          draw_cross=False,
                          bg_img=bg_img,
                          threshold=1e-15,
                          norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=w_xy.min(), vmax=w_xy.max()),
                          display_mode='tiled', cut_coords=(0,0,0))

        plotting.plot_img(unmask(enc_w_xy, mask_img),
                          output_file = opj(this_output_dir, "cpath_encoding.png"),
                          cmap=plotting.cm.cyan_orange_r,
                          black_bg=False,
                          colorbar=False, #title="a' path ",
                          draw_cross=False,
                          bg_img=bg_img,
                          threshold=1e-15,
                          norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=enc_w_xy.min(), vmax=enc_w_xy.max()),
                          display_mode='tiled', cut_coords=(0,0,0))

        plotting.plot_img(unmask(w_xmy[:-1], mask_img),
                          cmap=plotting.cm.cyan_orange_r,
                          black_bg=False,
                          output_file = opj(this_output_dir, "cppath_decoding.png"),
                          colorbar=False, #title="c' path ",
                          draw_cross=False,
                          bg_img=bg_img,
                          threshold=1e-15,
                          norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=w_xmy[:-1].min(), vmax=w_xmy[:-1].max()),
                          display_mode='tiled', cut_coords=(0,0,0))

        plotting.plot_img(unmask(enc_w_xmy[:-1], mask_img),
                          output_file = opj(this_output_dir, "cppath_encoding.png"),
                          cmap=plotting.cm.cyan_orange_r,
                          black_bg=False,
                          colorbar=False, #title="a' path ",
                          draw_cross=False,
                          bg_img=bg_img,
                          threshold=1e-15,
                          norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=enc_w_xmy[:-1].min(), vmax=enc_w_xmy[:-1].max()),
                          display_mode='tiled', cut_coords=(0,0,0))

        # Filtering
        t_X, p_X = np.apply_along_axis(ttest_1samp, 0, X, popmean=0)

        bootstrap_dir = opj(project_dir,
                             "results",
                             f"Y-{y_var}_M-{m_var}_task-{run_task}",
                             "bootstrapping",
                             f"{opts.model}"
                            )

        a_boots = np.array([np.load(file)['a'] for file in glob(opj(bootstrap_dir, "*"))])
        b_boots = np.array([np.load(file)['b'] for file in glob(opj(bootstrap_dir, "*"))])
        ab_boots = a_boots*(b_boots[:,None])
        cp_boots = np.array([np.load(file)['cprime'] for file in glob(opj(bootstrap_dir, "*"))])
        c_boots = np.array([np.load(file)['c'] for file in glob(opj(bootstrap_dir, "*"))])

        # transform bootrsrapp weights in voxel space to empirical PC space
        xm_model = load(opj(phenotypes_dir, "model_xm.joblib"))
        xy_model = load(opj(phenotypes_dir, "model_xy.joblib"))
        xmy_model = load(opj(phenotypes_dir, "model_xmy.joblib"))

        # This is the relation between (scaled) betas and weights
        # beta = sigma x VT x w
        VT = xm_model.best_estimator_[:-1][2].components_
        support = xm_model.best_estimator_[:-1][3].get_support()
        sigma = np.eye(sum(support))*xm_model.best_estimator_[:-1][4].scale_

        a_PC_boots = sigma @ (VT @ a_boots.T)[support]
        c_PC_boots = sigma @ (VT @ c_boots.T)[support]

        cp_PC_boots = sigma @ (VT @ cp_boots.T)[support]
        b_PC_boots =  1/xmy_model.best_estimator_[-2].scale_[-1]*b_boots
        ab_PC_boots = (a_PC_boots*b_PC_boots[None,:])

        # Compute p-values
        n_boots = ab_boots.shape[0]

        pv_PC_ab = 2*np.min(1-np.row_stack((np.sum(ab_PC_boots>0, axis=1)/n_boots,
                                            np.sum(ab_PC_boots<0, axis=1)/n_boots)),
                            axis=0)
        pv_PC_ab[pv_PC_ab==0] = 1/(n_boots)

        pv_PC_cp = 2*np.min(1-np.row_stack((np.sum(cp_PC_boots>0, axis=1)/n_boots,
                                            np.sum(cp_PC_boots<0, axis=1)/n_boots)),
                            axis=0)
        pv_PC_cp[pv_PC_cp==0] = 1/(n_boots)

        pv_PC_c = 2*np.min(1-np.row_stack((np.sum(c_PC_boots>0, axis=1)/n_boots,
                                           np.sum(c_PC_boots<0, axis=1)/n_boots)),
                           axis=0)
        pv_PC_c[pv_PC_c==0] = 1/(n_boots)

        # Compute masks
        mask_PC_ab = multipletests(pv_PC_ab, method="fdr_bh")[0]
        mask_PC_cp = multipletests(pv_PC_cp, method="fdr_bh")[0]
        mask_PC_c = multipletests(pv_PC_c, method="fdr_bh")[0]

        # loadings in each case
        V_ab_fdr = VT[support,:][mask_PC_ab,:]
        V_c_fdr = VT[support,:][mask_PC_c,:]
        V_cp_fdr = VT[support,:][mask_PC_cp,:]

        this_output_dir = opj(output_dir, "loadings")
        Path(this_output_dir).mkdir(exist_ok=True, parents=True)

        # indirect effects
        for ii, pc_id in enumerate(np.argsort(abs(xm_model.coef_[mask_PC_ab]))[::-1]):

            ix = np.where(xm_model.coef_ ==  xm_model.coef_[mask_PC_ab][pc_id])[0]
            title = f"PC {ix+ 1}, coef={np.round(xm_model.coef_[ix], 3)[0]}"

            sign = np.sign(np.round(xm_model.coef_[ix], 3)[0])

            cmap_pos,cmap_neg  = plotting.cm.black_red,plotting.cm.black_blue_r
            label_pos, label_neg = "pos", "neg"
            if sign < 0:
                cmap_pos, cmap_neg = cmap_neg, cmap_pos
                label_pos, label_neg = label_neg, label_pos

            plotting.plot_img(unmask(sign*V_ab_fdr[pc_id,:]*(V_ab_fdr[pc_id,:]>0)*multipletests(p_X)[0],
                                     mask_img),
                              black_bg=False,
                              threshold=1e-12,
                              display_mode='z',
                              cmap=cmap_pos,
                              cut_coords = [-30, -20,-10, 0, 10, 20, 30],
                              draw_cross=False, bg_img=bg_img,
                              colorbar=False,
                              output_file = opj(this_output_dir, f"loading_ab_order_{ii}_pc_{ix[0]}_{label_pos}.png"),
                             )

            plotting.plot_img(unmask(sign*V_ab_fdr[pc_id,:]*(V_ab_fdr[pc_id,:]<0)*multipletests(p_X)[0],
                                     mask_img),
                              black_bg=False,
                              threshold=1e-12,
                              display_mode='z',
                              cmap=cmap_neg,
                              cut_coords = [-30, -20,-10, 0, 10, 20, 30],
                              draw_cross=False, bg_img=bg_img,
                              colorbar=False,
                              output_file = opj(this_output_dir, f"loading_ab_order_{ii}_pc_{ix[0]}_{label_neg}.png"),
                             )

        # Direct effects
        for ii, pc_id in enumerate(np.argsort(abs(xy_model.coef_[mask_PC_c]))[::-1]):
            ix = np.where(xy_model.coef_ ==  xy_model.coef_[mask_PC_c][pc_id])[0]
            title = f"PC {ix+ 1}, coef={np.round(xy_model.coef_[ix], 3)[0]}"

            sign = np.sign(np.round(xy_model.coef_[ix], 3)[0])

            cmap_pos,cmap_neg  = plotting.cm.black_red,plotting.cm.black_blue_r
            label_pos, label_neg = "pos", "neg"
            if sign < 0:
                cmap_pos, cmap_neg = cmap_neg, cmap_pos
                label_pos, label_neg = label_neg, label_pos

            plotting.plot_img(unmask(sign*V_c_fdr[pc_id,:]*(V_c_fdr[pc_id,:]>0)*multipletests(p_X)[0],
                                     mask_img),
                              black_bg=False,
                              threshold=1e-12,
                              display_mode='z',
                              cmap=cmap_pos,
                              cut_coords = [-30, -20,-10, 0, 10, 20, 30],
                              draw_cross=False, bg_img=bg_img,
                              colorbar=False,
                              output_file = opj(this_output_dir, f"loading_c_order_{ii}_pc_{ix[0]}_{label_pos}.png"),
                             )

            plotting.plot_img(unmask(sign*V_c_fdr[pc_id,:]*(V_c_fdr[pc_id,:]<0)*multipletests(p_X)[0],
                                     mask_img),
                              black_bg=False,
                              threshold=1e-12,
                              display_mode='z',
                              cmap=cmap_neg,
                              cut_coords = [-30, -20,-10, 0, 10, 20, 30],
                              draw_cross=False, bg_img=bg_img,
                              colorbar=False,
                              output_file = opj(this_output_dir, f"loading_c_order_{ii}_pc_{ix[0]}_{label_neg}.png"),
                             )

        # Direct effects (partial)
        for ii, pc_id in enumerate(np.argsort(abs(xmy_model.coef_[:-1][mask_PC_cp]))[::-1]):
            ix = np.where(xmy_model.coef_[:-1] ==  xmy_model.coef_[:-1][mask_PC_cp][pc_id])[0]
            title = f"PC {ix+ 1}, coef={np.round(xmy_model.coef_[:-1][ix], 3)[0]}"

            sign = np.sign(np.round(xmy_model.coef_[:-1], 3)[0])

            cmap_pos,cmap_neg  = plotting.cm.black_red,plotting.cm.black_blue_r
            label_pos, label_neg = "pos", "neg"
            if sign < 0:
                cmap_pos, cmap_neg = cmap_neg, cmap_pos
                label_pos, label_neg = label_neg, label_pos

            plotting.plot_img(unmask(sign*V_cp_fdr[pc_id,:]*(V_cp_fdr[pc_id,:]>0)*multipletests(p_X)[0],
                                     mask_img),
                              black_bg=False,
                              threshold=1e-12,
                              display_mode='z',
                              cmap=cmap_pos,
                              cut_coords = [-30, -20,-10, 0, 10, 20, 30],
                              draw_cross=False, bg_img=bg_img,
                              colorbar=False,
                              output_file = opj(this_output_dir, f"loading_cp_order_{ii}_pc_{ix[0]}_{label_pos}.png"),
                             )

            plotting.plot_img(unmask(sign*V_cp_fdr[pc_id,:]*(V_cp_fdr[pc_id,:]<0)*multipletests(p_X)[0],
                                     mask_img),
                              black_bg=False,
                              threshold=1e-12,
                              display_mode='z',
                              cmap=cmap_neg,
                              cut_coords = [-30, -20,-10, 0, 10, 20, 30],
                              draw_cross=False, bg_img=bg_img,
                              colorbar=False,
                              output_file = opj(this_output_dir, f"loading_cp_order_{ii}_pc_{ix[0]}_{label_neg}.png"),
                             )


if __name__ == "__main__":
    sys.exit(main())
