import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from pingouin import compute_bootci

from nilearn.plotting.img_plotting import MNI152TEMPLATE
from nilearn.plotting import cm
    
def boxviolin_w_points(y, data, x=None, ax=None, points_kws = None, violin_kws = None, box_kws=None):
    """plot violinplots with boxes and cloud of points"""
    if ax is None:
        ax = plt.gca()
    if points_kws is None:
        points_kws = {'edgecolor': "black", 'linewidth':1}
    if violin_kws is None:
        violin_kws = {'inner': "quartile", 'linewidth':0}
    
    order=None
    if x:
        order = np.sort(data.loc[:,x].unique())
        
    sns.violinplot(x=x, y=y, data=data, ax=ax, scale = 'width', **violin_kws, order=order)
    plt.setp(ax.collections, alpha=.5)
    
    #sns.stripplot(x=x, y=y, data=data, ax=ax, hue=x, legend=False, **points_kws, order=order, hue_order=order)
    sns.stripplot(x=x, y=y, data=data, ax=ax, hue=x, **points_kws, order=order, hue_order=order)

    sns.boxplot(x=x, y=y, 
                data=data, ax=ax, width=0.3, 
                boxprops={'zorder': 2, 'edgecolor':'k'}, 
                medianprops={'color':'k'},
                capprops={'color':'k'},
                linewidth=points_kws['linewidth'], fliersize=0, order=order)
    
    # add mean with SE
    labels = [label.get_text() for label in ax.get_xticklabels()]

    means = [data[data.loc[:, x]==label][y].to_numpy().mean() for label in labels]
    cis = [compute_bootci(x=data[data.loc[:, x]==label][y].to_numpy(), func='mean', method='norm') 
           for label in labels]
    err_bars = [np.array([m-ci[0], ci[1]-m]) for ci, m in zip(cis, means)]
    err_bars = np.vstack(err_bars).T
    x_ticks = ax.get_xticks()
    ax.errorbar(x_ticks + 0.3, means, yerr=err_bars, 
                color='red', 
                 marker='x',
                linewidth=0, capsize=5, capthick=2, elinewidth=2)
    #sns.pointplot(x=ax.get_xticks(), y=y, data=data, ax=ax, errorbar = 'se', join=False)
   
    return ax

def plot_surf_subcortical(img, vmax=None, vmin=None, 
                          threshold=1e-6, cmap=cm.cold_white_hot, 
                          black_bg=False,
                          figure=None):
    from scipy.io import loadmat
    from nilearn.surface import load_surf_mesh, vol_to_surf
    from matplotlib.colors import LightSource
    import matplotlib.pylab as plt
    from matplotlib import gridspec

    if black_bg:
        bg = 'k'
    else:
        bg = 'w'
        
    # These surfaces have been borrowed from canlab toolbox
    brainstem=loadmat("../data/surf_spm2_brainstem.mat")
    thalamus=loadmat("../data/surf_spm2_thal.mat")
    
    if vmax is None:
        vmax = np.nanmax(load_img(img).get_fdata())
    if vmin is None:
        vmin = np.nanmin(load_img(img).get_fdata())

    mesh_brainstem = load_surf_mesh([brainstem['vertices'], brainstem['faces']-1])
    mesh_thalamus = load_surf_mesh([thalamus['vertices'], thalamus['faces']-1])

    brainstem_map_data = vol_to_surf(img, mesh_brainstem,  interpolation='nearest', n_samples=1)
    
    brainstem_map_faces = np.mean(brainstem_map_data[mesh_brainstem[1]], axis=1)
    brainstem_map_faces = np.nan_to_num(brainstem_map_faces)

    brainstem_idxs = np.where(np.abs(brainstem_map_faces) >= threshold)[0]

    brainstem_map_faces = (brainstem_map_faces - vmin)/ (vmax - vmin)
    
    # Use n_samples = 1 to just take one value. This ensures we are plotting real data, and not averaged one.
    thalamus_map_data = vol_to_surf(img, mesh_thalamus,  interpolation='nearest', n_samples=1)
    thalamus_map_faces = np.mean(thalamus_map_data[mesh_thalamus[1]], axis=1)
    thalamus_map_faces = np.nan_to_num(thalamus_map_faces)
    thalamus_idxs = np.where(np.abs(thalamus_map_faces) >= threshold)[0]
    
    thalamus_map_faces = (thalamus_map_faces - vmin)/ (vmax - vmin)
    

    xs = np.ptp(np.concatenate((mesh_brainstem[0][:,0], mesh_thalamus[0][:,0])))
    ys = np.ptp(np.concatenate((mesh_brainstem[0][:,1], mesh_thalamus[0][:,1])))
    zs = np.ptp(np.concatenate((mesh_brainstem[0][:,2], mesh_thalamus[0][:,2])))
    
    if figure is None:
        fig = plt.figure(figsize=(15, 15), constrained_layout=False)
        

    # plot mesh without data
    grid = gridspec.GridSpec(1, 2, wspace=-0.2)
    axes = []
    
    ax1 = fig.add_subplot(grid[0,0], projection="3d",  facecolor=bg)
    ax1.view_init(elev = 8, azim=135)

    ax1.set_box_aspect((xs, ys,zs), zoom=1.2)

    p3dcollec11 = ax1.plot_trisurf(mesh_brainstem[0][:,0], mesh_brainstem[0][:,1], mesh_brainstem[0][:,2],
                                   triangles=mesh_brainstem[1], linewidth=0.,
                                   antialiased=False, edgecolor='none', lightsource=LightSource(azdeg=-25), 
                                   color='white')

    face_colors11 = p3dcollec11._facecolors
    brainstem_colors = cmap(brainstem_map_faces[brainstem_idxs])
    
    face_colors11[brainstem_idxs] = brainstem_colors
    
    p3dcollec11.set_facecolors(face_colors11)


    p3dcollec12 = ax1.plot_trisurf(mesh_thalamus[0][:,0], mesh_thalamus[0][:,1], mesh_thalamus[0][:,2],
                                   triangles=mesh_thalamus[1], linewidth=0.,
                                   antialiased=False, edgecolor='none', lightsource=LightSource(azdeg=-25),
                                   color='white')

    face_colors12 = p3dcollec12._facecolors
    face_colors12[thalamus_idxs] = cmap(thalamus_map_faces[thalamus_idxs])
    
    p3dcollec12.set_facecolors(face_colors12)
    
    plt.axis('off')
    
    axes.append(ax1)

    ax2 = fig.add_subplot(grid[0,1], projection="3d",  facecolor=bg)
    ax2.view_init(elev=12, azim=25)
        
    ax2.set_box_aspect((xs, ys,zs), zoom=1.2)

    p3dcollec21 = ax2.plot_trisurf(mesh_brainstem[0][:,0], mesh_brainstem[0][:,1], mesh_brainstem[0][:,2],
                                   triangles=mesh_brainstem[1], linewidth=0.,
                                   antialiased=False, edgecolor='none', 
                                   lightsource=LightSource(azdeg=25), color='white')

    face_colors21 = p3dcollec21._facecolors
    face_colors21[brainstem_idxs] = cmap(brainstem_map_faces[brainstem_idxs])
    p3dcollec21.set_facecolors(face_colors21)

    p3dcollec22 = ax2.plot_trisurf(mesh_thalamus[0][:,0], mesh_thalamus[0][:,1], mesh_thalamus[0][:,2],
                                triangles=mesh_thalamus[1], linewidth=0.,
                                antialiased=False, edgecolor='none', lightsource=LightSource(azdeg=25),
                                color='white')
    face_colors22 = p3dcollec22._facecolors
    face_colors22[thalamus_idxs] = cmap(thalamus_map_faces[thalamus_idxs])
    p3dcollec22.set_facecolors(face_colors22)

    plt.axis('off')
    
    axes.append(ax2)
    return fig, axes
    


def plot_stat_subcortex(stat_map_img,
                        bg_img=MNI152TEMPLATE,
                        cut_coords=None,
                        output_file=None,
                        display_mode='ortho', colorbar=True,
                        figure=None, axes=None, title=None, threshold=1e-6,
                        annotate=True, draw_cross=True, black_bg='auto',
                        cmap=cm.cold_hot,
                        symmetric_cbar="auto",
                        dim='auto', vmax=None,
                        resampling_interpolation='continuous',
                        **kwargs):
    """Plot cuts of an ROI/mask image (by default 3 cuts: Frontal, Axial, and
    Lateral) of subcortex only. Most parts have been borrowed from nilearn.

    Parameters
    ----------
    stat_map_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        The statistical map image
    %(bg_img)s
        If nothing is specified, the MNI152 template will be used.
        To turn off background image, just pass "bg_img=None".
        Default=MNI152TEMPLATE.
    %(cut_coords)s
    %(output_file)s
    %(display_mode)s
    %(colorbar)s
        Default=True.
    %(figure)s
    %(axes)s
    %(title)s
    %(threshold)s
        Default=1e-6.
    %(annotate)s
    %(draw_cross)s
    %(black_bg)s
        Default='auto'.
    %(cmap)s

        .. note::
            The ccolormap *must* be symmetrical.

        Default=`plt.cm.cold_hot`.
    %(symmetric_cbar)s
        Default='auto'.
    %(dim)s
        Default='auto'.
    %(vmax)s
    %(resampling_interpolation)s
        Default='continuous'.

    Notes
    -----
    Arrays should be passed in numpy convention: (x, y, z) ordered.

    For visualization, non-finite values found in passed 'stat_map_img' or
    'bg_img' are set to zero.

    See Also
    --------
    nilearn.plotting.plot_anat : To simply plot anatomical images
    nilearn.plotting.plot_epi : To simply plot raw EPI images
    nilearn.plotting.plot_glass_brain : To plot maps in a glass brain

    """

    from nilearn.image import load_img
    from nilearn.datasets import load_mni152_template
    from nilearn.image import math_img, resample_to_img
    from nilearn.plotting import plot_stat_map

    mni_template = load_mni152_template(resolution=1)
    mni_subcortex = load_img(
        "/usr/local/fsl/data/standard/MNI152_T1_1mm_subbr_mask.nii.gz"
        )
    mni_subcortex = resample_to_img(mni_subcortex,
                                    mni_template,
                                    interpolation='nearest')
    bg_subcortex = math_img("img1*img2", 
                            img1=mni_template, 
                            img2=mni_subcortex)

    stat_img_subcortex = math_img("img1*img2",
                                  img1=resample_to_img(
                                      stat_map_img,
                                      mni_template,
                                      interpolation='nearest'),
                                  img2=mni_subcortex)
    display = plot_stat_map(
        stat_map_img=stat_img_subcortex,
        bg_img = bg_subcortex,
        cut_coords=cut_coords,
        output_file=output_file,
        display_mode=display_mode,
        colorbar=colorbar,
        figure=figure,
        axes=axes,
        title=title,
        threshold=threshold,
        annotate=annotate,
        draw_cross=draw_cross,
        black_bg=black_bg,
        cmap=cmap,
        symmetric_cbar=symmetric_cbar,
        dim=dim,
        vmax=vmax,
        resampling_interpolation=resampling_interpolation,
        **kwargs)
    return display
