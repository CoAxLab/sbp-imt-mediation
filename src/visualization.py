import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from pingouin import compute_bootci
    
def boxviolin_w_points(y, data, x=None, ax=None, points_kws = None, violin_kws = None, box_kws=None):
    
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
    
    sns.stripplot(x=x, y=y, data=data, ax=ax, hue=x, legend=False, **points_kws, order=order, hue_order=order)

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
