import h5py
import seaborn as sns
import hdbscan
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(y_true, y_pred, fig_path, extra_metric, title=None, color='C0'):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = np.mean([accuracy_score(y_t, y_p) for y_t, y_p in zip(y_true, y_pred)])
    micro_precision = precision_score(y_true, y_pred, average='micro')
    sample_precision = precision_score(y_true, y_pred, average='samples')
    micro_recall = recall_score(y_true, y_pred, average='micro')
    sample_recall = recall_score(y_true, y_pred, average='samples')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    sample_f1 = f1_score(y_true, y_pred, average='samples')

    fig, axs = plt.subplots()
    axs.bar([extra_metric[0]] + 
            ['Accuracy',
            'Sample precision',
            'Micro-av. precision',
            'Sample recall',
            'Micro-av. recall',
            'Sample F1',
            'Micro-av. F1'], 

            [extra_metric[1]] + 
            [accuracy, 
             micro_precision, 
             sample_precision, 
             micro_recall, 
             sample_recall, 
             micro_f1, 
             sample_f1], 
            color=color, 
            alpha=0.6)

    axs.set_title(title)
    axs.set_xticklabels(labels=['Coverage',
                                   'Accuracy', 
                                   'Sample precision', 
                                   'Micro-av. precision', 
                                   'Sample recall', 
                                   'Micro-av. recall', 
                                   'Sample F1', 
                                   'Micro-av. F1'], 
                        rotation=60, 
                        ha='right')
    axs.set_aspect(7.0)

    fig.savefig(fig_path,
                bbox_inches='tight',
                dpi=300)

def lollipops(spaces):
    import matplotlib
    matplotlib.use('cairo')
    import matplotlib.gridspec as gridspec
    names = ['coverage', 'precision', 'recall', 'n_clusters']
    labels = ['Cluster coverage', 'Label precision', 'Label recall', 'Number of clusters']
    from chemicalchecker.util.plot.util import cc_colors
    from matplotlib.ticker import MaxNLocator
    all_data = list()
    for name in names:
        data_list = list()
        for space in spaces:
            sign = cc_local.get_signature('char4', 'full', f'{space}.001', metric='cosine')
            pkl = os.path.join(sign.stats_path, f'{name}.pkl')
            with open(pkl, 'rb') as fh:
                data = pickle.load(fh)
            data_list.append(data)
        all_data.append(data_list)
    
    fig = plt.Figure(figsize=(8, 8))
    
    outer = gridspec.GridSpec(2, 1, wspace=0.2, hspace=0.15, height_ratios=(3, 1))

    inner1 = gridspec.GridSpecFromSubplotSpec(3, 1,
                    subplot_spec=outer[0], wspace=0.1, hspace=0)
    
    inner2 = gridspec.GridSpecFromSubplotSpec(1, 1,
                    subplot_spec=outer[1], wspace=0.1, hspace=0.1)
    
    colors = cm.tab10.colors[:len(spaces)]
    
    for idx in range(3):
        data = all_data[idx]
        ax = plt.Subplot(fig, inner1[idx])

        if ax.is_last_row():
            pass
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
            
        if ax.is_first_row():
            ax.set_title('Metrics')
        

        
        fig.add_subplot(ax)
        ax.scatter(data, spaces, color=colors)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(spaces) - 0.5)
        
        for i, point in enumerate(data):
            space = spaces[i]
            color = colors[i]
            ax.axhline(space, xmin=0, xmax=(point/ax.get_xlim()[1]), color=color)
    
        ax.invert_yaxis()
        ax.set_aspect(1/ax.get_data_ratio())
        ax.grid(axis='y')
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_label_position('right')
        ax.set_ylabel(labels[idx], rotation=270, va='bottom', fontsize=12)
          
    data = all_data[3]
    ax = plt.Subplot(fig, inner2[0])
    fig.add_subplot(ax)
    ax.scatter(data, spaces, color=colors)
    ax.grid(axis='y')
    ax.set_xlim(xmin=0, xmax=max(data)+5)
    
    ax.set_ylim(-0.5, len(spaces) - 0.5)
    for idx, point in enumerate(data):
        space = spaces[idx]
        color = colors[idx]
        ax.axhline(space, xmin=0, xmax=(point/ax.get_xlim()[1]), color=color)
        
    ax.set_aspect(1/ax.get_data_ratio())
    ax.yaxis.set_label_position('right')

    ax.set_ylabel('Number of clusters', rotation=270, va='bottom', fontsize=12)

    ax.invert_yaxis()
    ax.xaxis.set_major_locator(MaxNLocator(5))

    return fig

