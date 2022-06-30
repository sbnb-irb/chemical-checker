import h5py
import seaborn as sns
import hdbscan
from matplotlib import cm

def get_hierarchy_levels(space, features):
    """Gets a list of arrays containing the indexes for getting the different
    hierarchy levels of the features of a certain space."""
    
    if space in ['B4', 'B5', 'B2']:
        from chemicalchecker.util import psql
        chembl_dbname = 'chembl'

        chembl_class_query = psql.qstring(f'''SELECT DISTINCT pc.protein_class_id, pc.pref_name, pc.parent_id, pc.class_level
        FROM protein_classification pc
        ''', chembl_dbname)
        
        hierarchy_dict = dict()
        for entry in chembl_class_query:
            hierarchy_dict[f'Class:{entry[0]}'] = entry[-1]

        chembl_classes = dict()
        for key, value1, value2, value3 in chembl_class_query:
            chembl_classes[f'Class:{key}'] = value1, value2, value3
        hierarchy = list()
        
        for feature in features:
            if feature.startswith('Class:'):
                hierarchy.append(hierarchy_dict[feature])
            else:
                hierarchy.append(max(list(hierarchy_dict.values())) + 1)
        hierarchy = np.array(hierarchy)
                
        hierarchy_indexes = [(hierarchy==n) for n in np.unique(hierarchy)]

    if space == 'E1':
        letters = ['A', 'B', 'C', 'D', 'E']
        hierarchy_indexes = [np.char.startswith(features, letter) for letter in letters]
        
    if space == 'A4':
        hierarchy_indexes = [np.array([True for f in features])]
        
    if space not in ['B4', 'B5', 'B2', 'E1', 'A4']:
        hierarchy_indexes = [np.array([True for f in features])]
    return hierarchy_indexes



def cluster_rows(matrix, height=0.75):
    """Performs hierarchical clustering of the rows of a matrix.
    """
    import scipy.cluster.hierarchy as shc
    clusters = shc.cut_tree(shc.linkage(matrix, metric='jaccard'), height=height)

    clusters = [x[0] for x in clusters]

    new_matrix = list()
    for k in np.unique(clusters):
        rows = matrix[clusters==k]
        merged = np.zeros(matrix.shape[1], dtype=int)
        for row in rows:
            merged = merged | np.array(row)
        new_matrix.append(merged)
    new_matrix = np.array(new_matrix)
    return clusters, new_matrix

def unimodal_rows(matrix, coords, perc=65, mode='radius'):
    """Gets the rows of a matrix that are unimodal. Based on HDBSCAN."""
    new_matrix = list()
    unimodal_idxs = list()
    for row, idx in zip(matrix, range(len(matrix))):
        if row.nonzero()[0].shape[0] < 3:
            continue
        
        mask = row.nonzero()[0]
            
        clusterer = hdbscan.HDBSCAN(core_dist_n_jobs=-1, allow_single_cluster=False)
        
        labels = clusterer.fit_predict(coords[mask])
        n_unique = np.unique(labels).shape[0]
        cond1 = ((n_unique == 1) and (-1 not in np.unique(labels)))
        cond2 = ((n_unique == 2) and (-1 in np.unique(labels)))
        if n_unique == 1:
            new_matrix.append(row)
            unimodal_idxs.append(idx)
    try:
        new_matrix = np.vstack(new_matrix)
    except ValueError: # new_matrix is empty
        new_matrix = None
    unimodal_idxs = np.array(unimodal_idxs)
    return new_matrix, unimodal_idxs

def informative_rows(matrix, cutoff=30):
    """Gets the rows of a matrix that are informative, i.e. those with a number of positive observations
    higher than a given cutoff."""
    new_matrix = list()
    informative_idxs = list()
    for row, idx in zip(matrix, range(len(matrix))):
        if row.sum() >= cutoff:
            new_matrix.append(row)
            informative_idxs.append(idx)
    try:
        new_matrix = np.vstack(new_matrix)
    except ValueError:
        new_matrix = matrix
    informative_idxs = np.array(informative_idxs)
    return new_matrix, informative_idxs
            
def plot_landscapes(landscapes, scores, coords, idx=None, kde=True, ax=None, project=False, cmap=None, **kwargs):
    """Plots binary landscapes following the methods described in Costanzo et al.
    
    Args:
        * landscapes: binary matrix containing features in rows and neighborhoods in columns.
        * scores: matrix containing normalized scores ranging from 0 to 1. Its shape should be the
        same as for landscapes.
        * coords: coordinates of the different neighborhoods used for the projection."""
    from scipy.linalg import LinAlgError
    
    if idx is not None:
        landscapes = landscapes[idx]
        scores = scores[idx]
        if type(idx) == int:
            landscapes = np.array([landscapes])
            scores = np.array([scores])

    if cmap is None:
        colors = np.array([cm.rainbow(x) for x in np.linspace(0, 1, len(landscapes))])
    else:
        colors = np.array([cmap(x) for x in np.linspace(0, 1, len(landscapes))])
        
        
    final_colors = list()
    
    thresh = kwargs.pop('thresh', 0.2)
    s = kwargs.pop('s', 5)
    
    final_colors = list()
    for landscape, score in zip(landscapes.T, scores.T):
        final_color = np.zeros(4)
        
        count = 0
        for Ohat, O2, color in zip(landscape, score, colors):
            final_color += color*O2*Ohat
            count += Ohat

        #final_color = final_color/count
        #final_color[-1] = final_color[-1] * (1/count)
        final_color[final_color > 1] = 1
        final_colors.append(final_color)
    
    if ax is None:    
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'aspect': 'equal'})
    else:
        fig = ax.get_figure()
        
    back_cm = make_cmap([(0.2, 0.2, 0.2),(0.8, 0.8, 0.8)])
    
    if project:
        projection(coords, front_kwargs=[dict(density=True, cmap=back_cm, s_max=10)], ax=ax)
        ax.scatter(coords[:, 0], coords[:, 1], c=final_colors, s=s)
        return ax
    
    if kde:
        for row, color in zip(landscapes, colors):
            if len(coords[row]) == 0:
                continue
            try:
                sns.kdeplot(x=coords[row, 0], 
                            y=coords[row, 1],
                            color=color,
                            levels=1,
                            thresh=thresh,
                            fill=False,
                            ax=ax)
            except LinAlgError:
                print('a')
    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import cdist
    ch = coords[ConvexHull(coords).vertices]
    center = np.mean(ch, axis=0)
    radius = cdist(ch, np.vstack([center, center])).max()
    circle = plt.Circle(center, radius=radius, linestyle=(0, (5, 10)), color='white', fill=False)
    ax.add_patch(circle)
    
    ax.scatter(coords[:, 0], coords[:, 1], c=final_colors, s=s, edgecolors='none')
    ax.set_facecolor('black')
    ax.grid(False)
    return ax

from scipy.stats import fisher_exact
def compute_overlapping(matrix):
    idxs = list(range(len(matrix)))
    similarities = list()
    for idx, row in enumerate(matrix):
        rest = matrix[idxs!=idx].sum(1)[0]
        a = ((row!=0) & (rest!=0)).sum()
        b = ((row!=0) & (rest!=1)).sum()
        c = ((row!=1) & (rest!=0)).sum()
        d = ((row!=1) & (rest!=1)).sum()
        odds, p = fisher_exact([[a, b], [c, d]], alternative='less')
        similarities.append(p)

    similarities = np.array(similarities)
    return similarities

def merge_landscapes(matrix, scores, clusters):
    new_landscapes = list()
    new_scores = list()

    for k in np.unique(clusters):
        new_landscape = matrix[clusters==k].sum(0)
        new_landscape[new_landscape > 1] = 1
        new_landscapes.append(new_landscape)

        new_scores.append(scores[clusters==k].max(0))

    new_landscapes = np.vstack(new_landscapes)
    new_scores = np.vstack(new_scores)
    return new_landscapes, new_scores


from chemicalchecker.util.plot.util import projection, make_cmap, mc
import matplotlib.pyplot as plt
import os
import numpy as np

def main(datasignature, s, thr):
    self = datasignature
    
    V0 = self.get_h5_dataset('V0')
    V1 = self.get_h5_dataset('V')
    coords = self.get_h5_dataset('coords')
    features = self.get_h5_dataset('features')

    with h5py.File(os.path.join(self.model_path, 'safe.h5')) as f:
        scores = f['scores'][:]
        landscapes = (scores > thr).astype(int)
    n_enriched = landscapes.sum(axis=0)

    similarities = compute_overlapping(landscapes)
    fig, ax = plt.subplots()
    sns.kdeplot(similarities, ax=ax, cut=0)
    ax.set_xlabel('pvalue')
    ax.set_title('Distribution of overlapping pvalues')
    fig.savefig(os.path.join(self.diags_path, 'similarities_dist.png'), dpi=300)        

    import pandas as pd
    from collections import defaultdict
    df_dict = defaultdict(list)
    for f, sim in zip(features, similarities):
        try:
            df_dict['Description'].append(self.space_dict[f])
            df_dict['Feature'].append(f)
            df_dict['pvalue'].append(sim)
        except Exception:
            continue
    df = pd.DataFrame(df_dict)

    df = df.sort_values('pvalue', ascending=True)
    df['pvalue'] = df['pvalue'].map('{:.3e}'.format)
    df.to_csv(os.path.join(self.clus_path, 'overlapping.csv'), index=False)
    top_features = df['Feature'][:10]
    fig, ax = plt.subplots(figsize=(10, 10))
    fig = self.predict_safe(list(top_features))
    fig.savefig(os.path.join(self.diags_path, 'less_overlapping.png'), dpi=300)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(aspect='equal'))
    ax.scatter(coords[n_enriched!=0, 0], coords[n_enriched!=0, 1], s=s, edgecolors='none', c='C1', label='At least one enriched feature')
    ax.scatter(coords[n_enriched==0, 0], coords[n_enriched==0, 1], s=s, edgecolors='none', c='C0', label='No enriched features')

    ax.legend()
    ax.set_title(f'Signature {self.cctype[-1]} of {self.dataset}\ntSNE computed only with molecules with experimental data')
    plt.savefig(os.path.join(self.diags_path, 'enriched_vs_nonenriched.png'), dpi=300)

    fig, ax = plt.subplots()
    n_samples = landscapes.shape[1]
    n_vars = landscapes.shape[0]
    _ = ax.hist(n_enriched, 
                 bins=range(max(n_enriched)+2), 
                 weights=np.ones(n_samples) / n_samples,
                 edgecolor='None')


    ax.set_xlabel('Number of enriched features')
    ax.set_ylabel('Fraction of neighborhoods')
    ax.set_title('Number of enriched features per neighborhood')
    fig.savefig(os.path.join(self.diags_path, 'feats_per_neighborhood.png'), dpi=300)


    fig, ax = plt.subplots(2, 1, figsize=(15, 15), gridspec_kw={'height_ratios': (1, 20), 'hspace': 0.1})
    max_n = round(np.percentile(n_enriched, 75), ndigits=-1)
    if max_n < 6:
        max_n = 6
    n_enriched[n_enriched >= max_n] = max_n
    projection(coords, front_kwargs=[dict(c=n_enriched, s=s, edgecolors='none', cmap='viridis')], ax=ax[1])
    ax[1].set_aspect('equal')
    from matplotlib import colorbar
    cbar = colorbar.ColorbarBase(ax[0], orientation='horizontal',
                                     ticklocation='top', cmap=cm.viridis)
    cbar.ax.set_xlabel('Number of enriched features', labelpad=10, rotation=0)
    cbar.ax.tick_params(axis='x', pad=0)
    cbar.set_ticks([1, .8, .6, .4, .2, .0])
    cbar.set_ticklabels([f'>= {max_n}'] + [f'{x:.0f}' for x in np.linspace(n_enriched.max(), n_enriched.min(), 6)][1:])

    cbar.ax.set_aspect(0.05)

    fig.savefig(os.path.join(self.diags_path, 'n_enriched_tsne.png'), dpi=300)

    neigh_lengths = list()
    with h5py.File(os.path.join(self.clus_path, 'neigh.h5')) as f:
        for n in range(len(f['neighbors'])):
            neigh_lengths.append(len(f['neighbors'][str(n)]))
    neigh_lengths = np.array(neigh_lengths)

    fig, ax = plt.subplots()
    n_samples = landscapes.shape[1]
    n_vars = landscapes.shape[0]
    _ = ax.hist(neigh_lengths, 
                 bins=range(max(neigh_lengths)+2), 
                 weights=np.ones(n_samples) / n_samples,
                 edgecolor='None')


    ax.set_xlabel('Number of neighbors')
    ax.set_ylabel('Fraction of neighborhoods')
    ax.set_title('Number of neighbors per neighborhood')
    fig.savefig(os.path.join(self.diags_path, 'neighs_per_neighborhood.png'), dpi=300)

    fig, ax = plt.subplots(2, 1, figsize=(15, 15), gridspec_kw={'height_ratios': (1, 20), 'hspace': 0.1})
    max_n = round(np.percentile(neigh_lengths, 75), ndigits=-1)
    if max_n < 6:
        max_n = 6
    neigh_lengths[neigh_lengths >= max_n] = max_n
    projection(coords, front_kwargs=[dict(c=neigh_lengths, s=s, edgecolors='none', cmap='viridis')], ax=ax[1])
    ax[1].set_aspect('equal')
    from matplotlib import colorbar
    cbar = colorbar.ColorbarBase(ax[0], orientation='horizontal',
                                     ticklocation='top', cmap=cm.viridis)
    cbar.ax.set_xlabel('Number of neighbors', labelpad=10, rotation=0)
    cbar.ax.tick_params(axis='x', pad=0)
    cbar.set_ticks([1, .8, .6, .4, .2, .0])
    cbar.set_ticklabels([f'>= {max_n}'] + [f'{x:.0f}' for x in np.linspace(neigh_lengths.max(), neigh_lengths.min(), 6)][1:])

    cbar.ax.set_aspect(0.05)

    fig.savefig(os.path.join(self.diags_path, 'neigh_lengths_tsne.png'), dpi=300)

    n_neigh = landscapes.sum(1).reshape(1, -1)[0]
    n_vars = len(features)

    fig, ax = plt.subplots()
    ax.hist(n_neigh, bins=range(0, max(n_neigh)+2, 5), 
                 weights=np.ones(n_vars) / n_vars,
                 edgecolor='None')

    ax.set_ylabel('Fraction of landscapes')
    ax.set_xlabel('Landscape size (number of enriched neighborhoods)')
    ax.set_title('Landscape sizes')
    fig.savefig(os.path.join(self.diags_path, 'landscape_sizes.png'), dpi=300)

    informative_landscapes, informative_idxs = informative_rows(landscapes, cutoff=30)
    unimodal_landscapes, unimodal_idxs = unimodal_rows(landscapes, coords)

    informative = np.zeros(len(landscapes)).astype(bool)
    
    try:
        informative[informative_idxs] = True
    except IndexError:
        informative = False

    unimodal = np.zeros(len(landscapes)).astype(bool)
    try:
        unimodal[unimodal_idxs] = True
    except IndexError:
        unimodal = False

    #clusters, new_matrix = cluster_rows(landscapes)

    import pandas as pd
    df = pd.DataFrame({'feature': features, 'landscape': [x for x in landscapes], 'score': [x for x in scores], 'informative': informative, 'unimodal': unimodal})


    n_total = df.shape[0]
    n_informative = df.informative.sum()
    n_unimodal = df.unimodal.sum()
    n_both = (df.informative & df.unimodal).sum()

    labels = ['Total number of landscapes', 'Informative landscapes', 'Unimodal landscapes', 'Informative + unimodal']

    fig, ax = plt.subplots(figsize=(6, 6))
    bar = ax.bar(labels, [n_total, n_informative, n_unimodal, n_both])

    plt.ylim()
    for rect in bar:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, height + 10, f'{height:.0f}', ha='center', va='bottom')

    y_top = ax.get_ylim()[1]
    ax.set_ylim(0, y_top + 0.1*y_top)
    ax.set_xticklabels(labels, rotation=60, ha='right')
    ax.set_title('Classification of landscapes')
    ax.grid(False)
    plt.subplots_adjust(bottom=0.3)
    fig.savefig(os.path.join(self.diags_path, 'classification.png'), dpi=300)

    hierarchy_idxs = get_hierarchy_levels(self.dataset[:2], features)
    levels = [x for x in range(len(hierarchy_idxs))]

    for level in levels:
        hierarchy_mask = hierarchy_idxs[level]
        level_feat = features[hierarchy_mask]
        desc = list()
        for x in features[hierarchy_mask]:
            try:
                desc.append(self.space_dict[x])

            except Exception:
                desc.append('Unknown')

        from textwrap import wrap
        labels = ['\n'.join(wrap(f'{x}: {y}', 30)) for x, y in zip(features[hierarchy_mask], 
                                                                   desc)]

        fig, ax = plt.subplots(1, 2, gridspec_kw=dict(width_ratios=(50, 1)), figsize=(15, 10))
        plot_landscapes(landscapes[hierarchy_mask], scores[hierarchy_mask], coords, kde=False, ax=ax[0], s=s)
        ax[0].set_aspect('equal')
        fig.suptitle(f'Level {level}', fontsize=20)

        import matplotlib.patches as mpatches
        legends = list()
        colors = [cm.rainbow(x) for x in np.linspace(0, 1, len(labels))]
        for color, label in zip(colors, labels):
                legends.append((mpatches.Patch(color=color), label))

        ax[1].legend(*zip(*legends), title='Group', ncol=1, loc='center')
        ax[1].axis('off')
        plt.savefig(os.path.join(self.diags_path, f'level_{level}.png'), dpi=300)

    nrows = int(np.ceil((len(hierarchy_idxs)+1)/3))
    fig, axs = plt.subplots(nrows, 3, figsize=(20, 12), subplot_kw=dict(aspect='equal'))
    axs = axs.flatten()
    total_len = list()
    informative_len = list()
    unimodal_len = list()
    count = 0
    for ax, hierarchy_mask in zip(axs[:-1], hierarchy_idxs):

        full_mask = hierarchy_mask & df.informative & df.unimodal

        try:
            plot_landscapes(np.vstack(df[full_mask].landscape), np.vstack(df[full_mask].score), coords, ax=ax, s=s)
        except ValueError:
            plot_landscapes(np.vstack(df[hierarchy_mask].landscape), np.vstack(df[hierarchy_mask].score), coords, kde=False, ax=ax, s=s)

        ax.set_title(f'Level {count}')
        count += 1

        total_len.append(df[hierarchy_mask].shape[0])
        informative_len.append(df[hierarchy_mask & df.informative].shape[0])
        unimodal_len.append(df[full_mask].shape[0])

    bins = np.linspace(0, 1, len(hierarchy_idxs))

    bar1 = axs[-1].bar(bins-0.05, total_len, width=0.05, label='Total number of features')
    bar2 = axs[-1].bar(bins, informative_len, width=0.05, label='Informative features')
    bar3 = axs[-1].bar(bins+0.05, unimodal_len, width=0.05, label='Informative and unimodal features')
    axs[-1].set_ylim(0, axs[-1].get_ylim()[1] + 20)

    for rect in bar1 + bar2 + bar3:
        height = rect.get_height()
        axs[-1].text(rect.get_x() + rect.get_width() / 2.0, height + 10, f'{height:.0f}', ha='center', va='bottom', rotation=90)

    levels = [x for x in range(len(hierarchy_idxs))]
    axs[-1].set_xticks(bins)
    axs[-1].set_xticklabels(levels)
    axs[-1].set_aspect('auto')
    axs[-1].legend()
    plt.savefig(os.path.join(self.diags_path, 'hierarchical_levels_uni.png'), dpi=300)

    fig, axs = plt.subplots(nrows, 3, figsize=(20, 12), subplot_kw=dict(aspect='equal'))
    axs = axs.flatten()
    total_len = list()
    informative_len = list()
    unimodal_len = list()
    count = 0
    for ax, hierarchy_mask in zip(axs[:-1], hierarchy_idxs):
        full_mask = hierarchy_mask & df.informative & df.unimodal

        plot_landscapes(np.vstack(df[hierarchy_mask].landscape), np.vstack(df[hierarchy_mask].score), coords, kde=False, ax=ax, s=s)

        ax.set_title(f'Level {count}')
        count += 1

        total_len.append(df[hierarchy_mask].shape[0])
        informative_len.append(df[hierarchy_mask & df.informative].shape[0])
        unimodal_len.append(df[full_mask].shape[0])

    bins = np.linspace(0, 1, len(hierarchy_idxs))

    bar1 = axs[-1].bar(bins-0.05, total_len, width=0.05, label='Total number of features')
    bar2 = axs[-1].bar(bins, informative_len, width=0.05, label='Informative features')
    bar3 = axs[-1].bar(bins+0.05, unimodal_len, width=0.05, label='Informative and unimodal features')
    axs[-1].set_ylim(0, axs[-1].get_ylim()[1] + 20)

    for rect in bar1 + bar2 + bar3:
        height = rect.get_height()
        axs[-1].text(rect.get_x() + rect.get_width() / 2.0, height + 10, f'{height:.0f}', ha='center', va='bottom', rotation=90)

    axs[-1].set_xticks(bins)
    axs[-1].set_xticklabels(levels)
    axs[-1].set_aspect('auto')
    axs[-1].legend()
    plt.savefig(os.path.join(self.diags_path, 'hierarchical_levels_all.png'), dpi=300) 