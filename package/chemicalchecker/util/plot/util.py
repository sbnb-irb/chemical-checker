"""Chemical Checker palette coloring functions."""
import numpy as np
import colorsys
import matplotlib
import seaborn as sns
import matplotlib.colors as mc
from matplotlib import colorbar
import matplotlib.pyplot as plt
import itertools
from scipy.stats import gaussian_kde
from sklearn.preprocessing import minmax_scale


def set_style(style=None):
    """Set basic plotting style andfonts."""
    if style is None:
        style = ('ticks', {
            'font.family': 'sans-serif',
            'font.serif': ['Arial'],
            'font.size': 16,
            'axes.grid': True})
    else:
        style = style
    sns.set_style(*style)


def rgb2hex(r, g, b):
    """RGB to hexadecimal."""
    return '#%02x%02x%02x' % (r, g, b)


def predefined_cc_colors(coord, lighness=0):
    """Predefined CC colors."""
    colors = {
        'A': '#EA5A49', # '#EE7B6D', '#F7BDB6'],
        'B': '#B16BA8', # '#C189B9', '#D0A6CB'],
        'C': '#5A72B5', # '#7B8EC4', '#9CAAD3'],
        'D': '#7CAF2A', # '#96BF55', '#B0CF7F'],
        'E': '#F39426', # '#F5A951', '#F8BF7D'],
        'Z': '#000000', # '#666666', '#999999']
    }
    if not coord in colors:
        coord = 'Z'
    return lighten_color(colors[coord[:1]], amount=1-lighness)

def lighten_color(color, amount=0):
    if amount == 0:
        return color
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def cc_colors(coordinate, lighness=0, alternate=False, dark_first=True):
    """CC coordinate to CC color."""
    if alternate:
        return _cc_colors_alternate(coordinate, lighness, dark_first)
    else:
        return predefined_cc_colors(coordinate[0], lighness)


def coord_color(coordinate):
    return cc_colors(coordinate)


def _cc_colors_alternate(coordinate, lighness, dark_first):
    check = True
    if dark_first:
        check = not check
    if coordinate[0] in 'ACE':
        check = not check
    if int(coordinate[1]) % 2 == int(check):
        return predefined_cc_colors(coordinate[0], 0)
    else:
        return predefined_cc_colors(coordinate[0], lighness)


def cc_coords():
    coords = list()
    for name, code in itertools.product("ABCDE", "12345"):
        coords.append(name + code)
    return coords


def lighten_color(color, amount=0.5):
    """Lighthen a color."""
    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def make_cmap(colors, position=None, bit=False):
    bit_rgb = np.linspace(0, 1, 256)
    if position is None:
        position = np.linspace(0, 1, len(colors))
    else:
        if len(position) != len(colors):
            raise Exception("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            raise Exception("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red': [], 'green': [], 'blue': []}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = matplotlib.colors.LinearSegmentedColormap(
        'my_colormap', cdict, 256)
    return cmap


def homogenous_ticks(ax, n_ticks=5, x_ticks=None, y_ticks=None):
    if x_ticks is None:
        x_ticks = n_ticks
    if y_ticks is None:
        y_ticks = n_ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xticks = np.linspace(xlim[0], xlim[1], x_ticks + 2)[1:-1]
    yticks = np.linspace(ylim[0], ylim[1], y_ticks + 2)[1:-1]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)


def cm2inch(value):
    return value / 2.54


def canvas(width=None, columns=2, height=10, grid=(1, 1),
           constrained_layout=False, dpi=300,
           margins=dict(left=0.05, right=0.95, top=0.95, bottom=0.05),
           width_ratios=None, height_ratios=None):
    if width is None and columns == 2:
        width = 17.4
    if width is None and columns == 1:
        width = 8.5
    fig = plt.figure(figsize=(cm2inch(width), cm2inch(height)), dpi=dpi)
    if margins is not None:
        plt.subplots_adjust(**margins)
    grid = fig.add_gridspec(*grid, wspace=0.25, hspace=0.35,
                            width_ratios=width_ratios,
                            height_ratios=height_ratios)
    return fig, grid


def cc_grid(fig, grid, legend_out=True):
    axes = list()
    if legend_out:
        subgrid = grid[:, :].subgridspec(
            2, 1, height_ratios=(1, 40), wspace=0, hspace=0.02)
        ax_legend = fig.add_subplot(subgrid[0])
        axes.append(ax_legend)
        subgrid = subgrid[1].subgridspec(5, 5, wspace=-0.15, hspace=0)
    else:
        subgrid = grid[:, :].subgridspec(5, 5, wspace=0, hspace=0)
    for idx, (gs, ds) in enumerate(zip(subgrid, cc_coords())):
        ax = fig.add_subplot(gs)
        ax.set_aspect('equal')
        homogenous_ticks(ax, 2)
        ax.tick_params(axis='both', length=2, left=False, bottom=False,
                       right=False)
        if idx % 5 == 0:
            ax.tick_params(axis='y', left=True)
            ax.set_ylabel(ds[0], labelpad=2, rotation='horizontal',
                          va='center', ha='center')
        if idx >= 20:
            ax.tick_params(axis='x', bottom=True)
            # ax.xaxis.set_label_position('top')
            ax.set_xlabel(ds[1], labelpad=-1)
        axes.append(ax)
    return axes


def make_cbar_ax(ax, cmap=plt.get_cmap('viridis'), title=''):
    cbar = colorbar.ColorbarBase(ax, orientation='horizontal',
                                 ticklocation='top', cmap=cmap)
    cbar.ax.set_xlabel(title, labelpad=0)
    cbar.ax.tick_params(axis='x', pad=0)
    cbar.set_ticks([1, .8, .6, .4, .2, .0])
    cbar.set_ticklabels(['High', '', '', '', '', 'Low'])
    # cbar.ax.invert_xaxis()
    cbar.ax.set_aspect(0.04)


def projection(front, back=None, front_kwargs=[], back_kwargs={}, ax=None,
               density_subsample=1000):

    def _proj_lims(P):
        xlim = [np.min(P[:, 0]), np.max(P[:, 0])]
        ylim = [np.min(P[:, 1]), np.max(P[:, 1])]
        xscale = (xlim[1] - xlim[0]) * 0.05
        yscale = (ylim[1] - ylim[0]) * 0.05
        xlim[0] -= xscale
        xlim[1] += xscale
        ylim[0] -= yscale
        ylim[1] += yscale
        abs_lim = np.max(np.abs(np.vstack([xlim, ylim])))
        return (-abs_lim, abs_lim),(-abs_lim, abs_lim)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    if not isinstance(front, list):
        front = [front]

    if back is not None:
        ax.scatter(back[:, 0], back[:, 1], **back_kwargs)

    for proj, kwargs in zip(front, front_kwargs):
        x = proj[:, 0]
        y = proj[:, 1]
        density = kwargs.pop('density', False)
        color = kwargs.pop('color', 'black')
        cmap = kwargs.pop('cmap', 'viridis')
        lw = kwargs.pop('lw', 0)
        s_min = kwargs.pop('s_min', 5)
        s_max = kwargs.pop('s_max', 500)
        if len(x) <= 2:
            density = False
            color = 'black'
        if density:
            xy = np.vstack([x, y])
            z = gaussian_kde(xy[:, :density_subsample])(xy)
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
            ax.scatter(x, y, c=z, s=minmax_scale(z, (s_min, s_max)),
                       cmap=cmap, lw=lw, **kwargs)
        else:
            ax.scatter(x, y, c=color, **kwargs)

    all_projs = np.vstack(front)
    if back is not None:
        all_projs = np.vstack([all_projs, back])
    xlim, ylim = _proj_lims(all_projs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax