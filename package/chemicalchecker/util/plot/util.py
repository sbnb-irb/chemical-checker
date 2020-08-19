"""Chemical Checker palette coloring functions."""
import colorsys
import matplotlib
import seaborn as sns
import matplotlib.colors as mc


def rgb2hex(r, g, b):
    """RGB to hexadecimal."""
    return '#%02x%02x%02x' % (r, g, b)


def coord_color(coordinate):
    """CC coordinate to CC color."""
    if coordinate[0] == 'A':
        return rgb2hex(250, 100, 80)
    if coordinate[0] == 'B':
        return rgb2hex(200, 100, 225)
    if coordinate[0] == 'C':
        return rgb2hex(80, 120, 220)
    if coordinate[0] == 'D':
        return rgb2hex(120, 180, 60)
    if coordinate[0] == 'E':
        return rgb2hex(250, 150, 50)
    return rgb2hex(250, 100, 80)


def lighten_color(color, amount=0.5):
    """Lighthen a color."""
    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def cc_colors(coord, lighness=0):
    """Predefined CC colors."""
    colors = {
        'A': ['#EA5A49', '#EE7B6D', '#F7BDB6'],
        'B': ['#B16BA8', '#C189B9', '#D0A6CB'],
        'C': ['#5A72B5', '#7B8EC4', '#9CAAD3'],
        'D': ['#7CAF2A', '#96BF55', '#B0CF7F'],
        'E': ['#F39426', '#F5A951', '#F8BF7D'],
        'Z': ['#000000', '#666666', '#999999']}
    return colors[coord[:1]][lighness]


def set_style(style=None):
    """Set basic plotting style andfonts."""
    try:
        matplotlib.font_manager._rebuild()
    except Exception as ex:
        print(str(ex))
    if style is None:
        style = ('ticks', {
            'font.family': 'sans-serif',
            'font.serif': ['Arial'],
            'font.size': 16,
            'axes.grid': True})
    else:
        style = style
    sns.set_style(*style)
