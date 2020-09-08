import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def dynamic_heatmap(df, columns, fontsize=20, annot=False, palette=None, figsize=(15, 10), squaresize=500):
    """Plots a heatmap that changes size values depending on correlation Adapted from:
    https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec"""

    plt.figure(figsize=figsize)
    corr = df[columns].corr()
    sns.set(style="dark")
    grid_bg_color = sns.axes_style()['axes.facecolor']

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    corr = pd.melt(corr.reset_index(),
                   id_vars='index')  # Unpivot the dataframe, so we can get pair of arrays for x and y
    corr.columns = ['x', 'y', 'value']

    x = corr['x']
    y = corr['y']
    size = corr['value'].abs().fillna(0)

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right');

    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}

    size_scale = squaresize

    if palette:
        n_colors = len(palette)
    else:
        n_colors = 256  # Use 256 colors for the diverging color palette
        palette = sns.diverging_palette(20, 220, n=n_colors) # Create the palette
    color_min, color_max = [-1,
                            1]  # Range of values that will be mapped to the palette, i.e. min and max possible correlation
    color = corr["value"].fillna(value=0)

    def value_to_color(val):
        val_position = float((val - color_min)) / (
                    color_max - color_min)  # position of value in the input range, relative to the length of the input range
        ind = int(val_position * (n_colors - 1))  # target index in the color palette
        return palette[ind]

    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1)  # Setup a 1x15 grid
    ax = plt.subplot(plot_grid[:, :-1])  # Use the leftmost 14 columns of the grid for the main plot

    ax.scatter(
        x=x.map(x_to_num),  # Use mapping for x
        y=y.map(y_to_num),  # Use mapping for y
        s=size * size_scale,  # Vector of square sizes, proportional to size parameter
        c=color.apply(value_to_color),  # Vector of square colors, mapped to color palette
        marker='s'  # Use square as scatterplot marker
    )

    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    #     ax.set_fontsize(font_scale)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)

    numbers = corr['value'].round(decimals=2)

    if annot:
        for i, txt in enumerate(numbers):
            annot_font_size = int(fontsize * size[i] * annot)
            ax.annotate(txt, (x.map(x_to_num)[i], y.map(x_to_num)[i]),
                        horizontalalignment="center", verticalalignment="center",
                        color=grid_bg_color, fontweight="black", fontsize=annot_font_size)

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])

    # Add color legend on the right side of the plot
    ax = plt.subplot(plot_grid[:, -1])  # Use the rightmost column of the plot

    col_x = [0] * len(palette)  # Fixed x coordinate for the bars
    bar_y = np.linspace(color_min, color_max, n_colors)  # y coordinates for each of the n_colors bars

    bar_height = bar_y[1] - bar_y[0]
    ax.barh(
        y=bar_y,
        width=[5] * len(palette),  # Make bars 5 units wide
        left=col_x,  # Make bars start at 0
        height=bar_height,
        color=palette,
        linewidth=0
    )
    ax.set_xlim(1, 2)  # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
    ax.grid(False)  # Hide grid
    ax.set_xticks([])  # Remove horizontal ticks
    ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))  # Show vertical ticks for min, middle and max
    ax.yaxis.tick_right()  # Show vertical ticks on the right
    plt.show()

def hex_to_hsl(hex_color, dec=False):
    rgb = hex_to_rgb(hex_color, True)
    raw_hls = colorsys.rgb_to_hls(*rgb)
    if dec:
        return raw_hsl
    else:
        h, l, s = int(raw_hls[0]*360), int(raw_hls[1]*100), int(raw_hls[2]*100)
        return [h, s, l]

def hex_to_rgb(hex_color, dec=False):
    if hex_color[0] and len(hex_color) == 7:
        search = re.search("#(\w{2})(\w{2})(\w{2})", hex_color)
        if dec:
            return [int(search.group(i), 16)/256 for i in np.arange(1,4)]
        else:
            return [int(search.group(i), 16) for i in np.arange(1,4)]
    else:
        print(f"{hex_color} is not formatted correctly")