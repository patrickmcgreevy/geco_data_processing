import sys
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import numpy as np
import pandas as pd


def save_heatmap(data, rowlabels, collabels, title, imgname):
    plt.matshow(data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data)

    fig.colorbar(cax)
    #ax.set_xticks(np.arange(data.shape[0]))
    #ax.set_yticks(np.arange(data.shape[1]))
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    plt.xlabel(rowlabels)
    plt.ylabel(collabels)

    for edge, spine in ax.spines.items():
        spine.set_visible = False

    ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
    ax.set_title(title, pad=15)

    plt.tight_layout()
    plt.savefig(imgname)


if (len(sys.argv) < 6):
    print('Error: Too few arguments')
    print('Please include \'src\' \'dest\' \'X-Axis Label\' \'Y-Axis Label\' \'Graph Title\'', 'in that order.')
    print('Labels and titles with spaces in them must be surounded by \'NAME\' single quotes.')
else:
    src = sys.argv[1]
    dst = sys.argv[2]
    x_label = sys.argv[3]
    y_label = sys.argv[4]
    title = sys.argv[5]

    data = pd.read_csv(src)

    save_heatmap(data, x_label, y_label, title, dst)