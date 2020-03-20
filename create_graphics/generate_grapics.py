import sys
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import numpy as np
import pandas as pd

d_scale  = 1.135
time_scale = 2.58
n_x_ticks = 4
n_y_ticks = 4

def save_heatmap(data, rowlabels, collabels, title, imgname):
    #plt.gray()
    keys = {'cmap':'Greys', 'origin':'lower', 'aspect':'auto'}
    #plt.matshow(data, cmap='Greys', origin='lower')
    plt.matshow(data, **keys)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data, **keys)

    fig.colorbar(cax, boundaries=range(0, 250))
    #ax.set_xticks([int(float(x)) for x in list(data.columns)])
    #ax.set_xticks(list(map(str, range(0, int(float(data.columns[-1])), 50))))
    #ax.set_yticks(np.arange(data.shape[1]))
    #ax.set_xticks(np.arange(n_x_ticks))
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    #ax.set_ybound(0, 220)
    ax.set_xticklabels([0] + list(map(str, range(0, int(float(data.columns[-1])), int(float(data.columns[-1])/n_x_ticks)))))
    ax.set_yticklabels([0] + list(range(0, int(data.shape[1]*d_scale), int(float(data.shape[1]*d_scale/n_y_ticks)))))
    #ax.set_xticklabels(list(map(int, map(float, data.columns))))
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
    #data[0] = data[0] * time_scale
    save_heatmap(data, x_label, y_label, title, dst)