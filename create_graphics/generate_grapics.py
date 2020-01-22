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
    ax.set_xticks(np.arrange(data.shape[0]))
    ax.set_yticks(np.arrange(data.shape[1]))
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    ax.set_xticklabels(rowlabels)
    ax.set_yticklabels(collabels)

    for edge, spine in ax.spines.items():
        spine.set_visible=False

    ax.grid(which='minor', colow='w', linestile='-', linewidth=3)
    ax.set_title(title, pad=15)

    plt.tight_layout()
    plt.savefig(imgname)





data = pd.read_csv('../../processed_data/exp2_data.csv')


save_heatmap(data, data.shape[0], data.shape[1], 'Activation Ring Over Time', 'img1.png')