import matplotlib.pyplot as plt 
import numpy as np 
from pathlib import Path

def plot1D(original_hist, reconstructed_hist, run, hist_path, threshold):    
    fig, ax = plt.subplots()
    sse = np.mean(np.square(original_hist - reconstructed_hist))
    
    # for bin edges
    binEdges = np.linspace(0, 1, original_hist.shape[-1])
    width = binEdges[1] - binEdges[0]
    # plot original/recon 
    ax.bar(binEdges, original_hist[0], alpha=0.5, label='original', width=width)
    ax.bar(binEdges, reconstructed_hist[0], alpha=0.5, label='reconstructed', width=width)
    plotname = hist_path.split('/')[-1]
    ax.set_title(f'{plotname} {run}')
    ax.legend(loc='best')
    # create directory to save plot
    Path(f'plots/{run}').mkdir(parents=True, exist_ok=True)
    fig.savefig(f'plots/{run}/{plotname}.png')
    plt.close('all')
    
    if sse > threshold: 
        fig2, ax2 = plt.subplots()
        ax2.bar(binEdges, np.square(original_hist[0] - reconstructed_hist[0]), alpha=0.5, width=width)
        ax2.set_title(f'SSE {plotname} {run}')
        fig2.savefig(f'plots/{run}/{plotname}-SSE.png')
        plt.close('all')
    

