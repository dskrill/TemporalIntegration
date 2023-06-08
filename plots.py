import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.io import loadmat

colormap = loadmat('/scratch/snormanh_lab/dskrill/IntegrationAnalysis/code/colormap-custom-lightblue-to-yellow1.mat')['cmap']
colormap = matplotlib.colors.ListedColormap(colormap)
plt.style.use('seaborn-whitegrid')
rcParams['figure.dpi']= 1200


def plot_difference_tensor_individual_layers(D,layers):
    """Plot the difference tensor for each layer."""
    D = D.mean(-1)
    D = np.moveaxis(D, 0,-1)
    n_stim_time, n_model_time, n_layers = D.shape
    boundary_times = np.arange(0,D.shape[0],sentence_len)

    for i in layers:
        ax = plt.figure(figsize = (2,2)).gca()
        mdiag = np.median(np.diag(D[:,:,i]))
        ax.imshow(D[:,:,i].T/mdiag,cmap=colormap,interpolation='none',vmin=0,vmax=1)

        first_words = np.array([0,1,2,3,4])*sentence_len
        ax.set_xticks(first_words,labels = '')
        ax.set_yticks(first_words,labels = '')
        ax.grid(alpha=.5,linewidth=1)
        if sentence_len < 40:
            ax.set_ylim([n_stim_time-1,sentence_len])
            ax.set_xlim([sentence_len,n_model_time-1])
        ax.grid(alpha=.5,linewidth=1)

def plot_fit_params_individual_plots(stacked_fits):

    for i,(plot_idx,to_plot,r) in enumerate(zip([2,0,1],['c','a','b'], [[0,1],[0,1.5],[0,12],])):   
        axes = plt.figure(figsize=(3,2)).gca()
    

        axes.violinplot(stacked_fits[:,1:,plot_idx],showextrema=False,showmedians=True)

        axes.tick_params(labelright= False,labeltop= False,labelleft= False, labelbottom= False)
        axes.set_xticks(range(stacked_fits.shape[1]))
        axes.set_ylim(r)

def average_over_units_indiv_plots(layers,fitobj,D_delta):  
    # plot differences and fits
    
    idx = 0
    t = np.arange(0,max_window_size)
    D_delta_all = np.stack(D_delta,0).mean(0)
    for k,layer in enumerate(layers):
        ax = plt.figure(figsize=(2,3)).gca()


        for u in D_delta:
            ax.plot(t, np.median(u[:, :, layer],axis=1), color='gray',label = 'Indiv.' '\n' 'Units',linewidth=.5)
        ax.plot(t, np.median(D_delta_all[:, :, layer], axis=1), color = 'black', linestyle="-", linewidth=2,label="Mean")
        ax.set_ylim([0, 1.2])
        
        
        ax.set_xlim([0, max_window_size - 1])

        #     ax.set_xticks([])
        ax.tick_params(labelright= False,labeltop= False,labelleft= False, labelbottom= False)
        ax.grid(False)        
        ax.invert_xaxis()
        idx += 1

def single_units(units,layers,fitobj,D_delta):
    # plot differences and fits
    lsize=1.5
    idx = 0
    t = np.arange(0,max_window_size)
    for k,(unit,layer) in enumerate(zip(units,layers)):
        ax = plt.figure(figsize=(2,2)).gca()


        
       
        ax.plot(t, np.mean(D_delta[unit][:, :, layer], axis=1), color = 'black', linestyle = "-", linewidth=lsize+2,label="Observed",alpha=1)
        ax.plot(t, fit_func(t, *fitobj[unit][layer]), color='red',linestyle="--", linewidth=lsize,
            label='Estimated fit')
        ax.set_ylim([0, 1])

        ax.set_xlim([0, max_window_size - 1])
        
        ax.invert_xaxis()
        ax.tick_params(labelright= False,labeltop= False,labelleft= False, labelbottom= False)
        idx += 1

def yoking_plots(D,sentence_len,plot_type="vector"):
    assert plot_type in ["vector","summary"]
    D = np.moveaxis(D, 0,-1)
    n_swap_time, n_model_time,n_units, n_layers = D.shape
    boundary_times = np.arange(0,D.shape[0],sentence_len)

    # compute first order differences
    n_offdiag = sentence_len - 1 # number of off-diagonal entries to consider
    fdiff = np.full((n_swap_time, n_layers,n_units), np.nan)
    for i in range(n_layers):
        for u in range(n_units):
            mdiag = np.median(np.diag(D[:,:,u,i]))

            for j in range(n_swap_time-n_offdiag-1):
                d = D[j+1,j+1:(j+1+n_offdiag),u,i]-D[j,j:(j+n_offdiag),u,i]
                fdiff[j+1,i,u] = np.mean(d)/mdiag
    if plot_type == "vector":
        if model == "gpt2":
            yL = np.array([-1.05,1.05])*np.nanmax(fdiff[sentence_len:-sentence_len])
            layers = [12]
            for i, (layer) in enumerate(layers):
                ax = plt.figure(figsize = (2,2)).gca()
                ax.set_ylim(yL)
                ax.plot(np.arange(sentence_len,n_swap_time), fdiff[sentence_len:,i,:],color='grey',alpha=.5,label = "Indiv. units")
                for l in range(len(boundary_times)):
                    ax.plot(boundary_times[l]*np.array([1,1]), yL, 'k--', label = "Sentence boundary")
                ax.plot(np.arange(sentence_len,n_swap_time), fdiff.mean(-1)[sentence_len:,i], linewidth=1.5,color='red',label = "Mean")
            

                ax.set_xticks(boundary_times,labels = '')
                ax.set_yticks([-.1,.1],'')
                ax.set_xlim([11, 47])
    elif plot_type == "summary":

        ax = plt.figure(figsize = (2,2)).gca()
        n_sentences = 5
        fdiff_circ = fdiff[sentence_len:].reshape(sentence_len, n_sentences-1, n_layers,n_units, order='F')
        yoking_index = np.nanmean(fdiff_circ[0,:,:,:], axis=0)
        ax.violinplot(positions = np.arange(0,n_layers), dataset=yoking_index.T, showmeans=False, showextrema=False, showmedians=True,widths=0.8)
        ax.axhline(0, color='k', linestyle='--', linewidth=1)
        ax.tick_params(labelright= False,labeltop= False,labelleft= False, labelbottom= False)
        ax.set_xticks(range(n_layers))
        ax.set_yticks([0,.05,.1,.15,.2],'')


