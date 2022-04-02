import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm.notebook import trange

#============================================================================================
def sma(input_list, n):
    sma_mask = np.ones(n)/n
    output = np.convolve(input_list, sma_mask)[:1-n]
    return output

def eq_time(E_list, bins=50):
    histogram, bin_edges = np.histogram(E_list, bins=bins)
    peak_value = bin_edges[np.argmax(histogram)+1]

    eq_steps = None

    for i in range(len(E_list)):
        if E_list[i]<peak_value:
            eq_steps = i
            break
    
    if eq_steps == None:
        raise NameError('eq_steps not found somehow')

    return eq_steps

def draw(series):

    ims = []
    fig, ax = plt.subplots(figsize=(5,5))

    for i in trange(series.get_len()):
        
        im = ax.imshow(series.get_frame_np(i))
        ax.set_title('T='+str(series.T))
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                    repeat_delay=0)
    plt.show()