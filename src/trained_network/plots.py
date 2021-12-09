import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

def plot_learning_curve(data, labels, title):
    """
    Generate a simple plot of the learning performance.

    @param data:
    @param title:
    @return: plot
    """
    plt.figure()
    plt.title(title, fontweight="bold")

    plt.xlabel("Step")
    plt.ylabel("FIS score")

    # top = max(max(data[0]['fid10k']), max(data[1]['fid10k']))
    plt.ylim(bottom=0, top=450)

    plt.grid()
    for idx, el in enumerate(data):
        plt.plot(el['snapshot'], el['fid10k'], 'o-',  label=labels[idx])

    plt.legend(loc="best")

    return plt


# opening the CSV file
data_style = pd.read_csv('scores/FIDScore_Stylegan2_300kimg.csv', delimiter=" ")
data_ganfsimpnat = pd.read_csv('scores.FIDScore_GanformerSimplexNoAtt_300kimg.csv', delimiter=" ")

labels = ["StyleGAN2", "GANformer with Simplex attention \n(no discriminator attention)"]
title = "Learning Performance"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.

plot_learning_curve([data_style, data_ganfsimpnat], labels, title)

plt.savefig('out_imgs/FIDscore.pdf', dpi=600)
plt.close()
