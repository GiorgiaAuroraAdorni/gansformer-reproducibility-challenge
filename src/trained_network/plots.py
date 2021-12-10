import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

def annotate_min(y, idx, ax=None):
    ymin = y.min()
    text = "{:.0f}".format(ymin)

    if not ax:
        ax = plt.gca()

    multip = 10 * idx

    ax.annotate(text, xy=(300, ymin), xycoords='data',
                xytext=(-30 - multip, 30 - multip), textcoords='offset points',
                arrowprops=dict(arrowstyle="->")
                )

def plot_learning_curve(data, labels, title):
    """
    Generate a simple plot of the learning performance.

    @param data:
    @param labels:
    @param title:
    @return: plot
    """
    plt.figure()
    plt.title(title, fontweight="bold")

    plt.xlabel("Step")
    plt.ylabel("FID score")

    # plt.semilogy()
    plt.xlim(left=-20, right=320)
    plt.ylim(bottom=-20, top=450)
    plt.axis('tight')

    plt.grid()
    for idx, el in enumerate(data):
        plt.plot(el['snapshot'], el['fid10k'], 'o-',  label=labels[idx])
        annotate_min(el['fid10k'], idx)

    plt.legend(loc="best")

    return plt


# opening the CSV file
data_style = pd.read_csv('scores/FIDScore_Stylegan2_300kimg.csv', delimiter=" ")
data_ganfsimpnatt = pd.read_csv('scores/FIDScore_GanformerSimplexNoAtt_300kimg.csv', delimiter=" ")
data_ganfsimpatt = pd.read_csv('scores/FIDScore_GanformerSimplexAtt_300kimg.csv', delimiter=" ")

labels = ["StyleGAN2",
          "GANformer with Simplex attention \n(no discriminator attention)",
          "GANformer with Simplex attention \n(and discriminator attention)"]

title = "Learning Performance"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.

plot_learning_curve([data_style, data_ganfsimpnatt, data_ganfsimpatt], labels, title)

plt.savefig('out_imgs/FIDscore.pdf', dpi=600)
plt.close()
