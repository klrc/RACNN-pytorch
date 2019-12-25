import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def visualize():
    data = [float(line.split('loss: ')[-1].replace('\n', '')) for line in open('external/training.log', 'r').readlines() if 'loss: ' in line]
    sns.lineplot(x=np.linspace(0, 100, len(data)), y=data)
    plt.show()


visualize()
