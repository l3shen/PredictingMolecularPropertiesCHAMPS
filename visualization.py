import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


train = pd.read_csv("C:\\Users\\lawre\\Downloads\\train.csv (1)\\train.csv")
structures = pd.read_csv("C:\\Users\\lawre\\Downloads\\train.csv (1)\\structures.csv")
structures.head()
#print(structures.head())

#gotta try and plot this bad boy in 2d-2d

M = 8000
fig, ax = plt.subplots(1, 3, figsize=(20, 5))

colors = ["black", "gold", "blue", "darkred", "purple"]
atoms = structures.atom.unique()

for n in range(len(atoms)):
    ax[0].scatter(structures.loc[structures.atom==atoms[n]].x.values[0:M],
                  structures.loc[structures.atom==atoms[n]].y.values[0:M],
                  color=colors[n], s=2, alpha=0.5, label=atoms[n])
    ax[0].legend()
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")

    ax[1].scatter(structures.loc[structures.atom==atoms[n]].x.values[0:M],
                  structures.loc[structures.atom==atoms[n]].z.values[0:M],
                  color=colors[n], s=2, alpha=0.5, label=atoms[n])
    ax[1].legend()
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("z")

    ax[2].scatter(structures.loc[structures.atom==atoms[n]].y.values[0:M],
                  structures.loc[structures.atom==atoms[n]].z.values[0:M],
                  color=colors[n], s=2, alpha=0.5, label=atoms[n])
    ax[2].legend()
    ax[2].set_xlabel("y")
    ax[2].set_ylabel("z")

plt.show()