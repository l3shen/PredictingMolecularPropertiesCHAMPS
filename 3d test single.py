import tkinter
import ase
import pandas as pd
from ase import Atoms
import ase.visualize
import random


struct_file = pd.read_csv("C:\\Users\\azn_k\\PycharmProjects\\PredictingNMR\\structures.csv")


def view(molecule):
    # Select a molecule
    mol = struct_file[struct_file['molecule_name'] == molecule]

    # Get atomic coordinates
    xcart = mol.iloc[:, 3:].values

    # Get atomic symbols
    symbols = mol.iloc[:, 2].values

    # Display molecule
    system = Atoms(positions=xcart, symbols=symbols)
    print('Molecule Name: %s.' % molecule)
    return ase.visualize.view(system, viewer="ase")


random_molecule = random.choice(struct_file['molecule_name'].unique())
view('dsgdb9nsd_000001')


