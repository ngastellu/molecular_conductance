#!/usr/bin/env pythonw

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, cm
from ao_hamiltonian import read_MO_file, read_energies, inverse_participation_ratios, all_rgyrs, plot_MO


Ha2eV = 27.2114

qcffpi_datadir = '../../simulation_outputs/qcffpi_data'

MO_datadir = os.path.join(qcffpi_datadir,'MO_coefs')
orb_datadir = os.path.join(qcffpi_datadir,'orbital_energies')

rcParams['text.usetex'] = True
rcParams['font.size'] = 17
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

MOfile = os.path.join(MO_datadir,'MOs_kMC_MAC_20x40_178l.dat')
orbfile = os.path.join(orb_datadir,'orb_energy_kMC_MAC_20x40_178l.dat')

pos, M = read_MO_file(MOfile)
occ, virt = read_energies(orbfile)
occ = occ[:,1]*Ha2eV
virt = virt[:,1]*Ha2eV
energies = np.hstack((occ,virt))

eF = (occ[-1] + virt[0])/2.0

iprs = 1.0/np.sqrt(inverse_participation_ratios(M))
rgyrs = all_rgyrs(pos,M)

fig, ax1 = plt.subplots()

ye = ax1.scatter(iprs,rgyrs,marker='o',c=energies,s=4.0)
cbar = fig.colorbar(ye, ax=ax1)
ax1.set_ylabel('$\sqrt{\langle R^2\\rangle - \langle R\\rangle^2}$')
ax1.set_xlabel('1/$\sqrt{IPR}$')
plt.show()
