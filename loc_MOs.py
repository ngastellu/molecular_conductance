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
rcParams['font.size'] = 16
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

#nanoribbons = {'zigzag': [[13, 18, 27, 33, 40, 48], 17],
#        'armchair': [[11, 15, 23, 28, 35, 42], 10],
#        'pCNN_102': [[20, 28, 40, 50, 62, 73], 26]}

nanoribbons = {'zigzag': [[33, 40, 48], 17],
        'armchair': [[28, 35, 42], 10],
        'pCNN_102': [[20,50, 62, 73], 26]}

loc_data = {'zigzag':[[],[],[],[],[]],
        'armchair':[[],[],[],[],[]],
        'pCNN_102':[[],[],[],[],[]]}

for s, Ls in nanoribbons.items():

    Nxs, Ny = Ls

    for Nx in Nxs:
        MOfile = os.path.join(MO_datadir,'MOs_%s_%dx%d.dat'%(s,Nx,Ny))
        orbfile = os.path.join(orb_datadir,'orb_energy_%s_%dx%d.dat'%(s,Nx,Ny))

        pos, M = read_MO_file(MOfile)
        occ, virt = read_energies(orbfile)
        occ = occ[:,1]*Ha2eV
        virt = virt[:,1]*Ha2eV
        energies = np.hstack((occ,virt))

        eF = (occ[-1] + virt[0])/2.0

        #e_range = (np.max(energies) - np.min(energies))*5/4
        #shifted_e = (energies/e_range) - np.min(energies)/e_range
        #clrs = [cm.plasma(x) for x in shifted_e]

        iprs = 1.0/np.sqrt(inverse_participation_ratios(M))
        rgyrs = all_rgyrs(pos,M)

        fig, ax1 = plt.subplots()

        ye = ax1.scatter(iprs,rgyrs,marker='o',c=energies,s=4.0)
        cbar = fig.colorbar(ye, ax=ax1)
        ax1.set_ylabel('$\sqrt{\langle R^2\\rangle - \langle R\\rangle^2}$')
        ax1.set_xlabel('1/$\sqrt{IPR}$')
        plt.suptitle('%s %dx%d'%(' '.join(s.split('_')),Nx,Ny))
        plt.show()

        discrepancy = np.abs(iprs - rgyrs)
        #discrepancy = iprs - rgyrs
  
        #plt.plot(energies-eF,discrepancy,'ro')
        #plt.xlabel('$\\varepsilon - \\varepsilon_F$ [eV]')
        #plt.ylabel('$\left|R_g - \\frac{1}{\sqrt{IPR}}\\right|$')
        #plt.show()

        ## plot states with the greatest discrepancy between Rg and IPR
        #discrep_inds = np.argsort(discrepancy)[-3:]
        #for n in discrep_inds:
        #    plot_MO(pos,M,n,show_rgyr=True)

        #agreement_inds = np.argsort(discrepancy)[0:3]

        #for n in agreement_inds:
        #    plot_MO(pos,M,n,show_rgyr=True)

        
        #hist, bins = np.histogram(iprs,100)
        #width = bins[1] - bins[0]
        #center = (bins[1:] + bins[:-1])/2

        #plt.bar(center,hist,align='center',width=width)
        #plt.xlabel('$1/\sqrt{IPR}$')
        #plt.suptitle('%s %dx%d'%(' '.join(s.split('_')),Nx,Ny))
        #plt.show()

        #hist, bins = np.histogram(rgyrs,100)
        #width = bins[1] - bins[0]
        #center = (bins[1:] + bins[:-1])/2

        #plt.bar(center,hist,align='center',width=width)
        #plt.xlabel('$\sqrt{\langle R^2\\rangle - \langle R\\rangle^2}$')
        #plt.suptitle('%s %dx%d'%(' '.join(s.split('_')),Nx,Ny))
        #plt.show()

        L = np.max(pos[:,0]) - np.min(pos[:,0])
        loc_data[s][0].append(L)
        loc_data[s][1].append(np.mean(iprs))
        loc_data[s][2].append(np.std(iprs))
        loc_data[s][3].append(np.mean(rgyrs))
        loc_data[s][4].append(np.std(rgyrs))

#for k in range(6):
#    data = 
#
#hist, bins = np.histogram(energies,100)
#width = bins[1] - bins[0]
#center = (bins[1:] + bins[:-1])/2
#
#plt.bar(center,hist/energies.shape[0],align='center',width=width)
#plt.ylabel('Density of states [arb. units]')
#plt.xlabel('$E$ [Ha]')
##plt.title('pCNN MAC 160x160')
#plt.show()

#markers = ['h','H','o']
clrs = ['r','b','k']

fig, axs = plt.subplots(2,1)

for k, s in enumerate(loc_data):
    data = loc_data[s]
    Lxs = data[0]
    avg_iprs, std_iprs = data[1:3] 
    avg_rgyrs, std_rgyrs = data[3:]
    
    axs[0].plot(Lxs, avg_iprs,'o',c=clrs[k],label=' '.join(s.split('_')))
    axs[1].plot(Lxs, avg_rgyrs,'o',c=clrs[k],label=' '.join(s.split('_')))

axs[0].set_xlabel('$L_x$ [\AA]')
axs[1].set_xlabel('$L_x$ [\AA]')

axs[0].set_ylabel('$\langle 1/\sqrt{IPR}\\rangle$')
axs[1].set_ylabel('$\langle R_g\\rangle$')

plt.legend()
plt.show()
