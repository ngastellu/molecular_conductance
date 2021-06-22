#!/usr/bin/env pythonw

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, cm
from ao_hamiltonian import read_energies, read_MO_file,\
        AO_hamiltonian, AO_gammas, MCOs, MO_couplings,\
        MO_rgyr, inverse_participation_ratios,\
        interference_matrix_MCO_evals, inverse_participation_ratios

Ha2eV = 27.2114
gamma = 0.1

rcParams['text.usetex'] = True
rcParams['font.size'] = 16
rcParams['text.latex.preamble'] = r'\usepackage{amsmath},\usepackage{amsfonts}'

qcffpi_datadir = '../../simulation_outputs/qcffpi_data'

sysnames = ['zigzag_13x17', 'armchair_11x10', 'pCNN_102_20x26','pCNN_MAC_102x102']
lbls = ['Armchair $13\\times 17$', 'Zigzag $11\\times 10$', 'MAC $20\\times 26$', 'MAC 102\AA$\\times$102\AA']

for sysname, lbl in zip(sysnames,lbls):

    MO_datadir = os.path.join(qcffpi_datadir,'MO_coefs')
    MOfile = os.path.join(MO_datadir,'MOs_%s.dat'%sysname)

    orb_datadir = os.path.join(qcffpi_datadir,'orbital_energies')
    orbfile = os.path.join(orb_datadir,'orb_energy_%s.dat'%sysname)

    pos, M = read_MO_file(MOfile)
    occ, virt = read_energies(orbfile)

    occ = occ[:,1] * Ha2eV
    virt = virt[:,1] * Ha2eV

    eF = (occ[-1] + virt[0])/2
    E = eF

    energy_lvls = np.hstack((occ,virt))
    Hao = AO_hamiltonian(M,energy_lvls)

    gamL, gamR = AO_gammas(pos,M,gamma)

    zs, P, zbars, Pbar = MCOs(Hao,gamL,gamR,sort=True,return_evals=True)

    Q = interference_matrix_MCO_evals(E,M,energy_lvls,gamL,gamR)
    tola = 3e-3
    print('Energy lvls match up to %e = '%(tola), np.allclose(energy_lvls, np.real(zs),atol=tola,rtol=0))

    plt.plot(np.real(zs), np.abs(np.imag(zs)),'ro',ms=3.0)
    plt.xlabel('$\mathfrak{Re}\{\lambda_j\}$')
    plt.ylabel('$\mathfrak{Im}\{\lambda_j\}$')
    plt.suptitle(lbl)
    plt.show()

    GMOL, GMOR = MO_couplings(pos, M, gamma)
    harmonic_mean = GMOL*GMOR/(GMOL+GMOR)

    plt.plot(energy_lvls, np.abs(harmonic_mean), 'ro', ms=3.0)
    plt.xlabel('$\\varepsilon_j$ [eV]')
    plt.ylabel('$\\frac{\Gamma_L\Gamma_R}{\Gamma_L + \Gamma_R}$ [eV]')
    plt.suptitle(lbl)
    plt.show()


    fig, ax2 = plt.subplots()
    ye = ax2.scatter(GMOL,GMOR,c=harmonic_mean,s=4.0,cmap='plasma')
    cbar = fig.colorbar(ye, ax=ax2)
    ax2.set_xlabel('$\Gamma_L$')
    ax2.set_ylabel('$\Gamma_R$')
    ax2.text(1.15,1.05,'$\left\langle\\frac{\Gamma_R\Gamma_L}{\Gamma_L + \Gamma_R}\\right\\rangle$ [eV]',
            transform=ax2.transAxes,verticalalignment='top',fontsize=17)
    plt.suptitle(lbl)
    plt.show()

    fig, ax3 = plt.subplots()
    ye = ax3.scatter(harmonic_mean,np.abs(np.imag(zs)),c=harmonic_mean,s=4.0,cmap='plasma')
    cbar = fig.colorbar(ye, ax=ax3)
    ax3.set_xlabel('$\left\langle\\frac{\Gamma_R\Gamma_L}{\Gamma_L + \Gamma_R}\\right\\rangle$ [eV]')
    ax3.set_ylabel('$\mathfrak{Im}\{\lambda_j\}$')
    #ax3.text(1.15,1.05,,
            #transform=ax2.transAxes,verticalalignment='top',fontsize=17)
    plt.suptitle(lbl)
    plt.show()


    plt.plot(energy_lvls, np.abs(energy_lvls - np.real(zs)), 'ro', ms=1.0)
    plt.xlabel('$\\varepsilon_j$ [eV]')
    plt.ylabel('$|\\varepsilon_j - \mathfrak{Re}\{\lambda_j\}|$ [eV]')
    plt.suptitle(lbl)
    plt.show()

    gamL_MCO = np.conj(Pbar.T) @ gamL @ P
    gamR_MCO = np.conj(Pbar.T) @ gamR @ P
    harmonic_mean_MCO = np.conj(Pbar.T) @ harmonic_mean @ P

    fig, ax4 = plt.subplots()
    ye = ax4.scatter(GMOL,GMOR,c=harmonic_mean,s=4.0,cmap='plasma')
    cbar = fig.colorbar(ye, ax=ax4)
    ax4.set_xlabel('$\\tilde{\Gamma}_L$')
    ax4.set_ylabel('$\\tilde{\Gamma}_R$')
    ax4.text(1.15,1.05,'$\left\langle\\frac{\\tilde{\Gamma}_R\\tilde{\Gamma}_L}{\\tilde{\Gamma}_L + \\tilde{\Gamma}_R}\\right\\rangle$ [eV]',
            transform=ax2.transAxes,verticalalignment='top',fontsize=17)
    plt.suptitle(lbl)
    plt.show()

    IPR_MCO = np.sum(np.abs(P)**4, axis=0)
    IPR_MCO_bar = np.sum(np.abs(Pbar)**4, axis=0)
    IPR_MO = inverse_participation_ratios(M)

    fig, ax5 = plt.subplots()
    ye = ax5.scatter(IPR_MCO,IPR_MCO_bar,c=IPR_MO,s=4.0,cmap='plasma')
    cbar = fig.colorbar(ye, ax=ax5)
    ax5.set_xlabel('IPR($|\psi_j\\rangle$)')
    ax5.set_ylabel('IPR($|\\bar{\psi}_j\\rangle$)')
    ax5.text(1.15,1.05,'IPR($|\chi_j\\rangle$)',
            transform=ax5.transAxes,verticalalignment='top',fontsize=17)
    plt.suptitle(lbl)
    plt.show()

    D = np.abs(P - Pbar)
    plt.imshow(D)
    plt.colorbar()
    plt.show()

