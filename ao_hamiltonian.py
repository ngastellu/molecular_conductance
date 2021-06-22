#!/usr/bin/env pythonw

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from find_edge_carbons import concave_hull

def get_Natoms(infile):
    with open(infile) as fo:
        line1 = fo.readline()
        L = len(line1.split())
        Natoms = L - 5
        if L < 502:
            return Natoms
        else:
            init_line = False
            L = 0
            while not init_line:
                Natoms += L
                l = fo.readline()
                L = len(l.split())
                init_line = (L == 502)

    return Natoms


def read_MO_file(infile):
    """Reads MO coefs output file from QCFFPI and returns a list of atomic positions and a AO -> MO
    transformation matrix with elements M_ij = <AO_i|MO_j>."""
    
    Natoms = get_Natoms(infile)
    with open(infile) as fo:
        lines = fo.readlines()

    positions = np.zeros((Natoms,3),dtype=float)
    MO_matrix = np.zeros((Natoms,Natoms),dtype=np.float64)

    if Natoms <= 497:
        nlines_per_atom = 1
    else:
        nlines_per_atom = int(1 + np.ceil((Natoms-497)/500))

    for k, line in enumerate(lines):
        #print(k)
        atom_index = k // nlines_per_atom
        if atom_index == Natoms: break
        split_line = line.split()
        if k % nlines_per_atom == 0:
            counter = 0
            positions[atom_index,:] = list(map(float,split_line[2:5]))
            MO_matrix[atom_index,:497] = list(map(float,split_line[5:]))
            counter += 497
        else:
            n = len(split_line)
            MO_matrix[atom_index,counter:counter+n] = list(map(float,split_line))
            counter += n

    return positions, MO_matrix


def read_energies(orb_file):
    """Reads energies from QCCFPI output file `orb_file` and returns two arrays [i,e_i] 
    (where i labels the MOs) of the energies of occupied and virtual MOs.
    *** ASSUMES ENERGIES ARE SORTED *** """

    with open(orb_file) as fo:
        lines = fo.readlines()
    
    all_energies = np.array([list(map(float,l.split())) for l in lines[:int(len(lines)/2)]])
    lumo_index = int(len(all_energies)/2)

    occupied = all_energies[:lumo_index,:]
    virtual = all_energies[lumo_index:,:]

    #print(np.max(all_energies[:,1]) - np.min(all_energies[:,1]))

    return occupied, virtual

def AO_hamiltonian(MO_file,orb_file,delta=-1):
    """Expresses the reduced Hamiltonian of MOs within `delta` hartrees of the HOMO/LUMO
    in the AO basis. If `delta` = -1, then the full Hamiltonian in the AO basis is returned;
    it is furthermore not split into an occupied and virtual Hamiltonian.
    *** ASSUMES ENERGIES ARE SORTED *** """

    _, M = read_MO_file(MO_file)
    occ, virt = read_energies(orb_file)

    N = M.shape[0]
    #print(N)

    for orbs in [occ,virt]:
        sorted_indices = np.argsort(orbs[:,1])

        if not np.all(sorted_indices == np.arange(N/2)):
            print('Energies unsorted in orb_file!')
            print(sorted_indices.shape)
            print(np.arange(N).shape)
            print((sorted_indices != np.arange(N)))
            #orbs = orbs[sorted_indices]
        
    if delta > 0:
        occ = occ[:,1]
        virt = virt[:,1]
        E_homo = occ[-1]
        E_lumo = virt[0]

        relevant_occ_inds = (occ >= E_homo - delta).nonzero()[0]
        relevant_virt_inds = (virt <= E_lumo + delta).nonzero()[0] 

        print('Number of occupied MOs in reduced hamiltonian = ',relevant_occ_inds.shape)
        print('Number of virtual MOs in reduced hamiltonian = ',relevant_virt_inds.shape)
        
        occ_levels = occ[relevant_occ_inds]
        virt_levels = virt[relevant_virt_inds]

        D_occ = np.zeros((N,N))
        D_occ[relevant_occ_inds,relevant_occ_inds] = occ_levels
        print('D_occ:\n')
        print(D_occ)
        print('\n')

        D_virt = np.zeros((N,N))
        D_virt[relevant_virt_inds+(N//2),relevant_virt_inds+(N//2)] = virt_levels
        print('D_virt:\n')
        print(D_virt)
        print('\n')
        
        AO_hamiltonian_occ = M @ D_occ @ (M.T)
        AO_hamiltonian_virt = M @ D_virt @ (M.T)

        return AO_hamiltonian_occ, AO_hamiltonian_virt
    
    else: #delta = -1 ==> return full Hamiltonian in AO basis
        D = np.diag(np.hstack((occ[:,1],virt[:,1])))
        AO_hamiltonian = M @ D @ (M.T)
        return AO_hamiltonian


def inverse_participation_ratios(MO_matrix):

    return np.sum(MO_matrix**4, axis = 0)


def MO_couplings(pos,M,gamma,edge_tol=3.0,return_separate=True):
    """Computes each MO's average coupling to the leads.
    If `return_separate` is set to `True`, then the coupling to each lead (i.e. left
    and right) will be computed and returned separately. If `return_separate` is set to 
    `False`, each MO's total coupling (i.e. <MO|GammaL+GammaR|MO>) will be computed and
    returned."""

    if pos.shape[1] == 3:
        pos = pos[:,:2] #keep only x and y coords

    edge_bois = concave_hull(pos,3)
    xmin = np.min(pos[:,0])
    xmax = np.max(pos[:,0])
    right_edge = edge_bois[edge_bois[:,0] > xmax - edge_tol]
    left_edge = edge_bois[edge_bois[:,0] < xmin + edge_tol]

    right_inds = np.zeros(right_edge.shape[0],dtype=int)
    left_inds = np.zeros(left_edge.shape[0],dtype=int)
    
    for k, r in enumerate(right_edge):
        print(np.all(pos == r, axis=1).nonzero()[0])
        right_inds[k] = np.all(pos == r, axis=1).nonzero()[0]

    for k, r in enumerate(left_edge):
        left_inds[k] = np.all(pos == r, axis=1).nonzero()[0]
    

    N = M.shape[0]
    gammaR = np.zeros((N,N),dtype=float)
    gammaL = np.zeros((N,N),dtype=float)

    gammaR[right_inds,right_inds] = gamma
    gammaL[left_inds,left_inds] = gamma

    GmR = (M.T) @ gammaR @ M
    GmL = (M.T) @ gammaL @ M
    
    couplingsR = np.diag(GmR)
    couplingsL = np.diag(GmL)

    if return_separate:
        return couplingsL, couplingsR
    else:
        return couplingsL + couplingsR 


def MO_rgyr(pos,MO_matrix,n,center_of_mass=None):

    psi = MO_matrix[:,n]**2

    if np.all(center_of_mass) == None:
        com = psi @ pos

    else: #if center of mass has already been computed, do not recompute
        com = center_of_mass

    R_squared = (pos*pos).sum(axis=-1) #fast way to compute square length of all position vectors
    R_squared_avg = R_squared @ psi

    return np.sqrt(R_squared_avg - (com @ com))


def all_rgyrs(pos,MO_matrix,centers_of_mass=None):

    psis = MO_matrix**2

    if np.all(centers_of_mass) == None:
        coms = (psis.T) @ pos

    else: #if centers of mass have already been computed, do not recompute
        coms = centers_of_mass

    R_squared = (pos*pos).sum(-1)
    R_squared_avg = R_squared @ psis

    coms_squared = (coms*coms).sum(-1)

    return np.sqrt(R_squared_avg - coms_squared)
    

def plot_MO(pos,MO_matrix,n,dotsize=45.0,show_COM=False,show_rgyr=False):

    psi = MO_matrix[:,n]**2

    rcParams['text.usetex'] = True
    rcParams['font.size'] = 16
    rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    #if plot_type == 'nanoribbon':
    #    #rcParams['figure.figsize'] = [30.259946/2,7/2]
    #    figsize = [12,11/2]
    #elif plot_type == 'square':
    #    figsize = [4,4]  
    #else:
    #    print('Invalid plot type. Using default square plot type.')
    #    figsize = [4,4]

    fig, ax1 = plt.subplots()
    #fig.set_size_inches(figsize,forward=True)

    ye = ax1.scatter(pos.T[0,:],pos.T[1,:],c=psi,s=dotsize,cmap='plasma')
    cbar = fig.colorbar(ye,ax=ax1,orientation='vertical')
    plt.suptitle('$\langle\\varphi_n|\psi_{%d}\\rangle$'%n)
    ax1.set_xlabel('$x$ [\AA]')
    ax1.set_ylabel('$y$ [\AA]')
    ax1.set_aspect('equal')
    if show_COM or show_rgyr:
        com = psi @ pos[:,:2]
        ax1.scatter(*com, s=dotsize+1,marker='*',c='r')
    if show_rgyr:
        rgyr = MO_rgyr(pos,MO_matrix,n,center_of_mass=com)
        loc_circle = plt.Circle(com, rgyr, fc='none', ec='r', ls='--', lw=1.0)
        ax1.add_patch(loc_circle)
    plt.show()

    

# ******* MAIN *******

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    eV2Ha = 0.0367493 #eV to Ha conversion factor

    qcffpi_datadir = '../../simulation_outputs/qcffpi_data'
    mo_datadir = os.path.join(qcffpi_datadir,'MO_coefs')
    orb_datadir = os.path.join(qcffpi_datadir,'orbital_energies')

    L = 102

    #MOfile = os.path.join(mo_datadir,'MOs_pCNN_MAC_%dx%d.dat'%(L,L))
    #orbfile = os.path.join(orb_datadir,'orb_energy_pCNN_MAC_%dx%d.dat'%(L,L))

    MOfile = os.path.join(mo_datadir,'MOs_kMC_MAC_clean.dat')
    orbfile = os.path.join(orb_datadir,'orb_energy_kMC_MAC_clean.dat')

    energy_window = 1 #eV
    H_occ, H_virt = AO_hamiltonian(MOfile,orbfile,energy_window*eV2Ha)

    print(H_occ.shape)
    print(H_virt.shape)

    Jocc, Jvirt = read_energies(orbfile)
    Jall = np.vstack((Jocc,Jvirt))

    Ehomo = Jocc[-1,1]
    print('HOMO energy = %f Ha'%Ehomo)
    Elumo = Jvirt[0,1]
    print('LUMO energy = %f Ha'%Elumo)

    plt.plot(*Jall.T,'ro',ms=10)
    #plt.axhline(Ehomo,'k--',lw=0.8)
    #plt.axhline(Elumo,'k--',lw=0.8)
    plt.show()


    plt.imshow(np.abs(H_occ))
    plt.colorbar()
    plt.suptitle('HOMO-27:HOMO')
    plt.show()

    plt.imshow(np.abs(H_virt))
    plt.colorbar()
    plt.suptitle('LUMO:LUMO+35')
    plt.show()

    H = AO_hamiltonian(MOfile,orbfile,-1)
    plt.imshow(np.abs(H))
    plt.suptitle('Full Hamiltonian in AO_basis')
    plt.colorbar()
    plt.show()

    pos, _ = read_MO_file(MOfile)
    total_couplings = np.sum(H,axis=0)

    fig, ax = plt.subplots()
    ye = ax.scatter(*pos.T[:2],c=np.abs(total_couplings),s=10.0,cmap='plasma')
    cbar = fig.colorbar(ye,ax=ax)
    ax.set_aspect('equal')
    plt.show()

    np.save('H_occ.npy',H_occ)
    np.save('H_virt.npy',H_virt)
    np.save('H_full.npy',H)

    #check off-diagonal elements

    off_diagonal_elements = H[~np.eye(H.shape[0],dtype=bool)]
    print(off_diagonal_elements.shape)
    print('Average coupling: ', np.mean(off_diagonal_elements))
    print('Coupling standard deviation: ', np.std(off_diagonal_elements))

    fig, ax = plt.subplots(1,1)

    hist1, bins1 = np.histogram(off_diagonal_elements,300)
    width1 = bins1[1] - bins1[0]
    center1 = (bins1[1:] + bins1[:-1])/2
    ax.bar(center1,hist1,align='center',width=width1)
    ax.set_title('Couplings')
    plt.show()

    large_coupling_inds = (center1 >= -0.12)*(center1 <= -0.06)
    print(np.sum(hist1[large_coupling_inds]))
