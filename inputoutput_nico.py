#!/usr/bin/env python

import numpy as np

def get_coords(infile):
    
    suffix = infile.split('.')[-1]

    if suffix == 'npy':
        coords = np.load(infile)
    
    elif suffix == 'xyz':
        coords = read_xyz(infile)
    
    elif suffix == 'xsf':
        coords, *_ = read_xsf(infile)
    
    else:
        print('Invalid file type: %s\nValid file types: npy, xyz, xsf\nReturning 0'%suffix)
        coords = 0
    
    return coords

def xsf2xyz_MAC(filepath):
    prefix = '.'.join(filepath.split('.')[:-1])
    atoms, *_ = read_xsf(filepath)
    symbols = ['C']*len(atoms)
    outfile = prefix + '.xyz'
    write_xyz(atoms,symbols,outfile)


def read_xyz(filepath):
    """Returns the coordinates of all atoms stored in a .xyz file. It assumes all atoms are of the same
    element and thus does not keep track of the chemical symbols in the input file.

    Parameter
    ----------
    filepath: `str`
        Path to the .xyz file whose coordinates we wish to obtain.

    Output
    ------
    coords: `ndarray`, shape=(N,3)
        Array of coordinates stored in the input file.
    """
    with open(filepath) as fo:
        natoms = int(fo.readline().rstrip().lstrip().lstrip('#'))
        fo.readline() #skip 2nd line
        lines = fo.readlines()
    coords = np.array([list(map(float,line.lstrip().rstrip().split()[1:])) for line in lines])

    return coords


def write_xyz(coords,symbols,filename):
    """Writes the coordinates stored in NumPy array to a .xyz file.

    Parameter
    ----------
    coords: `numpy.ndarray`, shape=(N,M) with M > 1, dtype=float
        Cartesian coordinates that we wish to write to a file
    symbols: `numpy.ndarray`, shape=(N,), dtype=str
        List of strings that label the entities whose coordinates are stored in `coords`.
        For atoms, these are usually chemical symbols (e.g. 'C' for carbon).
    filepath: `str`
        Path to the .xyz file whose coordinates we wish to obtain.

    Output
    ------
    coords: `ndarray`, shape=(N,3)
        Array of coordinates stored in the input file.
    """
    symbol_field_size = len(sorted(symbols,key=len)[-1]) #get maximum size of symbols
    if coords.shape[1] == 2:
        new_coords = np.zeros((coords.shape[0],3),dtype=float)
        new_coords[:,:2] = coords
        coords = new_coords
    with open(filename,'w') as fo:
        fo.write(' {:d}\n\n'.format(coords.shape[0]))
        for s, r in zip(symbols,coords):
            x,y,z = r
            fo.write('{0:{width}}\t{1:2.8f}\t{2:2.8f}\t{3:2.8f}\n'.format(s,x,y,z,width=symbol_field_size))

def write_xsf(atoms, supercell, symbols=None, force_array=None, filename="carbon.xsf"):
    f=open(filename, "w")

    if symbols == None:
        symbols = ['C']*atoms.shape[0]

    if supercell==None:
        f.write("ATOMS\n")
        for s, atom in zip(symbols,atoms):
            f.write("%s %f %f %f\n" % (s, atom[1][0], atom[1][1], atom[1][2]))

    else:
        f.write("CRYSTAL\n")
        f.write("PRIMVEC\n")
        f.write("%f %f %f\n" % (supercell[0], 0.0, 0.0))
        f.write("%f %f %f\n" % (0.0, supercell[1], 0.0))
        f.write("%f %f %f\n" % (0.0, 0.0, 20.0))
        f.write("PRIMCOORD\n")
        f.write("%d 1\n" % (len(atoms)))
        
        if np.all(force_array == None) or np.any(force_array.shape!=atoms.shape):
            if np.any(force_array != None) and force_array.shape != atoms.shape:
                print('ERROR: write_xsf: atoms and forces arrays need to have the same shape.\nWriting only atoms.')
            for s, atom in zip(symbols, atoms):
                f.write("%s %f %f %f\n" % (s, atom[0], atom[1], atom[2]))
        else:
            for atom,force in zip(atoms,force_array):
                f.write('%s %f %f %f %f %f %f\n'%(s, *atom,*force))

    f.close()

def read_xsf(filename,read_forces=True):
    f=open(filename)
    for i in range(2): f.readline()

    supercell = []
    supercell.append( float( f.readline().strip().split()[0] ) )
    supercell.append( float( f.readline().strip().split()[1] ) )

    for i in range(2): f.readline()
    
    na = int( f.readline().strip().split()[0] )
    atoms = np.zeros((na,3),dtype=float)
    forces = np.zeros((na,3),dtype=float)
    forces_in_file = False
    for k in range(na):
        split_line = f.readline().strip().split()
        x,y,z = split_line[1:4]
        atoms[k,:] = np.array([x,y,z])
        if len(split_line) > 4:
            forces_in_file = True
            fx,fy,fz = split_line[4:]
            forces[k] = np.array([fx,fy,fz])
    f.close()
    if forces_in_file and read_forces:
        return atoms, forces, supercell
    else:
        return atoms, supercell

def write_LAMMPS_data(atoms, supercell, filename="carbon.data",minimum_coords=None):
    if np.all(minimum_coords == None):
        minimum_coords = np.zeros(3,dtype=float)
    f=open(filename,"w")
    f.write("carbon\n\n")
    f.write("%d atoms\n\n" % (len(atoms)))
    f.write("1 atom types\n\n")
    f.write("%f %f xlo xhi\n" % (minimum_coords[0], supercell[0]))
    f.write("%f %f ylo yhi\n" % (minimum_coords[1], supercell[1]))
    f.write("%f 20.0 zlo zhi\n\n" % (minimum_coords[2]))

    f.write("Masses\n\n")
    f.write("1 12.0\n\n")

    f.write("Atoms\n\n")
    for i in range(len(atoms)):
        f.write("%d 1 %f %f %f\n" % (i+1,
                                     atoms[i][0],
                                     atoms[i][1],
                                     atoms[i][2] )   #+(np.random.rand()-0.5)*0.5)
        )  # atom_ID atom_type x y z
    f.close()
    
def LAMMPS2XSF(dump):
    from dump2xsf import dump2xsf
    dump2xsf(dump)
