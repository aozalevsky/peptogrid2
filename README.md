grid_atom_* - scripts to create hdf5 grids
load_grid* - pymol scripts to load grids from hdf5 files into pymol

Typical grid preparation:
# Dump grid to file

grfile = 'grid8.hdf5'
F = h5py.File(grfile, 'w')
# Main grid data
F.create_dataset('grid', data=Ggrid.astype(np.int8))
# array with steps at every axis. At the moment only uniform steps are available
# [0.1, 0.1, 0.1]
F.create_dataset('step', data=Gsteps) 
# array with coordinates of lower left angle of grid
# [0.0, 0.0, 0.0]
F.create_dataset('origin', data=GminXYZ)
F.close()
