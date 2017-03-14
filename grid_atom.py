

import sys


# H5PY for storage
import h5py

# MPI parallelism
from mpi4py import MPI
import os


from prody import parsePDB
import numpy as np
import math
import time

sys.path.append('/home/domain/silwer/work/')
import grid_scripts.util as gu

debug = False

if debug is True:
    import cProfile
    import pstats
    import StringIO
    pr = cProfile.Profile()
    pr.enable()

# Get MPI info
comm = MPI.COMM_WORLD
NPROCS_LOCAL = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
# Get number of processes
NPROCS = comm.size
# Get rank
rank = comm.rank


step = float(sys.argv[1])

padding = max(gu.avdw.values())  # Largest VdW radii - 1.7A

center, box = gu.get_box_from_vina(sys.argv[2])


L = box + 2 * padding

GminXYZ = center - box / 2.0
GminXYZ = gu.adjust_grid(GminXYZ, step, padding)

N = int(math.ceil(L / step))

# print GminXYZ, N

with open('nucl', 'r') as f:
    model_list = map(str.strip, f.readlines())

with open('box_coords.txt', 'w') as f:
    f.write('BOX: ' + ' '.join(GminXYZ.astype(np.str)) + '\n')
    f.write('STEP: ' + sys.argv[1] + '\n')
    f.write('NSTEPS: ' + str(N) + '\n')

M = len(model_list)


NUCS = ['DA', 'DT', 'DG', 'DC']
NUCS_GRID_t = {}


# Init storage for matrices
# Get file name
SSfn = sys.argv[3]
# Open matrix file in parallel mode
SSf = h5py.File(SSfn, 'w', driver='mpio', comm=comm)


NUCS_GRID = {}
Ggrid = {}
dGgrid = {}
for i in NUCS:
    Ggrid[i] = np.ndarray((N, N, N), dtype=np.float32)
    dGgrid[i] = SSf.require_dataset(Ggrid.shape(), dtype=np.int8)

t0 = time.time()
c = 0


lm = len(model_list)

cm = rank

while cm < lm:
    m = model_list[cm]

    if rank == 0 and c % 100 == 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        print('RANK: %s STEP: %s TIME: %s' % (0, c, dt))
    c += 1

    M = parsePDB(m)
    try:
        for R in M.iterResidues():

            RminXYZ, Rshape = gu.get_bounding(R.getCoords(), padding, step)
            Rgrid = np.ndarray(Rshape, dtype=np.int)
            Rgrid.fill(0)

            for A in R.iterAtoms():

                AminXYZ, Agrid = gu.sphere2grid(
                    A.getCoords(), gu.avdw[A.getElement()], step)
                NA = Agrid.shape[0]

                adj = (AminXYZ - RminXYZ)
                adj = (adj / step).astype(np.int)
                x, y, z = adj
                Rgrid[x: x + NA, y: y + NA, z: z + NA] += Agrid

            np.clip(Rgrid, 0, 1, out=Rgrid)
            adj = (RminXYZ - GminXYZ)
            adj = (adj / step).astype(np.int)
            x, y, z = adj

            Ggrid[R.getResname()][
                x: x + Rshape[0],
                y: y + Rshape[1],
                z: z + Rshape[2]
                ] += Rgrid
    except ValueError:
        print("echo %s >> errored" % m)

Gstep = np.array([step, step, step], dtype=np.float)

for i in NUCS:
    Gmax  = np.max(Ggrid[i])
    submin, submax = gu.submatrix(Ggrid[i])

    SUBmin = comm.gather(submin)
    SUBmax = comm.gather(submax)


if rank == 0:

    submin = comm.bcast(np.min(SUBmin, axis=0))
    submax = comm.bcast(np.max(SUBmax, axis=0))




for i in NUCS:
    grfile = i + '_grid.hdf5'
#    Ggrid[i].dump(i + '_grid.dump')
    F = h5py.File(grfile, "w")
    Fg = F.create_dataset('grid', data=Ggrid[i])
    Sg = F.create_dataset('step', data=Gstep)
    Og = F.create_dataset('origin', data=GminXYZ)
    F.close()

# SSf.close()

if debug is True:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    ps.dump_stats('stats.gg')
    # print s.getvalue()
