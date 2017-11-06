import sys


# H5PY for storage
import h5py

# MPI parallelism
from mpi4py import MPI
import os

from prody import parsePDB
import numpy as np
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
# NPROCS_LOCAL = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
# Get number of processes
NPROCS = comm.size
# Get rank
rank = comm.rank


model_list = np.loadtxt(sys.argv[2], dtype='S128')

M = len(model_list)

# Init storage for matrices
# Get file name
SSfn = sys.argv[1]
tSSfn = 'tmp.' + SSfn
# Open matrix file in parallel mode
tSSf = h5py.File(tSSfn, 'w', driver='mpio', comm=comm)
score = tSSf.create_dataset('score', (M,), dtype=np.float)

# Open matrix file in parallel mode
SSf = h5py.File(SSfn, 'r', driver='mpio', comm=comm)

GminXYZ = SSf['origin'][:]
step = SSf['step'][0]
padding = np.max(gu.avdw.values())  # Largest VdW radii - 1.7A

NUCS = set(SSf.keys())
protected = set(['origin', 'step'])
NUCS -= protected

gNUCS = dict()

for i in NUCS:
    # Ggrid[i] = np.ndarray((N, N, N), dtype=np.float32)
    gNUCS[i] = SSf[i][()]


lm = len(model_list)

cm = rank

phosphate = ['N', 'CA', 'C', 'O']
phs = 'name ' + ' '.join(phosphate)

base = 'not ' + phs

t0 = time.time()
c = 0

while cm < lm:
    m = model_list[cm]

    if rank == 0 and c % 100 == 0:
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        print('RANK: %s STEP: %s TIME: %s' % (0, c, dt))
    c += 1

    M = parsePDB(m)
    mscore = 0.0

    try:
        for R in M.iterResidues():

            # Nucleobases

            tR = R.select(base)

            if tR:

                Rgrid, RminXYZ = gu.process_residue(tR, padding, step)

                adj = (RminXYZ - GminXYZ)
                adj = (adj / step).astype(np.int)
                x, y, z = adj

            # print Rgrid.shape

                mscore += np.sum(gNUCS[R.getResname()][
                    x: x + Rgrid.shape[0],
                    y: y + Rgrid.shape[1],
                    z: z + Rgrid.shape[2]
                ] * Rgrid) / len(tR)

            # Phosphates

            bck = False

            tR = R.select(phs)

            if tR and bck:

                Rgrid, RminXYZ = gu.process_residue(tR, padding, step)

                adj = (RminXYZ - GminXYZ)
                adj = (adj / step).astype(np.int)
                x, y, z = adj

                mscore += np.sum(gNUCS['BCK'][
                    x: x + Rgrid.shape[0],
                    y: y + Rgrid.shape[1],
                    z: z + Rgrid.shape[2]
                ] * Rgrid) / len(tR)

    except ValueError as e:
        print("echo %s >> errored" % m)
        print(e, 'LINE: ', sys.exc_info()[-1].tb_lineno)

    tSSf['score'][cm] = mscore

    cm += NPROCS

comm.Barrier()

if rank == 0:
    tscore = np.zeros((lm,), dtype=[('name', 'S128'), ('score', 'f8')])
    tscore['name'] = model_list[:]
    tscore['score'] = tSSf['score'][:]
    tscore.sort(order='score')
    tscore['score'] /= tscore['score'][-1]

    np.savetxt('score_table.csv', tscore[::-1], fmt="%s\t%.2f")

comm.Barrier()

tSSf.close()
SSf.close()


if rank == 0:
    os.remove(tSSfn)


if debug is True:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    ps.dump_stats('stats.gg# print s.getvalue()')
