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

step, padding, GminXYZ, N = None, None, None, None

if rank == 0:
    step = float(sys.argv[1])

    center, box = gu.get_box_from_vina(sys.argv[2])

    padding = max(gu.avdw.values())  # Largest VdW radii - 1.7A

    L = box + padding * (2 * np.int(sys.argv[4]))

    print 'CENTER', center
    print 'BOX', box, L

    GminXYZ = center - L / 2.0
    GminXYZ = gu.adjust_grid(GminXYZ, step, padding)

    N = np.ceil((L / step)).astype(np.int)

    print 'GMIN', GminXYZ, N

    with open('box_coords.txt', 'w') as f:
        f.write('BOX: ' + ' '.join(GminXYZ.astype(np.str)) + '\n')
        f.write('STEP: ' + sys.argv[1] + '\n')
        f.write('NSTEPS: ' + ';'.join(N.astype(np.str)) + '\n')


step = comm.bcast(step)
padding = comm.bcast(padding)
GminXYZ = comm.bcast(GminXYZ)
N = comm.bcast(N)

model_list = np.loadtxt('nucl', dtype='S128')

M = len(model_list)


NUCS = gu.three_let

iNUCS = dict(map(lambda x: (x[1], x[0],), enumerate(NUCS)))
fNUCS = np.zeros((len(NUCS), ), dtype=np.int)

# Init storage for matrices
# Get file name
SSfn = sys.argv[3]
tSSfn = 'tmp.' + sys.argv[3]
# Open matrix file in parallel mode
tSSf = h5py.File(tSSfn, 'w', driver='mpio', comm=comm)

for i in NUCS:
    # Ggrid[i] = np.ndarray((N, N, N), dtype=np.float32)
    tSSf.create_dataset(i, N, dtype=np.float, fillvalue=0.0)


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
    try:
        for R in M.iterResidues():

            if R.getResname() == "UNK":
                continue

            # Nucleobases

            tR = R.select(base)

            if tR:
                Rgrid, RminXYZ = gu.process_residue(tR, padding, step)

                adj = (RminXYZ - GminXYZ)
                adj = (adj / step).astype(np.int)
                x, y, z = adj

                tSSf[R.getResname()][
                    x: x + Rgrid.shape[0],
                    y: y + Rgrid.shape[1],
                    z: z + Rgrid.shape[2]
                ] += Rgrid

                fNUCS[iNUCS[R.getResname()]] += 1

            # Backbone

            tR = R.select(phs)

            if tR:

                Rgrid, RminXYZ = gu.process_residue(tR, padding, step)

                adj = (RminXYZ - GminXYZ)
                adj = (adj / step).astype(np.int)
                x, y, z = adj

                # assert tSSf[R.getResname()].shape[0] > (x + Rgrid.shape[0])
                # assert tSSf[R.getResname()].shape[1] > (x + Rgrid.shape[1])
                # assert tSSf[R.getResname()].shape[2] > (x + Rgrid.shape[2])

                tSSf['BCK'][
                    x: x + Rgrid.shape[0],
                    y: y + Rgrid.shape[1],
                    z: z + Rgrid.shape[2]
                ] += Rgrid

                fNUCS[iNUCS['BCK']] += 1

    except ValueError, e:
        print("echo %s >> errored" % m)
        print e, 'LINE: ', sys.exc_info()[-1].tb_lineno

    cm += NPROCS

comm.Barrier()

tSSf.close()

fNUCS = comm.reduce(fNUCS)

if rank == 0:
    print 'fNUCS after:'
    for i in iNUCS.keys():
        print i, fNUCS[iNUCS[i]]

if rank == 0:

    tSSf = h5py.File(tSSfn, 'r')
    SSf = h5py.File(SSfn, 'w')

    for i in NUCS:

        mult = max(1.0, fNUCS[iNUCS[i]])
        ttSSf = tSSf[i][:] / mult
        #ttSSf = tSSf[i][:]
        # nmax = max(1.0, np.max(ttSSf))
        nmax = np.max(ttSSf)

        if np.isnan(nmax) or (nmax == 0):
            nmax = 1.0

        ttSSf /= nmax
        ttSSf *= 100.0

        tG = np.ceil(ttSSf)
        SSf.create_dataset(i, data=tG.astype(np.int8))

    Gstep = np.array([step, step, step], dtype=np.float)
    SSf.create_dataset('step', data=Gstep)
    SSf.create_dataset('origin', data=GminXYZ)

    SSf.close()
    os.remove(tSSfn)

if debug is True:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    ps.dump_stats('stats.gg# print s.getvalue()')
