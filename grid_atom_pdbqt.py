import sys


# H5PY for storage
import h5py

# MPI parallelism
from mpi4py import MPI
import os

from prody import parsePDB
import numpy as np
import time
import oddt

from collections import Counter

sys.path.append('/home/domain/silwer/work/')
import grid_scripts.util as gu


# Get MPI info
comm = MPI.COMM_WORLD
# NPROCS_LOCAL = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
# Get number of processes
NPROCS = comm.size
# Get rank
rank = comm.rank

Sfn = None
model_list = None
step, padding, GminXYZ, N = None, None, None, None
debug = False

if rank == 0:

    args = gu.get_args()
    step = args['step']

    center, box = gu.get_box_from_vina(args['config'])

    padding = max(gu.aradii.values()) * 2 * args['pad']  # Largest VdW radii - 1.7A

    L = box + padding

    print 'CENTER', center
    print 'BOX', box, L

    GminXYZ = center - L / 2.0
    GminXYZ = gu.adjust_grid(GminXYZ, step, padding)

    N = np.ceil((L / step)).astype(np.int)

    print 'GMIN', GminXYZ, N

    with open('box_coords.txt', 'w') as f:
        f.write('BOX: ' + ' '.join(GminXYZ.astype(np.str)) + '\n')
        f.write('STEP: %.2f\n' % step)
        f.write('NSTEPS: ' + ';'.join(N.astype(np.str)) + '\n')

    Sfn = args['Sfn']
    model_list = args['pdb_list']



    debug = args['debug']

if debug:
   import cProfile
   import pstats
   import StringIO
   pr = cProfile.Profile()
   pr.enable()

step = comm.bcast(step)
padding = comm.bcast(padding)
GminXYZ = comm.bcast(GminXYZ)
N = comm.bcast(N)
Sfn = comm.bcast(Sfn)
model_list = comm.bcast(model_list)

M = len(model_list)

atypes = gu.vina_types

NUCS = atypes
iNUCS = dict(map(lambda x: (x[1], x[0],), enumerate(NUCS)))

lnucs = len(NUCS)
fNUCS = np.zeros((lnucs, ), dtype=np.int)

if rank == 0:
    print iNUCS


# Init storage for matrices
# Get file name

# Open matrix file in parallel mode
Sf = h5py.File(Sfn, 'w', driver='mpio', comm=comm)
for i in NUCS:
    Sf.create_dataset(i, N, dtype=np.int8, fillvalue=0)

tSfn = 'tmp.' + Sfn
tSf = h5py.File(tSfn, 'w', driver='mpio', comm=comm)
for i in NUCS:
    tSf.create_dataset(i, N, dtype=np.float, fillvalue=0.0)



lm = len(model_list)

cm = rank

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

    M = oddt.ob.readfile('pdbqt', m)

    for S in M:

        for A in S.atom_dict:

            Agrid, AminXYZ = gu.process_atom_oddt(A, padding, step)

            adj = (AminXYZ - GminXYZ)
            adj = (adj / step).astype(np.int)
            x, y, z = adj

            atype = A[5]

            if atype not in atypes:
                continue

            tSf[atype][
                x: x + Agrid.shape[0],
                y: y + Agrid.shape[1],
                z: z + Agrid.shape[2]
            ] += Agrid

            fNUCS[iNUCS[atype]] += 1

            #except Exception as e:

             #   print("echo %s >> errored" % m)
             #   print e, 'LINE: ', sys.exc_info()[-1].tb_lineno
             #   continue

    cm += NPROCS

comm.Barrier()



if rank == 0:
    print 'fNUCS before', fNUCS

fNUCS = comm.allreduce(fNUCS)

if rank == 0:
    print NUCS
    print 'fNUCS after', fNUCS


nNUCS = np.zeros((lnucs, ), dtype=np.float)


ln = len(NUCS)

i = rank
while i < ln:

    mult = fNUCS[i]

    if mult > 0:
        print NUCS[i]
        nmax = np.max(tSf[NUCS[i]])
        nNUCS[i] = nmax / mult

    i += NPROCS

comm.Barrier()
if rank == 0:
    print 'BEGOFRE', nNUCS
nNUCS = comm.reduce(nNUCS)
nmax = np.max(nNUCS)
if rank == 0:
    print nNUCS, nmax

nmax = comm.bcast(nmax)

i = rank

while i < ln:

    mult = fNUCS[i]

    if mult > 0:
        ttSf = tSf[NUCS[i]][:]  #/ mult
        ttSf /= mult
        ttSf /= nmax
        ttSf *= 100.0

        tG = np.ceil(ttSf).astype(np.int8)

    else:
        tG = np.zeros(tSf[NUCS[i]].shape, dtype=np.int8)

    Sf[NUCS[i]][:] = tG[:]

    i += NPROCS


comm.Barrier()
tSf.close()
Sf.close()

if rank == 0:
    Sf = h5py.File(Sfn, 'r+')
    Gstep = np.array([step, step, step], dtype=np.float)
    Sf.create_dataset('step', data=Gstep)
    Sf.create_dataset('origin', data=GminXYZ)
    Sf.close()
    os.remove(tSfn)

if debug:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'time'
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats(sortby)
    ps.print_stats()
    print s.getvalue()
