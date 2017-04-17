import sys

# H5PY for storage
import h5py

from mpi4py import MPI
import numpy as np

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


# Init storage for matrices
# Get file name
tSSfn = sys.argv[1]
SSfn = sys.argv[2]
# Open matrix file in parallel mode

tSSf = h5py.File(tSSfn, 'r')
GminXYZ = tSSf['origin'][:]
step = tSSf['step'][0]

SSf = h5py.File(SSfn, 'w')

NUCS = set(tSSf.keys())
protected = set(['origin', 'step'])
NUCS -= protected

SUBmin = list()
SUBmax = list()

if len(sys.argv) == 4:
    center, box = gu.get_box_from_vina(sys.argv[3])

    L = box
    padding = 0
    tGminXYZ = center - L / 2.0
    N = np.ceil((L / step)).astype(np.int)

    assert (tGminXYZ < GminXYZ).all

    submin = np.ceil((tGminXYZ - GminXYZ) / step).astype(np.int)
    submax = submin + N

else:

    for i in NUCS:
        try:
            submin, submax = gu.submatrix(tSSf[i])

            SUBmin.append(submin)
            SUBmax.append(submax)
        except ValueError:
            pass

    submin = np.min(np.array(SUBmin), axis=0).astype(np.int)
    submax = np.max(np.array(SUBmax), axis=0).astype(np.int)


GminXYZ = GminXYZ + submin * step


for i in NUCS:
    tG = tSSf[i][
        submin[0]:submax[0],
        submin[1]:submax[1],
        submin[2]:submax[2],
    ]
    SSf.create_dataset(i, data=tG.astype(np.int8))

Gstep = np.array([step, step, step], dtype=np.float)

SSf.create_dataset('step', data=Gstep)
SSf.create_dataset('origin', data=GminXYZ)

SSf.close()

if debug is True:
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    ps.dump_stats('stats.gg# print s.getvalue()')
