
# coding: utf-8


# In[10]:

import ConfigParser
import numpy as np
# import gromacs
# import gromacs.formats
# import mdtraj
# import sklearn.preprocessing

# In[11]:


# FN = sys.argv[1]
# FRAME = int(sys.argv[2])
# TOP = sys.argv[3]

# In[12]:


natoms = ('OP1', 'OP2', 'O1P', 'O2P', 'N4', 'N6', 'O4',
          'O6', 'N7', 'N2', 'N3', 'O2', "O3'", "O5'", "O2'", "O4'",)

# In[15]:
# proth = gromacs.formats.XVG('prot_hbdist.xvg')
# rawdata = proth.array

# bins = rawdata[0] * 10.0

# mmscaler = sklearn.preprocessing.MinMaxScaler()

# dat = mmscaler.fit_transform(rawdata[1])


# start = bins[0]
# stop = bins[-1]
# step = 0.2

# padding = stop - start + step


def get_bounding(R, padding, step):
    minXYZ = np.amin(R, axis=0)
    maxXYZ = np.amax(R, axis=0)
    minXYZ -= (padding + step)
    maxXYZ += (padding + step)
    shape = np.ceil((maxXYZ - minXYZ) / step).astype(np.int)
    minXYZ = adjust_grid(minXYZ, step)
    return(minXYZ, shape)


def adjust_grid(c, step, padding=0):
    nc = c - padding
    nc /= step
    nc = np.floor(nc) * step
    return nc


def sphere2grid(c, r, step):
    nc = adjust_grid(c, step)
    adj = nc - c
    r2x = np.arange(-r - adj[0], r - adj[0] + step, step) ** 2
    r2y = np.arange(-r - adj[1], r - adj[1] + step, step) ** 2
    r2z = np.arange(-r - adj[2], r - adj[2] + step, step) ** 2
    dist2 = r2x[:, None, None] + r2y[:, None] + r2z
    nc = nc - dist2.shape[0] / 2 * step
    vol = (dist2 <= r ** 2).astype(np.int)
    vol *= -999
    return (nc, vol)


def hist2grid(c, dat, bins, step):
    r = bins[-1]
    nc = adjust_grid(c, step)
    adj = nc - c
    r2x = np.arange(-r - adj[0], r - adj[0] + step, step) ** 2
    r2y = np.arange(-r - adj[1], r - adj[1] + step, step) ** 2
    r2z = np.arange(-r - adj[2], r - adj[2] + step, step) ** 2

    dist2 = r2x[:, None, None] + r2y[:, None] + r2z
    grid = np.ndarray(dist2.shape, dtype=np.float)
    grid.fill(0.0)

    for i in range(len(bins) - 1):
        grid[(dist2 > bins[i] ** 2) & (dist2 <= bins[i + 1] ** 2)] = dat[i]

    nc = nc - dist2.shape[0] / 2 * step

    return (nc, grid)


#  http://www.weslack.com/question/1552900000001895550

def submatrix(arr):
    x, y, z = np.nonzero(arr)
    # Using the smallest and largest x and y indices of nonzero elements,
    # we can find the desired rectangular bounds.
    # And don't forget to add 1 to the top bound to avoid the fencepost
    # problem.
    return (
        np.array(
            [x.min(), y.min(), z.min()]
        ),
        np.array(
            [x.max() + 1, y.max() + 1, z.max() + 1]
        ),
        )


# In[18]:

# In[19]:

aradii = {
    'H': 0.53,
    'C': 0.67,
    'N': 0.56,
    'O': 0.48,
    'P': 0.98,
    'S': 0.88,
}

avdw = {
    'H': 1.2,
    'C': 1.7,
    'N': 1.55,
    'O': 1.52,
    'P': 1.8,
    'S': 1.8,
}


# http://stackoverflow.com/questions/2819696/parsing-properties-file-in-python/2819788#2819788

class FakeSecHead(object):

    def __init__(self, fp):
        self.fp = fp
        self.sechead = '[asection]\n'

    def readline(self):
        if self.sechead:
            try:
                return self.sechead
            finally:
                self.sechead = None
        else:
            return self.fp.readline()


def get_box_from_vina(f):
    cp = ConfigParser.SafeConfigParser()
    cp.readfp(FakeSecHead(open(f)))
    cfg = dict(cp.items('asection'))

    center = np.array([
                      cfg['center_x'],
                      cfg['center_y'],
                      cfg['center_z'],
                      ],
                      dtype=np.float)

    box = np.array([
                   cfg['size_x'],
                   cfg['size_y'],
                   cfg['size_z'],
                   ],
                   dtype=np.float)

    return (center, box)
