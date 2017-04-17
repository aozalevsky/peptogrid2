from chempy.brick import Brick
from pymol import cmd
import h5py


def load_grid(fname, mapname, ramp='cool'):
    F = h5py.File(fname, 'r')
    data = F['grid'][()]
    grid = F['step'][()]
    origin = F['origin'][()]
    b = Brick.from_numpy(data, grid, origin)
    cmd.load_brick(b, mapname)

    volname = mapname + '_volume'

    cmd.volume(volname, mapname)

    rampname = 'ramp_' + ramp

    cmd.volume_ramp_new(rampname, [
                        0.00, 0.00, 0.00, 1.00, 0.00,
                        0.10, 0.06, 0.93, 1.00, 0.20,
                        0.50, 0.49, 0.51, 1.00, 0.30,
                        0.80, 0.80, 0.18, 1.00, 0.90,
                        1.00, 0.98, 0.01, 1.00, 1.00,
                        ])

    cmd.volume_color(volname, rampname)

# The extend command makes this runnable as a command, from PyMOL.
cmd.extend("load_grid_generic", load_grid)
