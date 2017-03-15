from chempy.brick import Brick
from pymol import cmd
import h5py


def load_grid_nuc(fname, prefix='nuc', ramp='nuc'):

    cmd.volume_ramp_new('ramp_DA', [
        0.00, 1.00, 1.00, 1.00, 0.00,
        25.00, 1.00, 0.77, 0.77, 0.02,
        50.00, 1.00, 0.54, 0.54, 0.06,
        75.00, 1.00, 0.38, 0.38, 0.18,
        90.00, 0.93, 0.00, 0.00, 0.60,
        100.00, 0.70, 0.00, 0.00, 1.00,
    ])

    cmd.volume_ramp_new('ramp_DG', [
        0.00, 1.00, 1.00, 1.00, 0.00,
        25.00, 0.77, 0.77, 1.00, 0.02,
        50.00, 0.54, 0.54, 1.00, 0.06,
        75.00, 0.38, 0.38, 1.00, 0.18,
        90.00, 0.00, 0.00, 0.93, 0.60,
        100.00, 0.00, 0.00, 0.70, 1.00,
    ])

    cmd.volume_ramp_new('ramp_DC', [
        0.00, 1.00, 1.00, 1.00, 0.00,
        25.00, 0.64, 1.00, 0.61, 0.02,
        50.00, 0.48, 1.00, 0.47, 0.06,
        75.00, 0.26, 1.00, 0.28, 0.18,
        90.00, 0.04, 0.85, 0.00, 0.44,
        100.00, 0.00, 0.52, 0.06, 1.00,
    ])

    cmd.volume_ramp_new('ramp_DT', [
        0.00, 1.00, 1.00, 1.00, 0.00,
        25.00, 1.00, 0.99, 0.55, 0.02,
        50.00, 1.00, 0.98, 0.46, 0.06,
        75.00, 1.00, 0.99, 0.37, 0.19,
        90.00, 0.92, 0.92, 0.00, 0.39,
        100.00, 0.76, 0.73, 0.00, 1.00,
    ])

    cmd.volume_ramp_new('ramp_P', [
        0.00, 1.00, 1.00, 1.00, 0.00,
        25.00, 1.00, 0.95, 0.93, 0.02,
        50.00, 1.00, 0.85, 0.69, 0.06,
        75.00, 1.00, 0.78, 0.55, 0.19,
        90.00, 1.00, 0.66, 0.30, 0.39,
        100.00, 0.97, 0.50, 0.00, 1.00,
    ])

    cmd.volume_ramp_new('ramp_jet', [
        0.00, 0.00, 0.00, 1.00, 0.00,
        25.00, 0.06, 0.93, 1.00, 0.02,
        50.00, 0.49, 0.51, 1.00, 0.07,
        75.00, 1.00, 1.00, 0.00, 0.23,
        90.00, 0.80, 0.18, 1.00, 0.48,
        100.00, 0.98, 0.01, 1.00, 1.00,
    ])

    F = h5py.File(fname, 'r')
    grid = F['step'][()]
    origin = F['origin'][()]

    for i in ['DG', 'DA', 'DT', 'DC', 'P']:
        data = F[i][()]
        b = Brick.from_numpy(data, grid, origin)
        bname = prefix + '_' + i
        cmd.load_brick(b, bname)
        volname = bname + '_volume'

        cmd.volume(volname, bname)

        if ramp == 'nuc':
            cmd.volume_color(volname, 'ramp_' + i)
        else:
            cmd.volume_color(volname, 'ramp_jet')

# The extend command makes this runnable as a command, from PyMOL.
cmd.extend("load_grid_nuc", load_grid_nuc)
