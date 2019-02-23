#!/usr/bin/env python2
#
# This file is part of the AffBio package for clustering of
# biomolecular structures.
#
# Copyright (c) 2015-2016, by Arthur Zalevsky <aozalevsky@fbb.msu.ru>
#
# AffBio is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# AffBio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public
# License along with AffBio; if not, see
# http://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#

# H5PY for storage
# import h5py

import numpy as np
import bottleneck as bn
import time
import argparse as ag
import sys
import h5py
sys.path.append('/home/golovin/_scratch/progs/grid_scripts/')
sys.path.append('/home/domain/silwer/work/grid_scripts/')
import extract_result as er
import dock_pept as dp
import util as gu


class PeptMPIWorker(object):

    database = None
    llist = None
    plist = None
    mpi = None
    args = None
    tb = None
    te = None

    def __init__(
        self,
        database=None,
        plist=None,
        verbose=False,
        receptor=None,
        *args,
        **kwargs
    ):

        self.mpi = gu.init_mpi()

        plist_ = None

        self.database = dp.PepDatabase(database).database

        if self.mpi.rank == 0:

            if plist:
                plist_ = gu.read_plist(plist)
            else:
                plist_ = self.database.keys()

            for i in plist_:
                gu.check_pept(i)

        self.plist = self.mpi.comm.bcast(plist_)

        self.eplist = dict([(t[1], t[0]) for t in enumerate(self.plist)])

        N = len(self.plist)
        tb, te = gu.task(N, self.mpi)
        te = min(te, N)
        self.tb = tb
        self.te = te
        self.aplist = self.plist[tb:te]

        if len(self.aplist) > 0:
            print(
                'RANK: %d FIRST: %d LAST: %d LIST_FIRST: %s LIST_LAST: %s' % (
                    self.mpi.rank, tb, te, self.aplist[0], self.aplist[-1]))
        else:
            print(
                'RANK: %d NOTHING TO DO' % (
                    self.mpi.rank))

    def process(self):
        pass


class GridCalc(PeptMPIWorker):

    step = None
    padding = None
    box_config = None
    GminXYZ = None
    N = None

    atypes = None
    rtypes = None

    def __init__(self, args):
        super(GridCalc, self).__init__(**args)
        self.args = args

        self.step = args['step']

        self.padding = max(gu.avdw.values()) * 2 * \
            args['pad']  # Largest VdW radii - 1.7A

        if self.mpi.rank == 0:
            box_config = dp.ConfigLedock(args['config'])

            GminXYZ = box_config.get_min_xyz()
            self.GminXYZ = gu.adjust_grid(GminXYZ, self.step, self.padding)

            box = box_config.get_box_size()
            L = box + self.padding

            self.N = np.ceil((L / self.step)).astype(np.int)

            print('GMIN', self.GminXYZ, self.N)
            print('BOX', L)

            with open('box_coords.txt', 'w') as f:
                f.write('BOX: ' + ' '.join(self.GminXYZ.astype(np.str)) + '\n')
                f.write('STEP: %.2f\n' % self.step)
                f.write('NSTEPS: ' + ';'.join(self.N.astype(np.str)) + '\n')

            atypes_, rtypes = gu.choose_artypes(args['atypes'])

            atypes_ = set(atypes_)

            try:
                atypes_.remove('H')  # we do not need hydrogens!
            except KeyError:
                pass
            self.atypes = tuple(atypes_)

            self.rtypes = rtypes

        self.atypes = self.mpi.comm.bcast(self.atypes)
        self.rtypes = self.mpi.comm.bcast(self.rtypes)
        self.box_config = self.mpi.comm.bcast(self.box_config)
        self.GminXYZ = self.mpi.comm.bcast(self.GminXYZ)
        self.N = self.mpi.comm.bcast(self.N)

    def process(self, args=None):
        if not args:
            args = self.args

        NUCS = self.atypes
        iNUCS = dict(map(lambda x: (x[1], x[0],), enumerate(NUCS)))

        lnucs = len(NUCS)
        fNUCS = np.zeros((lnucs, ), dtype=np.float)

        # Init storage for matrices
        # Get file name

        tSf = dict()
        for i in NUCS:
            tSf[i] = np.zeros(self.N, dtype=np.float32)

        args['mpi'] = self.mpi
        extractor = er.PepExtractor(**args)
        lM = len(self.aplist)

        if self.mpi.rank == 0:
            m = self.aplist[0]
            S = extractor.extract_result(m)
            lS = S.numCoordsets()

        self.mpi.comm.Barrier()

        t0 = time.time()

        for cm in range(lM):
            m = self.aplist[cm]

            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            print('STEP: %d PERCENT: %.2f%% TIME: %s' % (
                cm, float(cm) / lM * 100, dt))

            try:
                S = extractor.extract_result(m)
            except:
                print('ERROR: BAD PEPTIDE: %s' % m)
                continue

            lS = S.numCoordsets()

            for S_ in range(lS):

                S.setACSIndex(S_)

                for a in S.iterAtoms():

                    # skip hydrogens

                    if a.getElement() == 'H':
                        continue

                    try:
                        atype = self.rtypes[(a.getResname(), a.getName())]
                    except:
                        print('ATYPE not found', a.getResname(), a.getName())

                    if atype not in self.atypes:
                        continue

                    Agrid, AminXYZ = gu.process_atom(a, self.step)

                    adj = (AminXYZ - self.GminXYZ)
                    adj = (adj / self.step).astype(np.int)
                    x, y, z = adj

                    try:
                        tSf[atype][
                            x: x + Agrid.shape[0],
                            y: y + Agrid.shape[1],
                            z: z + Agrid.shape[2]
                        ] += Agrid

                        fNUCS[iNUCS[atype]] += 1

                    except:
                        print(m, a)

        nNUCS = np.zeros((lnucs, ), dtype=np.float)

        Sfn = args['Sfn']

        self.mpi.comm.Barrier()

        # Allocate results file

        if self.mpi.rank == 0:

            Sf = h5py.File(Sfn, 'w')

            for i in range(lnucs):
                Sf.create_dataset(NUCS[i], tSf[NUCS[i]].shape, dtype='f')

            Gstep = np.array(
                [self.step, self.step, self.step], dtype=np.float32)
            Sf.create_dataset('step', data=Gstep)
            Sf.create_dataset('origin', data=self.GminXYZ)
            Sf.create_dataset(
                'atypes', data=np.array([args['atypes'], ], dtype='S20'))

            Sf.close()

        self.mpi.comm.Barrier()
        # Open matrix file in parallel mode

        Sf = h5py.File(Sfn, 'r+', driver='mpio', comm=self.mpi.comm)
        Sf.atomic = True

        for i in range(lnucs):
            mult = fNUCS[i]

            if mult > 0.0:
                ttSf = tSf[NUCS[i]]

                tG = ttSf

                Sf[NUCS[i]][:] += tG[:]

        self.mpi.comm.Barrier()

        fNUCS_ = self.mpi.comm.allreduce(fNUCS)

        i = self.mpi.rank

        while i < lnucs:

            mult = fNUCS_[i]

            if mult > 0:
                ttSf = Sf[NUCS[i]][:]
                nNUCS[i] = np.max(ttSf)

            print(i, NUCS[i], nNUCS[i])

            i += self.mpi.NPROCS

        self.mpi.comm.Barrier()

        print nNUCS
        nNUCS_ = self.mpi.comm.allreduce(nNUCS)

        print 'nNUCS', nNUCS_
        print 'fNUCS', fNUCS_

        nmax = bn.nanmax(np.divide(nNUCS_, fNUCS_))

        if self.mpi.rank == 0:
            print('NMAX', nmax)

        self.mpi.comm.Barrier()

        i = self.mpi.rank

        while i < lnucs:

            mult = fNUCS_[i]

            ttSf = Sf[NUCS[i]][:]

            if mult > 0:

                med = np.median(ttSf)
                ttSf[ttSf < (med)] = 0

                ttSf /= float(mult)
                ttSf /= float(nmax)
                ttSf *= 100.0

            else:
                print('Array is empty for: ', NUCS[i])

            tG = ttSf
            Sf[NUCS[i]][:] = tG[:]

            i += self.mpi.NPROCS

        self.mpi.comm.Barrier()
        self.database.close()
        Sf.close()


def get_args():
    """Parse cli arguments"""
    parser = ag.ArgumentParser(
        description='Grid scripts')

    parser.add_argument('-g',
                        required=True,
                        dest='Sfn',
                        metavar='FILE.hdf5',
                        help='HDF5 file for all matrices')

    parser.add_argument('--debug',
                        action='store_true',
                        help='Perform profiling')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Be verbose')

    parser.add_argument('--atypes', '-t',
                        choices=[
                            'default',
                            'gromos',
                            'vina',
                            'tripos_neutral',
                            'tripos_charged'],
                        default='tripos_neutral',
                        type=str,)

    parser.add_argument('-s', '--step',
                        type=float,
                        dest='step',
                        metavar='STEP',
                        default=0.1,
                        help='Grid step in Angstroms')

    parser.add_argument('--padding',
                        type=int,
                        dest='pad',
                        metavar='PADDING',
                        default=1,
                        help='Padding around cell in Angstroms')

    parser.add_argument('-c', '--config',
                        type=str,
                        dest='config',
                        metavar='CONFIG',
                        help='Ledock config file')

    parser.add_argument('-d', '--database',
                        type=str,
                        required=True,
                        help='Database of peptides')

    parser.add_argument('-f', '--file',
                        dest='resfile',
                        type=str,
                        required=True,
                        help='Peptide results hdf5')

    parser.add_argument('-p', '--plist',
                        type=str,
                        dest='plist',
                        metavar='PLIST.TXT',
                        help='List of peptides')

    args = parser.parse_args()

    args_dict = vars(args)

    return args_dict


if __name__ == "__main__":
    args = get_args()
    worker = GridCalc(args)
    worker.process()
