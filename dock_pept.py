#!/usr/bin/env python2

import argparse as ag
import re
import os
# import pickle
from sqlitedict import SqliteDict
import StringIO
import subprocess as sp
import util

import prody
from collections import OrderedDict as OD

# H5PY for storage
import numpy as np
import numpy.ma as ma
import h5py
# from h5py import h5s
import oddt

prody.confProDy(verbosity='error')


class ConfigLedock(object):

    csize = OD([
        ('Receptor', 1),
        ('RMSD', 1),
        ('Binding pocket', 3),
        ('Number of binding poses', 1),
        ('Ligands list', 1),
        ('END', 0)
    ])

    fname = None
    config = None

    def __init__(self, fname=None):
        if fname:
            self.parse_config(fname)
            self.is_valid()
            self.fname = fname
        else:
            pass

    def parse_config(self, fname):

        with open(fname, 'r') as f:
            raw = f.readlines()
        lines = map(str.strip, raw)

        ckeys = self.csize.keys()

        config = OD.fromkeys(self.csize.keys())

        i = 0

        N = len(lines)

        while i < N:
            l = lines[i]
            if l in ckeys:
                i_ = self.csize[l]
                config[l] = lines[i + 1:i + i_ + 1]

                if l == 'END':
                    i_ = 1

                i += i_ + 1
            else:
                i += 1

        self.is_valid(config)

        self.config = config

        self.box = np.array(
            map(lambda x: x.split(), self.config['Binding pocket']),
            dtype=np.float
            )

    def is_valid(self, config=None):
        if config is None:
            config = self.config
        keys = self.csize.keys()

        for k in keys:

            if k not in config:
                raise Exception('Missing "%s" in config' % k)

            if self.csize[k] != len(config[k]):
                raise Exception('Wrong parameters for "%s"' % k)

        return True

    def write(self, fname=None):
        if fname is None:
            fname = self.fname

        c_ = ''

        for i in self.config.items():
            k, v = i
            c_ += k
            c_ += '\n'
            c_ += '\n'.join(v)
            c_ += '\n'
            c_ += '\n'

        with open(fname, 'w') as f:
            f.write(c_)

    def get_vina_style_center_box(self):
        pass

    def get_min_xyz(self):
        return self.box[:, 0]

    def get_box(self):
        return self.box

    def set_box(self, box):
        if type(box) == np.ndarray and box.shape == (3, 2):
            pass
        else:
            raise Exception("Wrong box")

        self.box = box
        self._update_box()

    def _update_box(self):

        self.config['Binding pocket'] = '\n'.join(
            map(lambda x: str(x[0]) + ' ' + str(x[1]), self.box)
            )

    def get_box_size(self):

        box_size = self.box[:, 1] - self.box[:, 0]

        return box_size

    def set_box_size(self, size):
        if type(size) == np.ndarray and size.shape == (3,):
            size_ = size

        elif type(size) == int and size > 0:
            size_ = np.ndarray((3,), dtype=np.float)
            size_.fill(size)

        else:
            raise Exception("Wrong box size")

        box_ = np.stack(
            (self.box[:, 0], self.box[:, 0] + size_),
            axis=-1
            )
        self.set_box(box_)


class PepDatabase(object):

    fname = None
    database = None

    def __init__(self, fname):
        if fname:
            try:
                # with open(fname, 'rb') as f:
                #     db_ = pickle.load(f)
                db_ = SqliteDict(fname)

            except:
                raise Exception('Something wrong with peptide databse')

            self.fname = fname
            self.database = db_
        else:
            raise Exception('Missing peptide database')
        pass


class PepDocker(object):

    docker = None
    database = None
    config = None
    checkpoint = None
    overwrite = False
    cleanup = True
    llist = None
    plist = None
    llist = None
    mpi = None

    def __init__(
        self,
        docker=None,
        database=None,
        config=None,
        checkpoint=None,
        plist=None,
        verbose=False,
        receptor=None,
        out=None,
        overwrite=False,
        cleanup=True,
    ):

        self.cleanup = cleanup
        self.mpi = util.init_mpi()

        if docker:
            self.docker = docker
        else:
            raise Exception('Missing Docker software!')

        self.database = PepDatabase(database).database

        if config:
            self.config = ConfigLedock(config)
        else:
            raise Exception('Missing Docker config!')

        if receptor:
            self.set_receptor(receptor)
        else:
            raise Exception('Missing receptor file')

        plist_ = None
        if self.mpi.rank == 0:
            if plist:
                plist_ = util.read_plist(plist)
            else:
                plist_ = self.database.keys()

            for i in plist_:
                util.check_pept(i)

        self.plist = self.mpi.comm.bcast(plist_)

        self.overwrite = overwrite

        if not out:
            out = 'out.hdf5'

        self.out = self.allocate_resfile(out, overwrite)
        self.out.flush()
        self.outfn = out

        # if not checkpoint:
        #    checkpoint = 'checkpoint.cpt'

        # self.checkpointfn = checkpoint

        # if not os.path.isfile(checkpoint):
        #    with open(checkpoint, 'a') as f:
        #        pass

        # self.from_checkpoint = self.read_plist(checkpoint)
        # self.checkpoint = open(checkpoint, 'a')

        self.checkpoint = self.out['checkpoint']

        self.eplist = dict([(t[1], t[0]) for t in enumerate(self.plist)])

        aplist = ma.array(self.plist, mask=self.checkpoint).compressed()

        N = len(aplist)
        tb, te = util.task(N, self.mpi)
        te = min(te, N)
        self.aplist = aplist[tb:te]

        if len(self.aplist) > 0:
            print(
                'RANK: %d FIRST: %d LAST: %d LIST_FIRST: %s LIST_LAST: %s' % (
                    self.mpi.rank, tb, te, self.aplist[0], self.aplist[-1]))
        else:
            print(
                'RANK: %d NOTHING TO DO' % (
                    self.mpi.rank))

    def run_once(self, seq):

        self.prepare_files(seq)
        self.run_docker()
        r = self.read_result(seq)
        self.write_result(seq, r)

        return r

    def allocate_resfile(self, fname, overwrite=False):

        def open_append(fname, comm):
            if comm:
                out = h5py.File(
                    fname, 'r+', driver='mpio', comm=comm)
                out.atomic = True
            else:
                out = h5py.File(
                    fname, 'r+')
            return out

        def open_new(fname, comm):
            if comm:
                out = h5py.File(
                    fname, 'w', driver='mpio', comm=comm)
                out.atomic = True
            else:
                out = h5py.File(fname, 'w')

            return out

        if not overwrite:
            try:
                out = open_append(fname, self.mpi.comm)
            except IOError:
                out = open_new(fname, self.mpi.comm)
        else:
            out = open_new(fname, self.mpi.comm)

        if 'plist' not in out:
            out.create_dataset('plist', data=self.plist)

        if 'checkpoint' not in out:
            out.create_dataset(
                'checkpoint',
                data=np.zeros(len(self.plist)),
                dtype=np.int8)

        for s in self.plist:
            if s not in out:
                G = out.create_group(s)
                for i in range(
                    int(
                        self.config.config['Number of binding poses'][0])):
                    dummyc = self.database[s].getCoords().shape
                    G.create_dataset('%d' % i, dummyc, dtype=np.float)

        return out

    def read_result(self, seq):
        fname = "%s.dok" % seq
        with open(fname, 'r') as f:
            raw = f.readlines()
        res = self.split_result(raw)
        pres = list()
        for r in res:
            e_ = self.get_energy(r)
            r_ = prody.parsePDBStream(StringIO.StringIO(r))
            pres.append((r_, e_))

        if self.cleanup:
            os.remove(fname)
            os.remove("%s.mol2" % seq)
            os.remove(self.llist_fname)
            os.remove(self.config.fname)
        return pres

    def get_energy(self, r):
        gg = re.findall(r'Score:\s+([-+]?\d+\.\d*)', r)
        if gg:
            e_ = float(gg[0])

        try:
            return e_
        except:
            raise Exception("Can't extract score value")

    def write_result(self, seq, allres):
        rG = self.out.require_group(seq)
        for i in range(len(allres)):
            pres, e = allres[i]
            r_ = rG["%d" % i]
            r_[:] = pres.getCoords()
            r_.attrs['score'] = e

    def split_result(self, lines):
        r = list()
        N = len(lines)
        i = 0
        s_ = None
        e_ = None
        while i < N:

            if re.match('REMARK', lines[i]) and s_ is None:
                s_ = i
            if re.match('END', lines[i]) and s_ is not None:
                e_ = i
                r.append(''.join(lines[s_: e_ + 1]))

                s_ = None
                e_ = None
            i += 1

        return r

    def run_docker(self):
        call = list()
        call.append(self.docker)
        call.append(self.config.fname)
        sp.check_call(call)
        # os.system(' '.join(call))
        # self.mpi.comm.Spawn(
        #    self.docker,
        #    args=[self.config_fname])

    def set_receptor(self, r):
        self.receptor = r
        self.config.config['Receptor'] = [r]

    def prepare_files(self, seq):

        llist = 'ligand_list_%d.txt' % self.mpi.rank
        self.config.config['Ligands list'] = [llist]
        self.llist_fname = llist

        self.write_config()

        prody.writePDB(seq + '.pdb', self.database[seq])
        mol = oddt.toolkit.readfile('pdb', seq + '.pdb').next()
        os.remove(seq + '.pdb')

        mol.write('mol2', seq + '.mol2', overwrite=True)

        with open(llist, 'w') as f:
            f.write('%s.mol2\n' % seq)

    def write_config(self):
        fname = '%d_config' % self.mpi.rank
        self.config.fname = fname
        self.config.write()

    def dock(self):
        print('Starting docking!')

        N = len(self.aplist)

        for i in range(N):
            s_ = self.aplist[i]
            self.run_once(s_)

            self.checkpoint[self.eplist[s_]] = 1
            print('Step %d of %d' % (i + 1, N))

        self.clean_files()

    def clean_files(self):
        self.out.close()


def get_args():
    """Parse cli arguments"""

    parser = ag.ArgumentParser(
        description='Dock peptides')

    parser.add_argument('-r', '--receptor',
                        required=True,
                        dest='receptor',
                        metavar='RECEPTOR.PDB',
                        type=str,
                        help='Preprocessed target structure')

    parser.add_argument('-o', '--output',
                        dest='out',
                        metavar='OUTPUT',
                        help='Name of output file')

    parser.add_argument('-d', '--database',
                        type=str,
                        required=True,
                        help='Database of peptides')

    parser.add_argument('-c', '--config',
                        type=str,
                        dest='config',
                        metavar='CONFIG',
                        required=True,
                        help='Ledock config')

    parser.add_argument('-l', '--ledock',
                        type=str,
                        dest='docker',
                        required=True,
                        metavar='LEDOCK',
                        help='Path to ledock')

    parser.add_argument('-p', '--plist',
                        type=str,
                        dest='plist',
                        metavar='PLIST.TXT',
                        help='List of peptides')

    parser.add_argument('-t', '--checkpoint',
                        type=str,
                        dest='checkpoint',
                        metavar='CHECKPOINT.TXT',
                        help='List of peptides')

    parser.add_argument('--no-cleanup',
                        action='store_false',
                        dest='cleanup',
                        help='overwrite previos results')

    parser.add_argument('--overwrite',
                        action='store_true',
                        dest='overwrite',
                        help='overwrite previos results')

    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Be verbose')

    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict


if __name__ == '__main__':
    args = get_args()
    docker = PepDocker(**args)
    docker.dock()
