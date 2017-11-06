#!/usr/bin/python

import argparse as ag

from Bio.PDB.Polypeptide import one_to_three as ott
from Bio.Seq import Seq
import Bio.Alphabet
import Bio.Alphabet.IUPAC
import os

os.environ['CHEMPY_DATA'] = '/usr/share/pymol/data/chempy'


def get_args():
    """Parse cli arguments"""

    parser = ag.ArgumentParser(
        description='Build custom peptide sequence using PyMOL API')

    parser.add_argument('-s', '--sequence',
                        required=True,
                        dest='seq',
                        metavar='SEQUENCE',
                        type=check_pept,
                        help='Peptide sequence')

    parser.add_argument('-o', '--output',
                        dest='out',
                        metavar='OUTPUT',
                        help='Name of output PDB file')

    parser.add_argument('-c', '--caps',
                        action='store_true',
                        help='Add ACE and NME caps at the ends of peptide')

    parser.add_argument('-k', '--kmers',
                        type=int,
                        dest='kmers',
                        metavar='KMERS',
                        # default=4,
                        help='Generate all kmers of length K from SEQUENCE')

    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Be verbose')

    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict


def check_pept(seq):
    tseq = Seq(seq, Bio.Alphabet.IUPAC.protein)
    if Bio.Alphabet._verify_alphabet(tseq):
        return seq
    else:
        msg = "%s is not a valid peptide sequence" % tseq
        raise ag.ArgumentTypeError(msg)


def init_pymol():
    import __main__
    __main__.pymol_argv = ['pymol', '-qc']
    import pymol
    pymol.finish_launching()
    return pymol


def gen_pept(
        seq,
        out=None,
        caps=False,
        kmers=None,
        *args,
        **kwargs):

    pymol = init_pymol()
    pymol.cmd.set('pdb_use_ter_records', 0)

    L = len(seq)

    if kmers:
        l = kmers
    else:
        l = len(seq)

    for i in range(L - l + 1):
        tseq = seq[i:i + l]
        print tseq
        if caps:
            pymol.cmd.editor.attach_amino_acid('pk1', 'ace')

        for a in tseq:
            pymol.cmd.editor.attach_amino_acid('pk1', ott(a).lower())

        if caps:
            pymol.cmd.editor.attach_amino_acid('pk1', 'nme')

        if not out:
            out = '%s.pdb' % seq

        if kmers:
            oname = '%s_%s.pdb' % (out[:-4], tseq.upper())
        else:
            oname = out
        pymol.cmd.save(oname, 'all')
        pymol.cmd.delete('all')

if __name__ == '__main__':
    args = get_args()
    try:
        gen_pept(**args)
    except:
        print('ERROR %s' % args['seq'])
