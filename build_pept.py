#!/usr/bin/python

import argparse as ag

from Bio.PDB.Polypeptide import one_to_three as ott
from Bio.Seq import Seq
import Bio.Alphabet
import Bio.Alphabet.IUPAC
import os


class PepBuilder(object):

    def __init__(self):

        self.pymol = self.init_pymol()

    def gen_pept(
            self,
            seq,
            out=None,
            caps=False,
            kmers=None,
            ncap='ace',
            ccap='nme',
            charged=True,
            rpath=None,
            *args,
            **kwargs):

        self.pymol.cmd.set('pdb_use_ter_records', 0)

        L = len(seq)

        if kmers:
            l = kmers
        else:
            l = len(seq)

        for i in range(L - l + 1):
            tseq = seq[i:i + l]
            print(tseq)
            if caps:
                self.pymol.cmd.editor.attach_amino_acid('pk1', ncap)

            for a in tseq:
                self.pymol.cmd.editor.attach_amino_acid('pk1', ott(a).lower())

            if caps:
                self.pymol.cmd.editor.attach_amino_acid('pk1', ccap)

            if not caps:
                self.pymol.cmd.run(rpath)
                self.pymol.cmd.do("renumber all, 1")

                rn_ = {'rn': []}
                self.pymol.cmd.iterate(
                    "i. 1 and n. N",
                    "rn.append(resn)",
                    space=rn_)
                rn = rn_['rn'][0]

                hs = 2
                if rn == 'PRO':
                    hs = 1

                self.pymol.cmd.select('pk1', 'i. 1 and name N')
                self.pymol.cmd.attach('H', 3, 1)
                self.pymol.cmd.alter('n. H01', 'elem="H"')
                self.pymol.cmd.alter('n. H01', 'name="H%d"' % hs)

                if charged:
                    self.pymol.cmd.attach('H', 3, 1)
                    self.pymol.cmd.alter('n. H01', 'elem="H"')
                    self.pymol.cmd.alter('n. H01', 'name="H%d"' % (hs + 1))

                self.pymol.cmd.select('pk1', 'i. %d and name C' % l)
                self.pymol.cmd.attach('O', 1, 1)
                self.pymol.cmd.alter('name O01', 'elem="O"')
                self.pymol.cmd.alter('name O01', 'name="OXT"')

                if charged:
                    pass
                else:
                    self.pymol.cmd.select('pk1', 'i. %d and name OXT' % l)
                    self.pymol.cmd.attach('H', 1, 1)
                    self.pymol.cmd.alter('n. H01', 'name="HO"')

            if not out:
                out = '%s.pdb' % seq

            if kmers:
                oname = '%s_%s.pdb' % (out[:-4], tseq.upper())
            else:
                oname = out
            self.pymol.cmd.save(oname, 'all')
            self.pymol.cmd.delete('all')

    def init_pymol(self):
        os.environ['CHEMPY_DATA'] = '/usr/share/pymol/data/chempy'
        import __main__
        __main__.pymol_argv = ['pymol', '-qc']
        import pymol
        pymol.finish_launching()
        return pymol


def check_pept(seq):
    tseq = Seq(seq, Bio.Alphabet.IUPAC.protein)
    if Bio.Alphabet._verify_alphabet(tseq):
        return seq
    else:
        msg = "%s is not a valid peptide sequence" % tseq
        raise ag.ArgumentTypeError(msg)


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

    parser.add_argument('-n', '--neutral',
                        dest='charged',
                        action='store_false',
                        help='Make terminal residues neutral')

    parser.add_argument('-k', '--kmers',
                        type=int,
                        dest='kmers',
                        metavar='KMERS',
                        # default=4,
                        help='Generate all kmers of length K from SEQUENCE')

    parser.add_argument('-e', '--c-cap',
                        type=str,
                        dest='ccap',
                        metavar='ENDCAP',
                        default='nme',
                        choices=['nme', 'amc'],
                        help='Cap to be placed on the C term')

    parser.add_argument('-b', '--n-cap',
                        type=str,
                        dest='ncap',
                        metavar='ENDCAP',
                        default='ace',
                        choices=['ace'],
                        help='Cap to be placed on the N term')

    parser.add_argument('-r', '--renumber',
                        type=str,
                        dest='rpath',
                        metavar='RENUMBER_PATH',
                        required=True,
                        help='Path to renumber script')


    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Be verbose')

    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict


if __name__ == '__main__':
    args = get_args()
    # try:
    builder = PepBuilder()
    builder.gen_pept(**args)
    # except:
    #     print('ERROR %s' % args['seq'])
