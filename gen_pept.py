#!/usr/bin/python
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

# General modules
import argparse as ag
import PeptideBuilder
import Bio.PDB


def get_args():
    """Parse cli arguments"""

    parser = ag.ArgumentParser(
        description='Parallel ffitiny propagation for biomolecules')

    parser.add_argument('-l', '--list',
                        dest='list',
                        metavar='FILE',
                        help='List of peptides')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='Be verbose')

    parser.add_argument('-p', '--peps',
                        nargs='*',
                        type=str,
                        dest='peps',
                        metavar='PEPSEQ',
                        help='PDB files')

    parser.add_argument('-k', '--kmers',
                        action='store_true',
                        dest='kmers',
                        help='Generate kmers from seqs')

    parser.add_argument('-m', '--kmers-length',
                        type=int,
                        dest='kmersl',
                        metavar='KMER',
                        default=4,
                        help='Generate kmers from seqs')

    args = parser.parse_args()

    args_dict = vars(args)

    return args_dict


def gen_pdb(seq):
    structure = PeptideBuilder.make_extended_structure(seq)
    out = Bio.PDB.PDBIO()
    out.set_structure(structure)
    out.save("%s.pdb" % seq)


def run():
    seqs = list()
    tseqs = list()

    args = get_args()

    if args['list']:
        with open(args['list'], 'r') as f:
            rseqs = f.readlines()
            rseqs = map(lambda x: x.strip(), rseqs)
            tseqs.extend(rseqs)

    if args['peps']:
        tseqs.extend(args['peps'])

    if args['kmers']:
        for seq in tseqs:
            pl = len(seq)
            for i in range(pl - args['kmersl'] + 1):
                seqs.append(seq[i:i + args['kmersl']])
    else:
        seqs = tseqs

    seqs = list(set(seqs))

    for s in seqs:
        gen_pdb(s)

if __name__ == "__main__" and __package__ is None:
    run()
