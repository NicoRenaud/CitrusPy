import time
import sys
import numpy as np
import argparse
import re
#from tools import *

from PyQuante import SCF, Molecule
from PyQuante.NumWrap import arange
from PyQuante.cints import coulomb_repulsion
from PyQuante import configure_output
from PyQuante.Constants import hartree2ev

#configure_output()

# thereshold for two2 calculation
EPS = 1E-6

########################################################
########################################################


def main(argv):

	parser = argparse.ArgumentParser(description='SCF (LDA/STO-3G) calculation for a given molecule')
	parser.add_argument('m1',help='xyz file of the first molecule',type=str)
	
	args=parser.parse_args()

	print '\n\n=================================================='
	print '== PyQuante - Citrus                            =='
	print '== SCF Caclulations of isolated molecule        =='
	print '==================================================\n'


	##########################################################
	##					First Molecule
	##########################################################
	# read the xyz file of the molecule
	print '\t Read molecule',

	f = open(args.m1,'r')
	data = f.readlines()
	f.close

	# create the molecule object
	xyz = []
	for i in range(2,len(data)):
		d = data[i].split()
		xyz.append((d[0],(float(d[1]),float(d[2]),float(d[3]))))
	mol1 = Molecule()
	mol1.add_atuples(xyz)
	print '\t\t done' 

	# LDA calculatiion
	print '\t SCF-DFT on molecule',
	sys.stdout.flush()
	t0 = time.time()
	lda = SCF(mol1,method="DFT",basis="sto-3g")
	lda.iterate()

	# get the info we need later
	orbs1 = lda.solver.orbs
	bfs1 = lda.basis_set.get()
	nbf1 = len(bfs1)
	nel1 = mol1.get_nel()

	# save the orbitals and nel
	name_mol = re.split(r'\.|/',args.m1)[-2]
	np.savetxt(name_mol+'_orbitals.dat',orbs1)
	np.savetxt(name_mol+'_nel.dat',[nel1])

	t1 = time.time()
	print '\t\t done (%1.2f min)\n' %(float(t1-t0)/60)


if __name__=='__main__':
	main(sys.argv[1:])
