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

	parser = argparse.ArgumentParser(description='SCF calculation for a given molecule')

	# XYZ File of the molecule
	parser.add_argument('mol',help='xyz file of the first molecule',type=str)

	# unit in the xyz file of the molecule
	parser.add_argument('-u', '--units', default = 'angs',help=' Unit in the xyz file (angs or bohr) (default : angs)',type=str)

	# basis set
	parser.add_argument('-b','--basis', default = 'sto-3g',help=' basis set used for the calculation (default : sto-3g)',type=str)
	
	# level of theory 
	parser.add_argument('-l','--level', default = 'DFT',help='Level of theory used in the calculation (default : DFT)',type=str)

	# XC funcional for DFT
	parser.add_argument('-xc', default = 'LDA',help='XC Functional used in the calculation (default : LDA)',type=str)

	args=parser.parse_args()

	print '\n\n\t =================================================='
	print '\t == PyQuante - Citrus                            =='
	print '\t == SCF Caclulations of single molecule          =='
	print '\t ==================================================\n'


	##########################################################
	##					Read the  Molecule
	##########################################################
	# read the xyz file of the molecule
	print '\t Read molecule\n\t File %s \n\t Units : %s' %(args.mol,args.units),

	f = open(args.mol,'r')
	data = f.readlines()
	f.close

	# create the molecule object
	xyz = []
	for i in range(2,len(data)):
		d = data[i].split()
		xyz.append((d[0],(float(d[1]),float(d[2]),float(d[3]))))
	mol = Molecule(units=args.units)
	mol.add_atuples(xyz)
	print '\t\t\t done\n'

	##########################################################
	##					SCF Caculations
	##########################################################

	# if DFT
	if args.level == 'DFT':
		print '\t %s calculation \n\t XC = %s Basis = %s ' %(args.level,args.xc,args.basis),
		sys.stdout.flush()
		t0 = time.time()
		solver = SCF(mol,method=args.level,basis=args.basis,functional=args.xc)
		solver.iterate()

		t1 = time.time()
		print '\t done in %1.2f min' %(float(t1-t0)/60)
		print '\t\t\t\t\t Etot = %1.3f ha\n' %solver.energy

	elif args.level == 'HF':
		print '\t %s calculation \n\t Basis = %s' %(args.level,args.basis),
		sys.stdout.flush()
		t0 = time.time()
		solver = SCF(mol,method=args.level,basis=args.basis)
		solver.iterate()

		t1 = time.time()
		print '\t\t done in %1.2f min' %(float(t1-t0)/60)
		print '\t\t\t\t\t Etot = %1.3f ha\n' %solver.energy

	##########################################################
	##		 Export the result for 2el calculations
	##########################################################

	# get the info we need later
	orbs1 = solver.solver.orbs
	bfs1 = solver.basis_set.get()
	nbf1 = len(bfs1)
	nel1 = mol.get_nel()

	# save the orbitals and and info
	name_mol = re.split(r'\.|/',args.mol)[-2]
	np.savetxt(name_mol+'_orbitals.dat',orbs1,header=args.basis)
	
	print '\t MO exported in %s' %(name_mol+'_orbitals.dat')
	
	print '\n\t ==================================================\n'

if __name__=='__main__':
	main(sys.argv[1:])
