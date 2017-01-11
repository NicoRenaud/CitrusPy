import time
import sys
import numpy as np
import argparse
from tools import *

from PyQuante import SCF, Molecule
from PyQuante.NumWrap import arange
from PyQuante.cints import coulomb_repulsion
from PyQuante import configure_output
from PyQuante.Constants import hartree2ev
from PyQuante.Ints import getbasis

#configure_output()
sys.stdout = open('citrus.out', 'w')


# thereshold for two2 calculation
EPS = 6

########################################################
########################################################


def main(argv):

	parser = argparse.ArgumentParser(description='PyQuante-Citrus : 2electron for Singlet Fission')
	parser.add_argument('m1',help='xyz file of the first molecule',type=str)
	parser.add_argument('m2',help='xyz file of the second molecule',type=str)

	# define the orbital we want
	parser.add_argument('-o1','--orb1', default = 'h1',help='first molecular orbital (default : homo_1)',type=str)
	parser.add_argument('-o2','--orb2', default = 'h1',help='second molecular orbital (default : lumo_1)',type=str)
	parser.add_argument('-o3','--orb3', default = 'h2',help='third molecular orbital (default : homo_2)',type=str)
	parser.add_argument('-o4','--orb4', default = 'h2',help='fourth molecular orbital (default : lumo_1)',type=str)

	parser.add_argument('-orb1','--orbitals1', default = 'orb1.dat',help='file containing the molecular orbitals of the first molecules',type=str)
	parser.add_argument('-orb2','--orbitals2', default = 'orb2.dat',help='file containing the molecular orbitals of the second molecules',type=str)

	parser.add_argument('-nel1','--numb_elec1', default = 'nel1.dat',help='file containing the number of electrons of the first molecules',type=str)
	parser.add_argument('-nel2','--numb_elec2', default = 'nel2.dat',help='file containing the number of electrons of the second molecules',type=str)
	args=parser.parse_args()

	print '\n\n=================================================='
	print '== PyQuante - Citrus                            =='
	print '== 2 electrons integrals for Singlet-Fission    =='
	print '==================================================\n'


	##########################################################
	##					First Molecule
	##########################################################
	# read the xyz file of the molecule
	print '\t Read molecule 1',

	
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
	print '\t Read SCF of molecule 1',
	sys.stdout.flush()
	t0 = time.time()

	# get the info we need later
	orbs1 = np.loadtxt(args.orbitals1)
	bfs1 = getbasis(mol1,'sto-3g')
	nbf1 = len(bfs1)
	nel1 = int(np.loadtxt(args.numb_elec1))


	t1 = time.time()
	print '\t done (%1.2f min)\n' %(float(t1-t0)/60)


	##########################################################
	##					Second Molecule
	##########################################################

	# read the xyz file of the molecule
	print '\t Read molecule 2',
	
	f = open(args.m2,'r')
	data = f.readlines()
	f.close

	# create the molecule object
	xyz = []
	for i in range(2,len(data)):
		d = data[i].split()
		xyz.append((d[0],(float(d[1]),float(d[2]),float(d[3]))))
	mol2 = Molecule()
	mol2.add_atuples(xyz)
	mol2.translate([0.5,0.5,0.5])
	print '\t\t done' 

	# LDA calculatiion
	print '\t Read SCF of molecule 2',
	sys.stdout.flush()
	t0 = time.time()
	
	# get the info we need later
	orbs2 = np.loadtxt(args.orbitals2)
	bfs2 = getbasis(mol2,'sto-3g')
	nbf2 = len(bfs2)
	nel2 = int(np.loadtxt(args.numb_elec2))


	t1 = time.time()
	print '\t done (%1.2f min)\n' %(float(t1-t0)/60)




	##########################################################
	##					Two electrons Calculations
	##########################################################

	# get the MO coefficients for HOMO/LUMO
	index_homo_1 = nel1/2-1
	index_lumo_1 = index_homo_1+1
	homo1 = orbs1[:,index_homo_1]
	lumo1 = orbs1[:,index_lumo_1]
	homo1 /= np.linalg.norm(homo1)
	lumo1 /= np.linalg.norm(lumo1)

	# get the MO coefficients for HOMO/LUMO
	index_homo_2 = nel2/2-1
	index_lumo_2 = index_homo_2+1
	homo2 = orbs2[:,index_homo_2]
	lumo2 = orbs2[:,index_lumo_2]
	homo2 /= np.linalg.norm(homo2)
	lumo2 /= np.linalg.norm(lumo2)

	# pick the mo we want in the calculations
	cmo1 = get_mo_coeffs(args.orb1,homo1,lumo1,homo2,lumo2)
	cmo2 = get_mo_coeffs(args.orb2,homo1,lumo1,homo2,lumo2)
	cmo3 = get_mo_coeffs(args.orb3,homo1,lumo1,homo2,lumo2)
	cmo4 = get_mo_coeffs(args.orb4,homo1,lumo1,homo2,lumo2)

	# pick the bfs we wnat
	bfs_1 = get_bfs(args.orb1,bfs1,bfs2)
	bfs_2 = get_bfs(args.orb2,bfs1,bfs2)
	bfs_3 = get_bfs(args.orb3,bfs1,bfs2)
	bfs_4 = get_bfs(args.orb4,bfs1,bfs2)

	# size of the bfs
	nbf1 = len(bfs_1)
	nbf2 = len(bfs_2)
	nbf3 = len(bfs_3)
	nbf4 = len(bfs_4)

	# loop over all the atomic orbitals
	INT = 0
	counter = 0
	next_print = 0
	print_increment = 0.1
	nT = nbf1*nbf2*nbf3*nbf4

	print '\t Two-electron integrals',
	sys.stdout.flush()
	t0 = time.time()
	for imo1 in range(nbf1):
		for imo2 in range(nbf2):
			for imo3 in range(nbf3):
				for imo4 in range(nbf4):

					#if ( float(counter)/(nT-1)  >= next_print):
					#	sys.stdout.write('\r\t\t\t\t\t '+str(next_print*100) + ' % done')
					#	sys.stdout.flush()
					#	next_print += print_increment
					#
					#counter += 1

					# pre-coeff of the integrals
					C = cmo1[imo1]*cmo2[imo2]*cmo3[imo3]*cmo4[imo4]
					

					# skip if C is too large
					if abs(C) < 10**-EPS:
						continue

					# skip if the distance between the orbitals is too important
					center_1 = np.array(bfs_1[imo1].origin)
					center_2 = np.array(bfs_2[imo2].origin)
					center_3 = np.array(bfs_3[imo3].origin)
					center_4 = np.array(bfs_4[imo4].origin)


					# distance
					r_12 = np.sqrt(  np.sum((center_1-center_2)**2)  )
					r_34 = np.sqrt(  np.sum((center_3-center_4)**2)  )

					# threshold
					min_12 = min(bfs_1[imo1].norm,bfs_2[imo2].norm)
					eps_12 = np.sqrt( min_12**(-1)  * np.log( (0.5*np.pi/min_12)**3*10**(2*EPS)))

					min_34 = min(bfs_3[imo3].norm,bfs_4[imo4].norm)
					eps_34 = np.sqrt( min_34**(-1)  * np.log( (0.5*np.pi/min_34)**3*10**(2*EPS)))

					if r_12 > eps_12 or r_34 > eps_34:
						continue

					INT +=  C*twoe_cgfp(bfs_1[imo1],bfs_2[imo2],bfs_3[imo3],bfs_4[imo4])

	t1 = time.time()
	print '\n\t Final results :'
	print '\t < %s %s| e/r | %s %s > = %1.3e hartree (%1.3e eV)' %(args.orb1,args.orb2,args.orb3,args.orb4,INT,INT/hartree2ev)
	print '\t Calculations done in %1.2f min ' %(float(t1-t0)/60)
if __name__=='__main__':
	main(sys.argv[1:])
