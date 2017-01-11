import time
import sys
import numpy as np
import argparse

from PyQuante import SCF, Molecule
from PyQuante.NumWrap import arange
from PyQuante.cints import coulomb_repulsion
from PyQuante import configure_output
from PyQuante.Constants import hartree2ev
from PyQuante.Ints import getbasis, coulomb

#configure_output()
#sys.stdout = open('citrus.out','w')

# thereshold for two2 calculation
EPS = 6

########################################################
## MAIN FUNCTION
########################################################

def main(argv):

	parser = argparse.ArgumentParser(description='PyQuante-Citrus : 2electron for Singlet Fission')

	# XYZ files of the first and second molecules
	parser.add_argument('m1',help='xyz file of the first molecule',type=str)
	parser.add_argument('m2',help='xyz file of the second molecule',type=str)

	# translation vector for the second molecule
	parser.add_argument('-t','--trans_m2',nargs='+', default=[0.,0.,0.],help='translation vector for the second molecule (default: [0,0,0])',type=float)

	# units in the xyz file 
	# we assume that both files have the same unit 
	parser.add_argument('-u','--units', default='angs',help='units in the XYZ files angs or bohr (default : angs)',type=str)

	# define the molecular orbital we want in the calculation
	# (s1 s2 | e/r | s3 s4) Warning Chemist notation (right ?)
	parser.add_argument('-s1','--state1', default = 'h1',help='first molecular orbital (default : homo_1)',type=str)
	parser.add_argument('-s2','--state2', default = 'h1',help='second molecular orbital (default : homo_1)',type=str)
	parser.add_argument('-s3','--state3', default = 'h2',help='third molecular orbital (default : homo_2)',type=str)
	parser.add_argument('-s4','--state4', default = 'h2',help='fourth molecular orbital (default : homo_2)',type=str)

	# files containing the HF or KS orbitals of the two molecules
	# these files are obtained via the scf.py program
	parser.add_argument('-mo1','--molecular_orbitals1', default = 'orb1.dat',help='file containing the molecular orbitals of the first molecules',type=str)
	parser.add_argument('-mo2','--molecular_orbitals2', default = 'orb2.dat',help='file containing the molecular orbitals of the second molecules',type=str)

	# done 
	args=parser.parse_args()

	print '\n\n\t =================================================='
	print '\t == PyQuante - Citrus                            =='
	print '\t == 2 electrons integrals between MOs            =='
	print '\t ==================================================\n'


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
	mol1 = Molecule(units=args.units)
	mol1.add_atuples(xyz)
	print '\t\t done' 

	# Read the SCF data
	print '\t Read SCF of molecule 1',
	sys.stdout.flush()
	t0 = time.time()

	# read the molecular orbitals coefficients
	orbs1 = np.loadtxt(args.molecular_orbitals1)

	# read the basis set that was used in the SCF
	with open(args.molecular_orbitals1,'r') as f:
		basis1 = f.readline()
	basis1 = basis1.split()[1]

	# construct the basis info
	bfs1 = getbasis(mol1,basis1)
	nbf1 = len(bfs1)	
	nel1 = mol1.get_nel()

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
	mol2 = Molecule(units=args.units)
	mol2.add_atuples(xyz)
	print '\t\t done' 

	# translate the second molecule
	mol2.translate(args.trans_m2)

	# Read the SCF data
	print '\t Read SCF on molecule 2',
	sys.stdout.flush()
	t0 = time.time()
	
	# read the molecular orbitals coefficients
	orbs2 = np.loadtxt(args.molecular_orbitals2)

	# read the basis set that was used in the SCF
	with open(args.molecular_orbitals2,'r') as f:
		basis2 = f.readline()
	basis2 = basis2.split()[1]

	# construct the basis info
	bfs2 = getbasis(mol2,basis2)
	nbf2 = len(bfs2)
	nel2 = mol2.get_nel()

	t1 = time.time()
	print '\t done (%1.2f min)\n' %(float(t1-t0)/60)

	##########################################################
	##			Extract the mo inofrmations
	##########################################################

	# get the MO coefficients for HOMO/LUMO
	index_homo_1 = nel1/2-1
	index_lumo_1 = index_homo_1+1
	homo1 = orbs1[:,index_homo_1]
	lumo1 = orbs1[:,index_lumo_1]

	# normalize the coefficients
	homo1 /= np.linalg.norm(homo1)
	lumo1 /= np.linalg.norm(lumo1)

	# get the MO coefficients for HOMO/LUMO
	index_homo_2 = nel2/2-1
	index_lumo_2 = index_homo_2+1
	homo2 = orbs2[:,index_homo_2]
	lumo2 = orbs2[:,index_lumo_2]

	# normalize the coefficients
	homo2 /= np.linalg.norm(homo2)
	lumo2 /= np.linalg.norm(lumo2)

	# pick the mo we want in the calculations
	cmo1 = get_mo_coeffs(args.state1,homo1,lumo1,homo2,lumo2)
	cmo2 = get_mo_coeffs(args.state2,homo1,lumo1,homo2,lumo2)
	cmo3 = get_mo_coeffs(args.state3,homo1,lumo1,homo2,lumo2)
	cmo4 = get_mo_coeffs(args.state4,homo1,lumo1,homo2,lumo2)

	# pick the bfs we wnat
	bfs_1 = get_bfs(args.state1,bfs1,bfs2)
	bfs_2 = get_bfs(args.state2,bfs1,bfs2)
	bfs_3 = get_bfs(args.state3,bfs1,bfs2)
	bfs_4 = get_bfs(args.state4,bfs1,bfs2)

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

	##########################################################
	##			Two electrons Calculations
	##########################################################


	print '\t Two-electron integrals',
	sys.stdout.flush()
	for imo1 in range(nbf1):
		for imo2 in range(nbf2):
			for imo3 in range(nbf3):
				for imo4 in range(nbf4):

					if ( float(counter)/(nT-1)  >= next_print):
						sys.stdout.write('\r\t\t\t\t\t '+str(next_print*100) + ' % done')
						sys.stdout.flush()
						next_print += print_increment

					counter += 1

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

					# compute the coulomb integral
					INT += C*coulomb(bfs_1[imo1],bfs_2[imo2],bfs_3[imo3],bfs_4[imo4])

	print '\n\n\t Final results :'
	print '\t ( %s %s | e/r | %s %s ) = %1.3e eV (%1.3e hartree)' %(args.state1,args.state2,args.state3,args.state4,INT/hartree2ev,INT)
	print '\n\t ==================================================\n'

########################################################

########################################################
## Get the mo coeffs we want
########################################################
def get_mo_coeffs(argument,h1,l1,h2,l2):

    switcher = {
        'h1': h1,
        'l1': l1,
        'h2': h2,
        'l2': l2,
    }
    return switcher.get(argument, "nothing")


########################################################
## Get the mo coeffs we want
########################################################
def get_bfs(argument,bfs1,bfs2):

    switcher = {
        'h1': bfs1,
        'l1': bfs1,
        'h2': bfs2,
        'l2': bfs2,
    }
    return switcher.get(argument, "nothing")



if __name__=='__main__':
	main(sys.argv[1:])
