import time
import sys
import numpy as np
import argparse
from tools import *
from mpi4py import MPI

from PyQuante import SCF, Molecule
from PyQuante.NumWrap import arange
from PyQuante.cints import coulomb_repulsion
from PyQuante import configure_output
from PyQuante.Constants import hartree2ev
from PyQuante.Ints import getbasis

#configure_output()

# thereshold for two2 calculation
EPS = 6

########################################################
########################################################


def main(argv):

	parser = argparse.ArgumentParser(description='PyQuante-Citrus : 2electron for Singlet Fission')
	parser.add_argument('m1',help='xyz file of the first molecule',type=str)
	parser.add_argument('m2',help='xyz file of the second molecule',type=str)

	# define the orbital we want
	parser.add_argument('-s1','--state1', default = 'h1',help='first molecular orbital (default : homo_1)',type=str)
	parser.add_argument('-s2','--state2', default = 'h1',help='second molecular orbital (default : homo_1)',type=str)
	parser.add_argument('-s3','--state3', default = 'h2',help='third molecular orbital (default : homo_2)',type=str)
	parser.add_argument('-s4','--state4', default = 'h2',help='fourth molecular orbital (default : homo_2)',type=str)

	parser.add_argument('-mo1','--molecular_orbitals1', default = 'orb1.dat',help='file containing the molecular orbitals of the first molecules',type=str)
	parser.add_argument('-mo2','--molecular_orbitals2', default = 'orb2.dat',help='file containing the molecular orbitals of the second molecules',type=str)

	parser.add_argument('-nel1','--numb_elec1', default = 'nel1.dat',help='file containing the number of electrons of the first molecules',type=str)
	parser.add_argument('-nel2','--numb_elec2', default = 'nel2.dat',help='file containing the number of electrons of the second molecules',type=str)
	args=parser.parse_args()

	############################
	# Initialize MPI
	############################
	comm = MPI.COMM_WORLD
	rank = comm.rank
	############################

	if rank == 0:
		print '\n\n=================================================='
		print '== PyQuante - Citrus - MPI                      =='
		print '== 2 electrons integrals for Singlet-Fission    =='
		print '== Execution on %d processors 		  	==' %comm.size
		print '==================================================\n'


	##########################################################
	##	All the process read the data
	##  it might actually be faster than broadcasting
	##########################################################
	
	##########################################################
	##					First Molecule
	##########################################################
	# read the xyz file of the molecule
	if rank == 0:
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
	if rank == 0:
		print '\t\t done' 

	# LDA calculatiion
	if rank == 0:
		print '\t Read SCF of molecule 1',
	sys.stdout.flush()
	t0 = time.time()

	# get the info we need later
	orbs1 = np.loadtxt(args.molecular_orbitals1)
	bfs1 = getbasis(mol1,'sto-3g')
	nbf1 = len(bfs1)
	nel1 = np.loadtxt(args.numb_elec1)


	t1 = time.time()
	if rank == 0:
		print '\t done (%1.2f min)\n' %(float(t1-t0)/60)


	##########################################################
	##					Second Molecule
	##########################################################

	# read the xyz file of the molecule
	if rank == 0:
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
	if rank == 0:
		print '\t\t done' 

	# LDA calculatiion
	if rank == 0:
		print '\t Read SCF on molecule 2',
	sys.stdout.flush()
	t0 = time.time()
	
	# get the info we need later
	orbs2 = np.loadtxt(args.molecular_orbitals2)
	bfs2 = getbasis(mol2,'sto-3g')
	nbf2 = len(bfs2)
	nel2 = np.loadtxt(args.numb_elec2)


	t1 = time.time()
	if rank == 0:
		print '\t done (%1.2f min)\n' %(float(t1-t0)/60)




	##########################################################
	##					Two electrons Calculations
	##########################################################

	# get the MO coefficients for HOMO/LUMO
	index_homo_1 = int(nel1/2)-1
	index_lumo_1 = int(index_homo_1)+1
	homo1 = orbs1[:,index_homo_1]
	lumo1 = orbs1[:,index_lumo_1]

	# normalize the coefficients
	homo1 /= np.linalg.norm(homo1)
	lumo1 /= np.linalg.norm(lumo1)

	# get the MO coefficients for HOMO/LUMO
	index_homo_2 = int(nel2/2)-1
	index_lumo_2 = int(index_homo_2)+1
	homo2 = orbs2[:,index_homo_2]
	lumo2 = orbs2[:,index_lumo_2]

	# noralize the coefficients
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
	##	Each process makes a bit of the calculations
	##  and send their values to the master
	##	which prints the final values
	##########################################################

	# total number of terms
	NTOT = nbf1*nbf2*nbf3*nbf4

	# number of terms per procs
	npp = int(float(nbf1)/comm.size)

	# make sure that the last term compute all the remaining
	if rank < comm.size - 1:
		terms = range(rank*npp,(rank+1)*npp)
	elif rank == comm.size-1:
		terms = range(rank*npp,nbf1)
	


	t0 = time.time()
	if rank == 0:
		print '\t Two-electron integrals'
		nterm_tot = 0
		nterm = []
		for i in range(comm.size):
			if i < comm.size-1:
				nterm.append(len(range(i*npp*N3,(i+1)*npp*N3)))
			if i == comm.size-1:
				nterm.append(len(range(i*npp*N3,nbf1*N3)))
			nterm_tot += nterm[i]

		if nterm_tot != NTOT:
			print '\t\t Error : missing terms'
			for i in range(comm.size):
				print '\t\t Proc %d \t\t %1.2e terms' %(i,nterm[i])
			print '\t\t Total \t\t\t %1.2e' %(nterm_tot)
			print '\t\t Expected \t\t %1.2e' %(NTOT)


	sys.stdout.flush()

	if rank ==0:
		t0 = time.time()
		
	for imo1 in terms:
		for imo2 in range(nbf2):
			for imo3 in range(nbf3):
				for imo4 in range(nbf4):

					if rank ==0:
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

					INT +=  C*twoe_cgfp(bfs_1[imo1],bfs_2[imo2],bfs_3[imo3],bfs_4[imo4])



   	##############################
   	# MASTER SLAVE COMMUNICATION
   	##############################
   	comm.barrier()

   	if rank == 0:
   		print '\t\t MPI Communication'

   	if rank == 0:
   		buffer = np.zeros(1)
   		for iC in range(1,comm.size):
   			comm.Recv(buffer,source=iC)
   			INT = INT + buffer[0]
   			print '\t\t Proc [%03d] : data received from [%03d]' %(0,iC)
   	else:
   		comm.Send(np.array(INT),dest=0)

   	if rank == 0:
   		t1 = time.time()
   		print'\t Two electrons integrals computed in %1.2f min on %d procs' %(float(t1-t0)/60,comm.size)
   	sys.stdout.flush()

   	##################################
   	# Print the restults
   	##################################
   	comm.barrier()
   	if rank == 0:
   		print '\n\t Final results :'
		print '\t < %s %s| e/r | %s %s > = %1.3e hartree (%1.3e eV)' %(args.state1,args.state2,args.state3,args.state4,INT,INT/hartree2ev)


  

if __name__=='__main__':
	main(sys.argv[1:])
