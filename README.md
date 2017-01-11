# CitrusPy
Compute 2el Couplings for Singlet Fission

This code allows for the computation of the two electronc coupling between the singlet and the double triplet state in Singlet Fission Hamiltonian. The code relies heavily on PyQuante.

The calculation is done in two steps. 
1 - calculation of the ground state of the molecular fragment. This is done with scf.py
2 - calculation of the S->2T coupling with citrus.py or citrus_mpi.py. 

Two examples are provided. The first deals with 2 H2 molecule and the second one with a dimer of benzene molecule.
