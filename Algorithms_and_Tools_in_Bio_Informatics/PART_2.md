# Similiarity of sequences and sequemnt allignemnt
- What is a match, mismatch, what is a gap

match = 2 nucleotides same
mismatch not
gap = insertion or deltion occured

Requirements for allignement algorithms
- consider gaps, insertation deltion replacemtn and the chemical properties of the proteins
- fast  
- Dynamic programming


Scoring system
- ssubstition and identity matrices
- substitution: 1s in the diagonal



Odd's ration: this is the value how we score an allignment, 
prob that 2 sequences derive from mutation 
or they are not related and are random

- P (A,b | M(muation))
_______________________
 P (A,b | R(random)) 


Use Logarithm: sum of the logs
We use logarithm to make it human and computer readable due to mulitplication of the number below 1 



 Strategy to find if sequences are realted =
 
Position specific scoring matrix

3 types of a matrix

- 1. Count: count each occureance 
- 2. Frequency: each column adds up to 1
- 3. Weight Matrix: ln (Freq/ 0.25 (4 nucleotides )),  sum of all the weight in the matrix is the result, higher than 1 means that it is related

What is concensucs seq: most commonly occured nucleotides


Position independent scoring
- Identity Matrices
- PAMs: They are for amino acids but also contain info about protein
- BLOSUM: 

PAM (Perecent Accepted Mutation)
- use phylogenetic tree to get to a Replacement Matrix and Rleative Frequences  LARGE TRIANGLE 
- 1 PAM is the time period in which 1% of the amino acids mutate

The lower the PAM the closer the sequences
PAM 250 = 20% similiarity
PAM 1 = 99%
