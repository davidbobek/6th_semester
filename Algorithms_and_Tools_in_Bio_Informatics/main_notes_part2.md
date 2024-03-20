# Part 2: Algorithms and Tools in Bioinformatics


## Motivation
    - Why do we need to compare sequences?
     - The answer is to find out if the sequences are related or not and how much they are related
    
### Sequence Comparison
    - Comparison of nucleotide sequences: 4 nucleotides
    - Comparison of amino acid sequences: 20 amino acids

### Similarity vs Allignment
    - Similarity of two sequences is the measure of how well the sequences match
    - Allignment is the process of overlapping the sequences to find the best match using gaps and mismatches

### Requirements for allignement algorithms
    - consider: (gaps, mismatch and match)
    - fast 
    - Dynamic programming
    - Need to have a scoring system to score the allignemnt


## Similarity of Sequences/ Scoring matrices
    - Types of Scoring matrices
        - Substitution matrices
        - Position specific scoring matrix
            - Count
            - Frequency
            - Weight Matrix
        - Position independent scoring 
            - Identity matrices
            - PAMs
            - BLOSUMs

### Substitution matrices
    - Used by BLAST
    - Used to score the allignemnt
    - 1s in the diagonal
    - The higher the score the better the allignemnt

    - Example
    
    A = AGGACT
    B = GTGAGT

    Table
    +---+---+---+---+---+---+
    |   | A | G | C | T |
    |---|---|---|---|---|
    | A | 1 | 0 | 0 | 0 |
    | G |-2 | 1 | 0 | 0 |
    | T |-2 |-2 | 1 | 0 |
    | C |-2 |-2 |-2 | 1 |
    +---+---+---+---+---+---+

    Result: -2 + (-2) +1 +1 + (-2) +1 = -3 
    Each of the letters are compared to each other and the score is added up based on the table

### Odds Ratio
    - This is the value how we score an allignment
    - prob that 2 sequences derive from mutation 
    - or they are not related and are random

    - P (A,b | M(muation))
    _______________________
     P (A,b | R(random)) 

    - Use Logarithm: sum of the logs
    - We use logarithm to make it human and computer readable due to mulitplication of the number below 1

    
### Position specific scoring matrix
    - 3 different types of matrices

        - Count: count nucleotides in each position in Alligned sequences
        - Frequency: Frequency of each nucleotide in each position, columns add up to 1
        - Weight Matrix: ln (Freq/ 0.25 (4 nucleotides )),  sum of all the weight in the matrix is the result, higher than 1 means that it is related

### Position independent scoring
    - Identity Matrices
        - Diagonal is the same positive number
        - Rest are negative 
        
    - PAMs: They are for amino acids but also contain info about protein
    - BLOSUM:



## Global/Local Alignments
## Heuristic Methods
## Multiple Sequence Alignments