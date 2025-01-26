# CATH Database:

The CATH hierarchy is a classification system for protein domain structures. It organizes protein domains into four main levels:

1. Class (C): Domains are categorized based on their secondary structure content. The main classes are:
   - Mainly Alpha (predominantly α-helices)
   - Mainly Beta (predominantly β-sheets)
   - Mixed Alpha-Beta (significant amount of both α-helices and β-sheets)
   - Few Secondary Structures

2. Architecture (A): This level describes the overall shape and arrangement of secondary structures in three-dimensional space.

3. Topology/fold (T): At this level, domains are grouped based on the connectivity and arrangement of their secondary structure elements.

4. Homologous superfamily (H): Domains are classified into this level if there is strong evidence of evolutionary relatedness. This is based on structural, functional, and sequence similarities.

The CATH database was created in the mid-1990s by Professor Christine Orengo and colleagues at University College London. It uses a combination of automatic methods and manual curation to classify protein domains from experimentally determined structures in the Protein Data Bank. The original goal of this classification was to help in identifying relationships between protein structures. In biology, it is known that stucture informs function. To understand structure can lead to the understanding of protein evolution and function!
 
 ## cath_3class

For this work, I am focusing on the simplest dataset, `cath_3class.npz`. This dataset plays a significant role in the CATH classification system by focusing on the Class level of the hierarchy. 

1. Class-level focus: The dataset concentrates on three of the four main classes in CATH: Mainly Alpha, Mainly Beta, and Alpha Beta (no "Few secondary structures" class due to its small size and heterogeneity).

2. Balanced representation: The dataset reduces each of the three classes to have an equal number of members.

3. Structural simplification: It contains only Carbon-alpha positions for each protein (each position represents a single atom for each amino acid).

4. Secondary structure detection: The main task facilitated by this dataset is the detection and quantification of protein secondary structures (alpha helices and beta strands).

# Training Data

The data used for training are the (x,y,z) coordinates of single Carbon-alpha position for each amino acide within each protein. From these (x,y,z), coordiantes, the hope is to be able to predict whether a protein is 'Mainly Alpha', 'Mainly Beta', or 'Alpha-Beta'.

The data itself is of shape (16962, 1202, 3) where:
- 16962 is the number of proteins in the dataset
- 1202 is the number of columns (each column representing an atom in the protein)
  - The max atoms in a single protein in this data is 1202 (therefore, the data itself resembles a sparse matrix of sorts - (x, y, z)'s instead of 1's)
- 3 is the (x,y, z) coordinate of each atom in the protein

In more detail, the first protein has shape (1202, 3) with the following values:

```data['positions'].shape````

```bash
array([[-0.46251011, 14.0216713 , -0.11604881],
       [ 2.52248955, 14.92267227,  2.01995087],
       [ 0.57148933, 16.07967186,  5.23795128],
       ...,
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ]])
```

Each row in the array above is the (x,y,z) coordinate of a single atom in the protein. This protein has 175 atoms and so the rest of the 1027 rows (1202-175) are arrays with [0, 0, 0] for each coordinate.

# Visualizing the Data:

Based of the (x, y, z) coordinates, we can see the general shape of each protein with their accompanying label:

### 1. Mainly Alpha
![alpha_420](https://github.com/user-attachments/assets/f31912be-6a52-490e-bade-f3fa53d5146f)

### 2. Mainly Beta
![beta_150](https://github.com/user-attachments/assets/7b0d5a56-2e63-421e-a03f-30bcabdd4f0b)

### 3. Alpha-Beta
![alpha_beta_16961](https://github.com/user-attachments/assets/264ab5cf-3eb8-4e01-a69a-8f2fcbfe94df)

In this experiment, the goal is to try different Neural Networks/Convolutional Nueral Networks architectures, tune their hyperparameters, and create a model that can better learn the specific relationships/features of the 3D data in order to be able to predict the correct secondary structures.

Citations:
- http://www.pdg.cnb.uam.es/cursos/BioInfo2002/pages/Farmac/CATH/class.html
- https://pmc.ncbi.nlm.nih.gov/articles/PMC4678953/
- https://en.wikipedia.org/wiki/CATH_database
- https://comis.med.uvm.edu/VIC/coursefiles/MD540/MD540-Protein_Organization_10400_574581210/Protein-org/Protein_Organization_print.html
- https://pubmed.ncbi.nlm.nih.gov/9309224/
- https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/protein-hierarchical-structure
- https://pmc.ncbi.nlm.nih.gov/articles/PMC2686597/
- https://www.khanacademy.org/science/biology/macromolecules/proteins-and-amino-acids/a/orders-of-protein-structure
- https://www.ucl.ac.uk/orengo-group/group-resources/cath-protein-structure-classification
