# RNAonGraph: a graph representational learning approach for RNAs
This is a repository dedicated to neural representational learning of RNA secondary structures.

Prior to this date, the imputation of RNA secondary structures has formed an important line of investigation in the community of computational molecular biology, and various folding algorithms have been developed to make more accurate and more efficient predictions.

The structures (secondary, tertiary) of RNAs have also been linked to their functions, especially for the family of non-coding RNAs. The same rules apply to the messenger RNAs, such as the structural accessibility when interacting with RNA binding proteins, and the subsequent subcellular localization.  

However, due to the highly stochastic nature of RNA folding and the uncertainty that arise thereof, the global structural characterization of slightly longer RNA molecules has been an unsatisfactory, although one may argue that their local structures would suffice to harbour interesting signals correlated with their functions.

Then, some immediate questions include:
- how to more effectively learn a representation of RNA secondary structures, either local or global; and, 
- how to adequately extract signals from these structures.

To answer these questions, we seek an end-to-end learning approach directly defined on RNA graphs, enabled by a more general framework called graph neural nets that has been recently adpated to other biological and chemical domains.

## Local RNA folding and base-pairing probability annotated adjacency matrix

The first step is to obtain local RNA secondary structures for long RNAs, and for that purpose, a number of folding algorithms are considered in this project

- RNAplfold
    * Rather than a single graph, RNAplfold gives the equilibrium base pairing probabilities inside a local folding window
    * Base-pairs appearing in multiple folding windows have averaged base pairing probability 
    * A larger folding window such as 150 bases is strongly advised against the default size (70 bases)
    
- RNAshapes
    * RNA local folding considering abstract shapes, which essentially outputs hyper-graphs of RNA secondary structures
    * Its sampling option provided can be used to estimate base pairing with probabilities
    * A large folding window such as 150 bases is again suggested
    
It has been known that RNA folding is stochastic and instead of folding into one single most stable secondary structure, a population of the same RNA sequence may simultaneously adopt other possible folding configurations.

To better model RNA secondary structures and to ease the subsequent representational learning detail, we propose to fill in the usually binary adjacency matrix with base pairing probabilities considering all possible secondary structures from the ensemble of RNA stochastic folding.  
    
## Baselines and graph neural nets

The baseline is a sequence model that takes as input plain RNA sequences, using a set of convolutions to learn motifs, then using a bidirectional LSTM to learn longer ranger dependencies, and finally a Set2Set module to globally pool the RNA along its nucleic axis to make a prediction of whether or not the sequence contains a binding site.

The graph neural net model simply replaces the convolutions with a graph message-passing layers connected by a LSTM, which arguably performs similar functions but inside the neighbourhood around each nucleotide in the two dimensional space specified by the adjacency matrix.

- We model two types of bonds in a secondary RNA graphs, that are the covalent bond forming the backbone of RNA, and the hydrogen bond connecting a base pair. We also consider direction of the RNA from 5' to 3', which effectively gives us four relations in the end. 

