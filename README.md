# RNAonGraph: a graph representational learning approach for RNAs
This is a repository dedicated to neural representational learning of RNA secondary structures.

Prior to this date, the imputation of RNA secondary structures has formed an important line of investigation in the community of computational molecular biology, and various folding algorithms have been developed to make more accurate and more efficient predictions.

The structures (secondary, tertiary) of RNAs have also been linked to their functions, especially for the family of non-coding RNAs. The same rules apply to messenger RNAs, such as the structural accessibility when interacting RNA binding proteins, and the subsequent subcellular localization.  

However, due to the highly stochastic nature of RNA folding and the uncertainty that arise thereof, the global structural characterization of slightly longer RNA molecules has been an unsatisfactory, although one may argue that their local structures would suffice to harbour interesting signals correlated with their functions.

Then, some immediate questions include:
- how to more effectively learn a representation of RNA secondary structures, either local or global; and, 
- how to adequately extract signals from these structures.

To answer these questions, we seek an end-to-end learning approach directly defined on RNA graphs, enabled by a more general framework called graph neural nets that has been recently adpated to other biological and chemical domains.

## Local RNA folding

The first step is to obtain local RNA secondary structures for long RNAs, and for that purpose, a number of folding algorithms are considered in this project

- RNAplfold
    * Rather than a single graph, RNAplfold gives the equilibrium base pairing probabilities
    * A larger folding window such as 150 bases is usually advised instead of the default value (70 bases)
    
- RNAshapes
    * Provides a sampling option, which can be used to annotate base pairing with probabilities
    * Once again, use a large folding window such as 150 bases

