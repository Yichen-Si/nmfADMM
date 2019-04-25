# nmfADMM
A Rcpp package implementing NMF variations using ADMM 

Functions:

ConvexNMF_ADMM: convex NMF takes matrixes of any shape with real numbers (potentially mix-signed) and fractorize as X~XWG^T.

SymNMF_ADMM: symmetric NMF takes a symmetric non-negative similarity matrix and factorize as X~HH^T

SymNMFs_ADMM: multiple symmetric NMF takes multiple symmetric non-negative matrices as input and extract the shared low rank (clustering) pattern.
Currently it only accepts two matrices, one supposed to be similarity measure while no constrain on the other.

NMF_ADMM: standard NMF implemented with ADMM, taking a (m x n) non-negative matrix.
