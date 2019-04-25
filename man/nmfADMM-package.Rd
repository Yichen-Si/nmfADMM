\name{nmfADMM-package}
\alias{nmfADMM-package}
\alias{nmfADMM}
\docType{package}
\title{
\packageTitle{nmfADMM}
}
\description{
\packageDescription{nmfADMM}
}
\details{

The DESCRIPTION file:
\packageDESCRIPTION{nmfADMM}
\packageIndices{nmfADMM}
Functions:

ConvexNMF_ADMM: convex NMF takes matrixes of any shape with real numbers (potentially mix-signed) and fractorize as X~XWG^T.

SymNMF_ADMM: symmetric NMF takes a symmetric non-negative similarity matrix and factorize as X~HH^T

SymNMFs_ADMM: multiple symmetric NMF takes multiple symmetric non-negative matrices as input and extract the shared low rank (clustering) pattern.
Currently it only accepts two matrices, one supposed to be similarity measure while no constrain on the other.

NMF_ADMM: standard NMF implemented with ADMM, taking a (m x n) non-negative matrix.

}
\author{
\packageAuthor{nmfADMM}

Maintainer: \packageMaintainer{nmfADMM}
}
\references{
Convex and semi-nonnegative matrix factorizations.
C. Ding, T. Li and M. I. Jordan. 2010. IEEE Transactions on Pattern Analysis and Machine Intelligence.

Symmetric Nonnegative Matrix Factorization for Graph Clustering.
D. Kuang, C. Ding and H. Park. 2012. SIAM.

Alternating direction method of multipliers for non-negative matrix factorization with the beta-divergence.
D. L. Sun and C. Févotte. 2014. ICASSP.
}
\keyword{ package }
\examples{
\dontrun{
  ConvexNMF_ADMM(V, k, cluster, verbose=1, maxiter = 100, rho=3, tol=1e-3)
  SymNMFs_ADMM(Vp, Vm, k, cluster, maxiter = 100)
  SymNMF_ADMM(V, k, cluster, maxiter = 100)
  NMF_ADMM(V, k, cluster, maxiter = 100)
}
}
