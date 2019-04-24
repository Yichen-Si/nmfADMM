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
}
\author{
\packageAuthor{nmfADMM}

Maintainer: \packageMaintainer{nmfADMM}
}
\references{
~~ Literature or other references for background information ~~
}
~~ Optionally other standard keywords, one per line, from file KEYWORDS in ~~
~~ the R documentation directory ~~
\keyword{ package }
\seealso{
~~
}
\examples{
\dontrun{
  ConvexNMF_ADMM(V, k, cluster, maxiter = 100)
  SymNMFs_ADMM(Vp, Vm, k, cluster, maxiter = 100)
  SymNMF_ADMM(V, k, cluster, maxiter = 100)
}
}
