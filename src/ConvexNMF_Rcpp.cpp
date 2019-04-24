#include <RcppArmadillo.h>

#include <string>
#include <vector>

using namespace Rcpp;
using namespace std;

// [[Rcpp::depends(RcppArmadillo)]]

//' Convex NMF for mix-signed matricies using ADMM
//'
//' This function implements a ADMM for NMF taking a single matrix
//' of any shape and potentially mix-signed.
//' X~XWG with W & G non-negative.
//'
//' @param V         A (m x n) vector of data matrix with m features and n units
//' @param k         An integer to specify desired rank of the low rank factorization result
//' @param cluster   A (k x 1) vector of integer 0~(k-1) indicating initial cluster assignment
//' @param verbose   A 0/1 integer (default 0). If 1: output the relative error after each iteration
//' @param maxiter   An integer (default 100) for the maximal number of iterations
//' @param fixediter An integer (optional) for a fixed number of iterations
//' @param rho       Penalty parameter in ADMM (default 3)
//' @param tol       Tolerance parameter to declare convergence
//' @return A list containing the following values
//' * W     : A (n x k) non-negative matrix containing W in X~XWG
//' * G     : A (k x n) non-negative matrix containing G in X~XWG
//' * Error : A vector containing the relative distance of this approximation in each iteration
//' @export
// [[Rcpp::export]]
List ConvexNMF_ADMM(arma::mat& V, int k,
                    arma::uvec cluster,
                    bool verbose = 0,
                    int maxiter = 100, int fixediter = 0,
                    double rho = 3,
                    double tol = 1e-3) {

  int n = (int) V.n_cols; // Number of data points
  int m = (int) V.n_rows; // Number of features
  double sqt = arma::accu(arma::pow(V, 2));

  if (fixediter != 0)
    maxiter = fixediter;

  NumericVector Error(maxiter, -1.0);
  double err0 = 0, err1 = 0;

  // Initialization
  arma::mat Ik = arma::eye<arma::mat>(k,k);
  arma::mat X(m,n), FG(m,n), F(m,k,arma::fill::zeros);
  arma::mat W(n,k,arma::fill::zeros), G(k,n,arma::fill::ones);
  arma::mat alphax(m,n,arma::fill::ones), alphaf(m,k,arma::fill::randu);
  arma::mat alphaw(n,k,arma::fill::randu), alphag(k,n,arma::fill::randu);

  for (int i=0; i<n; ++i)
    W(i, cluster[i]) = 1;

  G = W.t();
  arma::rowvec nk = arma::sum(W,0);
  W.each_row() /= nk;
  F = V * W;
  X = F * G;

  arma::mat Wp = W;
  arma::mat Gp = G;

  // Precompute
  arma::mat VVinv = arma::eye<arma::mat>(n,n);
  VVinv += V.t() * V;
  VVinv = arma::inv_sympd(VVinv);

  // Iteration
  // int iter;
  for (int iter = 0; iter < maxiter; ++iter) {

    F = (X*G.t()+V*W+(alphax*G.t()-alphaf)/rho) * arma::inv_sympd(G*G.t()+Ik);
    W = VVinv * (V.t()*F+Wp+(V.t()*alphaf-alphaw)/rho);
    G = arma::inv_sympd(F.t()*F+Ik) * (F.t()*X+Gp+(F.t()*alphax-alphag)/rho);
    FG = F*G;
    X = (FG*rho+2.0*V-alphax)/(2.0+rho);

    Wp = W + alphaw/rho;
    Wp.transform( [] (double x) {return(max(x, 0.0));} );
    Gp = G + alphag/rho;
    Gp.transform( [] (double x) {return(max(x, 0.0));} );

    alphax += (X-FG)*rho;
    alphaf += (F-V*W)*rho;
    alphaw += (W-Wp)*rho;
    alphag += (G-Gp)*rho;

    // Check convergence
    err1 = arma::accu(arma::pow(V-FG, 2));
    err1 /= sqt;
    Error[iter] = err1;
    // if (fixediter == 0 && abs(err1-err0) < err0*tol)
    if (fixediter == 0 && err1 < tol)
      break;
    err0 = err1;
    if (verbose)
      Rcout << iter << " " << err1 << endl;
  }

  NumericMatrix Wr(wrap(W));
  NumericMatrix Gr(wrap(G));
  return List::create(Named("W") = Wr,
                      Named("G") = Gr,
                      Named("Error") = Error[Error!=-1]);

}


