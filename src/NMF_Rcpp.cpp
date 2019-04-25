#include <RcppArmadillo.h>

#include <string>
#include <vector>

using namespace Rcpp;
using namespace std;

// [[Rcpp::depends(RcppArmadillo)]]

//' Standard NMF implemented in ADMM
//'
//' This function implements a ADMM for standard NMF:
//' V ~ WG^t with V, W & G non-negative.
//'
//' @param V         A (m x n) non-negative matrix
//'                  in the original data matrix.
//' @param k         An integer to specify desired rank of the low rank factorization result
//' @param cluster   A (k x 1) vector of integer 0~(k-1) indicating initial cluster assignment
//' @param dist      A string indicating the distance measure (default: "F2").
//' Currently only "F2", for Euclidean distance and "KL" for KL divergence are implemented
//' @param verbose   A 0/1 integer (default 0). If 1: output the relative error after each iteration
//' @param maxiter   An integer (default 100) for the maximal number of iterations
//' @param fixediter An integer (optional) for a fixed number of iterations
//' @param rho       Penalty parameter in ADMM (default 3)
//' @param tol       Tolerance parameter to declare convergence
//' @return A list containing the following values
//' * W      : A (m x k) non-negative matrix containing W in V ~ WG^t
//' * G      : A (n x k) non-negative matrix containing G in V ~ WG^t
//' * Error  : A vector containing the relative distance of this approximation in each iteration
//' @export
// [[Rcpp::export]]
List NMF_ADMM(arma::mat& V, int k,
                arma::uvec cluster,
                std::string dist = "F2",
                bool verbose = 0,
                int maxiter = 100, int fixediter = 0,
                double rho = 3,
                double tol = 1e-3) {

  int n = (int) V.n_cols; // Dimension of input arma::matrix
  int m = (int) V.n_rows;
  double err0 = 0, err1 = 0;
  double sqt = arma::accu(arma::pow(V, 2));
  int beta = 2;
  if (dist=="KL")
    beta = 1;
  if (fixediter != 0)
    maxiter = fixediter;
  NumericVector Error(maxiter, -1.0);

  // Initialization
  arma::mat Ik = arma::eye<arma::mat>(k,k);
  arma::mat X(m,n), W(m,k,arma::fill::zeros), G(n,k,arma::fill::zeros);
  arma::mat alphax(m,n,arma::fill::ones), alphaw(m,k,arma::fill::randu), alphag(n,k,arma::fill::randu);

  for (int i=0; i<n; ++i)
    G(i, cluster[i]) = 1;

  W = V * G;
  arma::rowvec nk = arma::sum(W,0);
  W.each_row() /= nk;

  arma::mat Wp = W;
  arma::mat Gp = G;

  X = W * G.t();

  // Iteration
  for (int iter = 0; iter < maxiter; ++iter) {

    W = (X*G+Wp+(alphax*G-alphaw)/rho) * inv_sympd(G.t()*G+Ik);
    G = (X.t()*W+Gp + (alphax.t()*W-alphag)/rho) * inv_sympd(W.t()*W+Ik);
    bool flag = 1;
    if (beta == 1) {
      flag = 0;
      arma::mat delta = W*G.t()*rho-alphax-1.0;
      arma::mat deter = arma::pow(delta, 2) + 4.0 * rho * V;
      int ct = arma::accu(deter<0);
      if (ct > 0)
        flag = 1;
      else
        X = (delta + sqrt(deter))/2.0/rho;
    }
    if (flag == 1) {
      X = (W*G.t()*rho+2*V-alphax)/(2.0+rho);
    }

    Wp = W + alphaw/rho;
    Wp.transform( [] (double x) {return(max(x, 0.0));} );
    alphax += (X-W*G.t())*rho;
    alphaw += (W-Wp)*rho;
    alphag += (G-Gp)*rho;
    // Check convergence
    if (beta == 1) {
      arma::mat kl1 = V%log(V/X)-V+X;
      kl1.transform( [] (double x) {return(isnan(x) ? 0.0:x);} );
      err1 = arma::accu(kl1)/sqt;
      Error[iter] = err1;
      if (fixediter == 0 && abs(err1-err0) < err0*tol)
        break;
      err0 = err1;
      if (verbose)
        Rcout << iter << " KL Error: " << err1 << endl;
    }
    if (beta != 1 || isnan(err1)) {
      err1 = arma::accu(arma::pow(V-X, 2))/sqt;
      Error[iter] = err1;
      // if (fixediter == 0 && abs(err1-err0) < err0*tol)
      if (fixediter == 0 && err1 < tol)
        break;
      err0 = err1;
      if (verbose)
        Rcout << iter << " F2 Error: " << err1 << endl;
    }
  }

  NumericMatrix Wr(wrap(W));
  NumericMatrix Gr(wrap(G));
  return List::create(Named("W") = Wr,
                      Named("G") = Gr,
                      Named("Error") = Error[Error!=-1]);

}


