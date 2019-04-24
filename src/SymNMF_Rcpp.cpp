#include <RcppArmadillo.h>

#include <string>
#include <vector>

using namespace Rcpp;
using namespace std;

// [[Rcpp::depends(RcppArmadillo)]]

//' Joint symmetric NMF for multiple input matricies using ADMM
//'
//' This function implements a ADMM for NMF taking two n x n matrices.
//' (One of them presumably measures similarity and symmetric)
//' V := Vp - Vm ~ W1G^t + W2G^t with Vp, Vm, W1, W2 & G non-negative.
//'
//' @param Vp        A (n x n) non-negative matrix measuring similariy
//' @param Vm        A (n x n) non-negative matrix containing the abs value of the negative part
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
//' * W1     : A (n x k) non-negative matrix containing W1 in Vp ~ W1G^t
//' * W2     : A (n x k) non-negative matrix containing W2 in Vm ~ W1G^t
//' * G      : A (n x k) non-negative matrix containing G in V ~ W1G^t + W2G^t
//' * Error  : A vector containing the relative distance of this approximation in each iteration
//' @export
// [[Rcpp::export]]
List SymNMFs_ADMM(arma::mat& Vp,
                 arma::mat& Vm,
                 int k,
                 arma::uvec cluster,
                 std::string dist = "F2",
                 bool verbose = 0,
                 int maxiter = 100, int fixediter = 0,
                 double rho = 3,
                 double tol = 1e-3) {

  int n = (int) Vp.n_cols; // Dimension of input arma::matrix
  double err0 = 0, err1 = 0;
  double sqt = arma::accu(arma::pow(Vp, 2)) + arma::accu(arma::pow(Vm, 2));
  int beta = 2;
  if (dist=="KL")
    beta = 1;
  if (fixediter != 0)
    maxiter = fixediter;
  NumericVector Error(maxiter, -1.0);

  // Initialization
  arma::mat Ik = arma::eye<arma::mat>(k,k);
  arma::mat X(n,n), Y(n,n), W1(n,k,arma::fill::zeros), W2(n,k,arma::fill::zeros), G(n,k,arma::fill::zeros);
  arma::mat alphax(n,n,arma::fill::ones), alphay(n,n,arma::fill::ones);
  arma::mat alphaw1(n,k,arma::fill::randu), alphaw2(n,k,arma::fill::randu), alphag(n,k,arma::fill::randu);

  for (int i=0; i<n; ++i)
    G(i, cluster[i]) = 1;

  W1 = G;
  arma::rowvec nk = arma::sum(W1,0);
  W1.each_row() /= nk;
  W1 = Vp * W1;
  W2 = Vm * W1;

  arma::mat W1p = W1;
  arma::mat W2p = W2;

  X = W1 * G.t();
  Y = W2 * G.t();

  // Iteration
  for (int iter = 0; iter < maxiter; ++iter) {

    W1 = (X*G+G+W1p+(alphax*G-alphaw1-alphag)/rho) * inv_sympd(G.t()*G+2.0*Ik);
    W2 = (Y*G  +W2p+(alphay*G-alphaw2       )/rho) * inv_sympd(G.t()*G+    Ik);
    G = (X.t()*W1+Y.t()*W2+W1 + (alphax.t()*W1+alphay.t()*W2+alphag)/rho) * inv_sympd(W1.t()*W1+W2.t()*W2+Ik);

    bool flag = 1;
    if (beta == 1) {
      flag = 0;
      arma::mat delta = W1*G.t()*rho-alphax-1.0;
      arma::mat deter = arma::pow(delta, 2) + 4.0 * rho * Vp;
      int ct = arma::accu(deter<0);
      if (ct > 0)
        flag = 1;
      else
        X = (delta + sqrt(deter))/2.0/rho;
      delta = W2*G.t()*rho-alphay-1.0;
      deter = arma::pow(delta, 2) + 4.0 * rho * Vm;
      ct = arma::accu(deter<0);
      if (ct > 0)
        flag = 1;
      else
        Y = (delta + sqrt(deter))/2.0/rho;
    }
    if (flag == 1) {
      X = (W1*G.t()*rho+2*Vp-alphax)/(2.0+rho);
      Y = (W2*G.t()*rho+2*Vm-alphay)/(2.0+rho);
    }

    W1p = W1 + alphaw1/rho;
    W1p.transform( [] (double x) {return(max(x, 0.0));} );
    W2p = W2 + alphaw2/rho;
    W2p.transform( [] (double x) {return(max(x, 0.0));} );
    alphax += (X-W1*G.t())*rho;
    alphay += (Y-W2*G.t())*rho;
    alphaw1 += (W1-W1p)*rho;
    alphaw2 += (W2-W2p)*rho;
    alphag += (W1-G)*rho;

    // Check convergence
    if (beta == 1) {
      arma::mat kl1 = Vp%log(Vp/X)-Vp+X;
      kl1.transform( [] (double x) {return(isnan(x) ? 0.0:x);} );
      arma::mat kl2 = Vm%log(Vm/Y)-Vm+Y;
      kl2.transform( [] (double x) {return(isnan(x) ? 0.0:x);} );
      err1 = arma::accu(kl1) + arma::accu(kl2);
      err1 /= sqt;
      Error[iter] = err1;
      if (fixediter == 0 && abs(err1-err0) < err0*tol)
        break;
      err0 = err1;
      if (verbose)
        Rcout << iter << " KL Error: " << err1 << endl;
    }
    if (beta != 1 || isnan(err1)) {
      err1 = arma::accu(arma::pow(Vp-Vm-X+Y, 2));
      err1 /= sqt;
      Error[iter] = err1;
      // if (fixediter == 0 && abs(err1-err0) < err0*tol)
      if (fixediter == 0 && err1 < tol)
        break;
      err0 = err1;
      if (verbose)
        Rcout << iter << " F2 Error: " << err1 << endl;
    }
  }

  NumericMatrix W1r(wrap(W1));
  NumericMatrix W2r(wrap(W2));
  NumericMatrix Gr(wrap(G));
  return List::create(Named("W1") = W1r,
                      Named("W2") = W2r,
                      Named("G") = Gr,
                      Named("Error") = Error[Error!=-1]);

}


//' Symmetric NMF for one non-negative symmetric matrix using ADMM
//'
//' This function implements a ADMM for NMF taking a non-negative square matrix
//' presumably measuring similarity
//' V ~ WG^t with V, W & G non-negative.
//'
//' @param V         A (n x n) non-negative symmetric matrix
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
//' * W      : A (n x k) non-negative matrix containing W in V ~ WG^t
//' * G      : A (n x k) non-negative matrix containing G in V ~ WG^t
//' * Error  : A vector containing the relative distance of this approximation in each iteration
//' @export
// [[Rcpp::export]]
List SymNMF_ADMM(arma::mat& V,
                  int k,
                 arma::uvec cluster,
                 std::string dist = "F2",
                 bool verbose = 0,
                 int maxiter = 100, int fixediter = 0,
                 double rho = 3,
                 double tol = 1e-3) {

  int n = (int) V.n_cols; // Dimension of input arma::matrix
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
  arma::mat X(n,n), W1(n,k,arma::fill::zeros), G(n,k,arma::fill::zeros);
  arma::mat alphax(n,n,arma::fill::ones), alphaw1(n,k,arma::fill::randu);
  arma::mat alphag(n,k,arma::fill::randu), WGt(n,n);

  for (int i=0; i<n; ++i)
    G(i, cluster[i]) = 1;

  W1 = G;
  arma::rowvec nk = arma::sum(W1,0);
  W1.each_row() /= nk;
  W1 = V * W1;
  arma::mat W1p = W1;
  X = W1 * G.t();

  // Iteration
  for (int iter = 0; iter < maxiter; ++iter) {

    W1 = (X*G+G+W1p+(alphax*G-alphaw1-alphag)/rho) * inv_sympd(G.t()*G+Ik);
    G = (X.t()*W1+W1 + (alphax.t()*W1+alphag)/rho) * inv_sympd(W1.t()*W1+Ik);

    WGt = W1*G.t();
    bool flag = 1;
    if (beta == 1) {
      flag = 0;
      arma::mat delta = WGt*rho-alphax-1.0;
      arma::mat deter = arma::pow(delta, 2) + 4.0 * rho * V;
      int ct = arma::accu(deter<0);
      if (ct > 0)
        flag = 1;
      else
        X = (delta + sqrt(deter))/2.0/rho;
    }
    if (flag == 1) {
      X = (WGt*rho+2*V-alphax)/(2.0+rho);
    }

    W1p = W1 + alphaw1/rho;
    W1p.transform( [] (double x) {return(max(x, 0.0));} );
    alphax += (X-WGt)*rho;
    alphaw1 += (W1-W1p)*rho;
    alphag += (W1-G)*rho;

    // Check convergence
    if (beta == 1) {
      arma::mat kl1 = V%log(V/WGt)-V+WGt;
      kl1.transform( [] (double x) {return(isnan(x) ? 0.0:x);} );
      err1 = arma::accu(kl1);
      err1 /= sqt;
      Error[iter] = err1;
      if (fixediter == 0 && err1 < tol)
        break;
      err0 = err1;
      if (verbose)
        Rcout << iter << " KL Error: " << err1 << endl;
    }
    if (beta != 1 || isnan(err1)) {
      err1 = arma::accu(arma::pow(V-WGt, 2));
      err1 /= sqt;
      Error[iter] = err1;
      // if (fixediter == 0 && abs(err1-err0) < err0*tol)
      if (fixediter == 0 && err1 < tol)
        break;
      err0 = err1;
      if (verbose)
        Rcout << iter << " F2 Error: " << err1 << endl;
    }
  }

  NumericMatrix W1r(wrap(W1));
  NumericMatrix Gr(wrap(G));
  return List::create(Named("W") = W1r,
                      Named("G") = Gr,
                      Named("Error") = Error[Error!=-1]);

}


