////////////////////////////////////////////////////////////////////////////////
// This code was based on that provided in the Expokit library. All copyright is 
// reserved, please consult the Expokit package: 
//  Roger B. Sidje (rbs@maths.uq.edu.au)
//  EXPOKIT: Software Package for Computing Matrix Exponentials.
//  ACM - Transactions On Mathematical Software, 24(1):130-156, 1998
////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <cmath>
#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]

using vec = Eigen::VectorXd; 
using mat = Eigen::MatrixXd; 
using spmat = Eigen::SparseMatrix<double>; 
using std::pow; 
using std::log; 
using std::fmax;
using std::fabs; 
using std::exp; 

// Computes exp(At)*v where exp(At) is the matrix exponential. Uses the Krylov
// approximation and time-stepping.
// This code is transcibed from the Fortran Expokit software.
//
// Inputs:
//   A: sparse matrix
//   v: column vector
//   t: scalar constant
//   krylov_dim: dimension of approximating Krylov space
//   tol: tolerance in error of approximation (warning produced if not met)
//
// Output:
//  exp(At)*v

// [[Rcpp::export]]
Eigen::VectorXd sparse_action(const Eigen::SparseMatrix<double, Eigen::RowMajor>& A,
                              const Eigen::VectorXd& v,
                              const double& t,
                              const int& krylov_dim = 30,
                              const double& tol = 1e-10) {
  double m = fmin(A.rows(), krylov_dim);
  double anorm = A.norm(); 
  double mxrej = 10;
  double mx;
  double btol = 1e-7;
  double gamma = 0.9;
  double mb = m;
  int nstep = 0;
  double t_now = 0;
  double t_step;
  double delta = 1.2;
  double t_out = fabs(t);
  double s_error = 0;
  double rndoff = anorm * 1e-16;
  
  int k1 = 1;
  double xm = 1 / m;
  double normv = v.norm();
  double avnorm = 0;
  double beta = normv;
  double t_new = t_out; 
  double s = std::pow(10, std::floor(std::log10(t_new)) - 1);
  t_new = std::ceil(t_new / s) * s;
  double sgn = t > 0 ? 1 : -1;
  int ireject;
  double err_loc;
  double phi1;
  double phi2;
  
  Eigen::VectorXd w = v;
  double hump = normv;
  Eigen::MatrixXd vmat = Eigen::MatrixXd::Zero(A.rows(), m + 1);
  Eigen::MatrixXd hmat = Eigen::MatrixXd::Zero(m + 2, m + 2);
  Eigen::MatrixXd fmat;
  Eigen::VectorXd p;
  while (t_now < t_out) {
    ++nstep;
    t_step = fmin(t_out - t_now, t_new);
    vmat.col(0) = (1 / beta) * w;
    for (int j = 0; j < m; ++j) {
      p = A * vmat.col(j);
      for (int i = fmax(0, j - 1); i <= j; ++i) {
        hmat(i, j) = p.dot(vmat.col(i));
        p -= hmat(i, j) * vmat.col(i);
      }     
      s = p.norm();
      if (s < btol) {
        k1 = 0;
        mb = j;
        t_step = t_out - t_now;
        break;
      }
      hmat(j + 1, j) = s;
      vmat.col(j + 1) = (1 / s) * p;
    }
    if (k1 != 0) {
      hmat(m + 1, m) = 1;
      avnorm = (A * vmat.col(m)).norm();
    }
    ireject = 0;
    while (ireject <= mxrej) {
      mx = mb + k1;
      fmat = (sgn * t_step * hmat.topLeftCorner(mx + 1, mx + 1)).exp();
      if (k1 == 0) {
        err_loc = btol;
        break;
      }
      else {
        phi1 = fabs(beta * fmat(m, 0));
        phi2 = fabs(beta * fmat(m + 1, 0) * avnorm);
        if (phi1 > 10 * phi2) {
          err_loc = phi2;
          xm = 1 / m;
        }
        else if (phi1 > phi2) {
          err_loc = (phi1 * phi2) / (phi1 - phi2);
          xm = 1 / m;
        }
        else {
          err_loc = phi1;
          xm = 1 / (m - 1);
        }
      }
      if (err_loc <= delta * t_step * tol) break;
      else {
        t_step = gamma * t_step * std::pow(t_step * tol / err_loc, xm);
        s = std::pow(10, std::floor(std::log10(t_step)) - 1);
        t_step = std::ceil(t_step / s) * s;
        ++ireject;
      }
    }
    mx = mb + fmax(0, k1 - 1);
    w = vmat.leftCols(mx + 1) * beta * fmat.col(0).topRows(mx + 1);
    beta = w.norm();
    hump = fmax(hump, beta);
    
    t_now = t_now + t_step;
    t_new = gamma * t_step * std::pow(t_step * tol / err_loc, xm);
    s = std::pow(10, std::floor(std::log10(t_new) - 1));
    t_new = std::ceil(t_new / s) * s;
    
    err_loc = fmax(err_loc, rndoff);
    s_error += err_loc;
  }
  hump = hump / normv;
  return w;
}
