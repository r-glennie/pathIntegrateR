# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

sparse_action <- function(A, v, t, krylov_dim = 30L, tol = 1e-10) {
    .Call(`_pathintegrateR_sparse_action`, A, v, t, krylov_dim, tol)
}

