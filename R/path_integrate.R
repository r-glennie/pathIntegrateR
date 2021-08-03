#'  Compute approximate moment generating function with path integral 
#'
#' @param xgrid regular grid of x locations 
#' @param dx spacing of regular x grid 
#' @param dt time-step for integration 
#' @param Q transition rate matrix over xgrid, movement model 
#' @param f integrand function such that f(x) returns a single number for a single location x
#' @param xi function computes expectation of exp(xi * int f(x) dx) so xi is argument of 
#'           moment generating function of int f(x) dx 
#' @param p0 initial probability of being in each location, default is uniform 
#'
#' @return approximation to expectation of exp(xi * int f(x) dx)
#' @export
mgf_path_integrate <- function(xgrid, dx, T, dt, Q, f, xi = 1, p0 = NULL) {
  # initial distribution, default is uniform 
  nx <- length(xgrid)
  if (is.null(p0)) p0 <- rep(1 / nx, nx)
  p <- p0
  val <- 0
  nt <- floor(T/dt)
  for (i in 1:nt) {
    p <- sparse_action(Q, p, t = dt)
    r0 <- f(xgrid, xi = xi) * dt 
    p <- p * exp(r0)
    vsum <- sum(p)
    val <- val + log(vsum)
    p <- p / vsum
  }
  return(exp(val))
}

#' Compute expectation of f(p) over paths p
#'
#' @inheritParams mgf_path_integrate 
#' @param der return expectation of exp(f(p)) (der = 0), f(p) (der = 1, default), 
#'            or f(p)^2 (der = 2)
#'
#' @return approximate value of expectation 
#' @export
path_integrate <- function(xgrid, dx, T, dt, Q, f, p0 = NULL, der = 1) {
  tmp_function <- function(xi) {
    return(mgf_path_integrate(xgrid, dx, T, dt, Q, f, xi = xi, p0 = p0))
  }
  if (der == 0) {
    g <- tmp_function(1)
  }
  else if (der == 1) {
    g <- grad(tmp_function, 0)
  } else if (der == 2) {
    g <- hessian(tmp_function, 0) 
  } else {
    stop("der argument must 0, 1 or 2")
  }
  return(as.numeric(g))
}


