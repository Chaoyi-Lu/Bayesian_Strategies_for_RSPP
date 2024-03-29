#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
################################################
### Simulation Study 1 Strauss Point process ###
################################################
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------

# Noisy M-H algorithm for the Strauss point process with parallel computation
Noisy_E_kth_Ratio <- function(x, beta_p, gamma_p, R, beta_c, gamma_c){ 
  # beta_c, gamma_c: Current beta and gamma state
  # beta_p, gamma_p: Proposed beta and gamma state
  Z_p <- rStrauss(beta = beta_p, gamma = gamma_p, R = R, W = square(1))
  X_p <- cbind(Z_p$x, Z_p$y) # Generate X' from likelihood( |(beta',gamma'))
  return(((beta_c/beta_p)^(Z_p$n))*((gamma_c/gamma_p)^(sum(dist(X_p)<=R))))
}

# Note that we use notation "N" for the number of the auxiliary draws in the Noisy M-H algorithm in the code 
SPP_Parallel_Noisy_MH <- function(Y, beta0, gamma0, eps_beta, eps_gamma, R, K, T){ 
  # beta0: initial beta, gamma0: initial gamma, 
  # Y: N X 2 observed data, the ith row is the ith position vector
  # eps_beta: Proposal epsilon of beta
  # eps_gamma: Proposal epsilon of gamma
  # R: Radius in S_R function of the SPP
  # T: total iterations
  # N: Number of auxiliary draws for Noisy M-H algorithm
  
  # Initialize parameter
  beta_list <- c(beta0)
  gamma_list <- c(gamma0)
  N_Y <- nrow(Y) # Number of vectors in Y
  s_R_Y <- sum(dist(Y)<=R) # calculate s_R(Y)
  acceptance <- 0 # monitor the acceptance rate
  
  # # Set how many cores to parallel
  # cl <- parallel::makeCluster(detectCores()[1]-1)
  # clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel running
  for (t in 1:T){
    if ((t%%1000) == 0){
      print(t) # monitor the implementation iteration
    }
    # Propose the next states
    beta_p_lb <- max(50,beta_list[t]-eps_beta) # beta proposal lower bound
    beta_p_ub <- min(400,beta_list[t]+eps_beta) # beta proposal upper bound
    gamma_p_lb <- max(0,gamma_list[t]-eps_gamma) # gamma proposal lower bound
    gamma_p_ub <- min(1,gamma_list[t]+eps_gamma) # gamma proposal upper bound
    beta_p <- runif(1, beta_p_lb, beta_p_ub)
    gamma_p <- runif(1, gamma_p_lb, gamma_p_ub)
    
    # Calculate the unbiased importance sampling estimator of the normalizing constant ratio
    noisy_E <- mean(parSapply(cl, 1:K, Noisy_E_kth_Ratio, beta_p = beta_p, gamma_p = gamma_p, R = R, beta_c = beta_list[t], gamma_c = gamma_list[t]))
    # Calculate the log acceptance ratio alpha_NMH
    log_alpha_right <-  N_Y*(log(beta_p)-log(beta_list[t])) + s_R_Y*(log(gamma_p)-log(gamma_list[t])) + log(noisy_E) + 
      log(beta_p_ub-beta_p_lb) + log(gamma_p_ub-gamma_p_lb) - 
      log(min(400,beta_p+eps_beta)-max(50,beta_p-eps_beta)) - log(min(1,gamma_p+eps_gamma)-max(0,gamma_p-eps_gamma))
    if (log(runif(1)) <= min(0, log_alpha_right)){
      acceptance <- acceptance + 1
      beta_list[t+1] <- beta_p
      gamma_list[t+1] <- gamma_p
    }else{
      beta_list[t+1] <- beta_list[t]
      gamma_list[t+1] <- gamma_list[t]
    }
  }
  # stopCluster(cl)
  return(list(beta = beta_list, gamma = gamma_list, AcceptanceRate = acceptance/T))
}

#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
# Fearnhead & Prangle ABC-MCMC main algorithm for Strauss point process

F.P.ABC.MCMC.Strauss <- function(Y, beta0, gamma0, eps_beta, eps_gamma, lmCoefBeta, lmCoefGamma, Pilot.VarBeta, Pilot.VarGamma, eps, R, T){
  # Y: Observation in point pattern, i.e. ppp()
  # beta0: initial beta
  # gamma0: initial gamma
  # eps_beta: Proposal epsilon of beta
  # eps_gamma: Proposal epsilon of gamma
  # lmCoefBeta: linear regression coefficients for beta, i.e. logbeta = a_beta + b_beta1*eta_1 + b_beta2*eta_2
  # lmCoefGamma: linear regression coefficients for gamma, i.e. loggamma = a_gamma + b_gamma1*eta_1 + b_gamma2*eta_2
  # Pilot.VarBeta: variance of pilot estimated logbeta
  # Pilot.VarGamma: variance of pilot estimated loggamma
  # eps: epsilon at p* percentile of Pilot.psi
  # R: estimated radius of SPP, R_hat used for SPP
  
  N_Y <- Y$n
  Kfunc_obs=as.function(Kest(Y, correction="isotropic"))
  Kfunc_obs_R <- Kfunc_obs(R)
  beta_list <- c(beta0)
  gamma_list <- c(gamma0)
  acceptance <- 0
  
  for (t in 1:T){
    if ((t%%1000) == 0){
      print(t)
    }
    #Define proposal bounds
    beta_p_lb <- max(50,beta_list[t]-eps_beta) # beta proposal lower bound
    beta_p_ub <- min(400,beta_list[t]+eps_beta) # beta proposal upper bound
    gamma_p_lb <- max(0,gamma_list[t]-eps_gamma) # gamma proposal lower bound
    gamma_p_ub <- min(1,gamma_list[t]+eps_gamma) # gamma proposal upper bound
    beta_p <- runif(1, beta_p_lb, beta_p_ub)
    gamma_p <- runif(1, gamma_p_lb, gamma_p_ub)
    psi_p <- parSapply(cl, 1, F.P.ABC.Strauss.fair.drawing,beta_p=beta_p,gamma_p=gamma_p,R=R,N_Y=N_Y,Kfunc_obs_R=Kfunc_obs_R,
                       lmCoefBeta=lmCoefBeta,lmCoefGamma=lmCoefGamma,Pilot.VarBeta=Pilot.VarBeta,Pilot.VarGamma=Pilot.VarGamma)
    
    if (psi_p<=eps){
      # Determine alpha ratio which is the ratio of proposal because priors are uniform distributions
      log_alpha_right <- log(beta_p_ub-beta_p_lb) + log(gamma_p_ub-gamma_p_lb) - 
        log(min(400,beta_p+eps_beta)-max(50,beta_p-eps_beta)) - log(min(1,gamma_p+eps_gamma)-max(0,gamma_p-eps_gamma))
      if (log(runif(1)) <= min(0, log_alpha_right)){
        acceptance <- acceptance + 1
        beta_list[t+1] <- beta_p # update theta list
        gamma_list[t+1] <- gamma_p
      }else{
        beta_list[t+1] <- beta_list[t] # update theta list
        gamma_list[t+1] <- gamma_list[t]
      }
    }else{
      beta_list[t+1] <- beta_list[t] # update theta list
      gamma_list[t+1] <- gamma_list[t]
    }
  }
  return(list(beta = beta_list, gamma = gamma_list, AcceptanceRate = acceptance/T))
}

F.P.ABC.Strauss.fair.drawing <- function(x,R,beta_p,gamma_p,N_Y,Kfunc_obs_R,lmCoefBeta,lmCoefGamma,Pilot.VarBeta,Pilot.VarGamma){ # Current state beta and gamma
  X_p <- rStrauss(beta_p,gamma_p,R,square(1))
  Kfunc_X_p <- as.function(Kest(X_p, correction="isotropic"))
  eta_p <- c(log(X_p$n)-log(N_Y),(sqrt(Kfunc_X_p(R))-sqrt(Kfunc_obs_R))^2)
  psi_p <- (lmCoefBeta[2:3]%*%eta_p)^2/Pilot.VarBeta + (lmCoefGamma[2:3]%*%eta_p)^2/Pilot.VarGamma
  return(psi_p)
}

#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
#############################################################################
### Simulation Study 2 Determinantal Point Process with a Gaussian Kernel ###
#############################################################################
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------

# Calculate the log density without normalizing constant
dppG_logDensity <- function(X,eigen){
  # e.g. eigen = dppeigen(dppGauss(lambda=tau_p, alpha=sigma_p, d=2),trunc=0.99, Wscale = c(1,1))
  N <- nrow(X)
  res.matrix <- matrix(NA,N,N)
  eigen.eig <- eigen$eig # Extract the eigenvalues
  for (i in 1:N){
    for (j in i:N){
      res.matrix[i,j] <- res.matrix[j,i] <-
        colSums(Re(fourierbasisraw(matrix(X[i,]-X[j,],1), eigen$index, boxlengths = c(1,1)))*eigen.eig/(1-eigen.eig)) # Evaluate each \tilde{\hat{C}}(x_i,x_j)
    }
  } 
  return(logdet(res.matrix))
}

# Metropolis-Hasting algorithm for dppG
dppG_MH <- function(Y, tau0, sigma0, eps_tau, eps_sigma, T){ # tau0: initial tau, sigma0: initial sigma, 
  # Y: N X 2 observed data, the ith row is the ith point
  # tau0: initial tau
  # sigma0: initial sigma, 
  # eps_tau: Proposal epsilon of tau
  # eps_sigma: Proposal epsilon of sigma
  # T: total iterations
  # N: Number of auxiliary draws
  
  # Initialize parameter
  tau_list <- c(tau0)
  sigma_list <- c(sigma0)
  N_Y <- nrow(Y) # Number of vectors in Y
  acceptance <- 0
  
  # # Set how many cores to parallel
  # cl <- parallel::makeCluster(detectCores()[1]-1)
  # clusterExport(cl=cl, list("simulate","dppGauss","Kest","logdet")) # In order to use this function for parallel running
  for (t in 1:T){
    if ((t%%1000) == 0){
      print(t)
    }
    # Propose the next states
    tau_p_lb <- max(50,tau_list[t]-eps_tau) # tau proposal lower bound
    tau_p_ub <- min(200,tau_list[t]+eps_tau) # tau proposal upper bound
    tau_p <- runif(1, tau_p_lb, tau_p_ub)
    sigma_p_lb <- max(0.001,sigma_list[t]-eps_sigma) # sigma proposal lower bound
    sigma_p_ub <- min(1/sqrt(pi*tau_p),sigma_list[t]+eps_sigma) # sigma proposal upper bound
    sigma_p <- runif(1, sigma_p_lb, sigma_p_ub)
    dppGauss.eigen_p <- dppeigen(dppGauss(lambda=tau_p, alpha=sigma_p, d=2),trunc=0.99, Wscale = c(1,1))
    dppGauss.eigen_c <- dppeigen(dppGauss(lambda=tau_list[t], alpha=sigma_list[t], d=2),trunc=0.99, Wscale = c(1,1))
    
    log_alpha_right <-  dppG_logDensity(Y,dppGauss.eigen_p)-sum(log(dppGauss.eigen_p$eig/(1-dppGauss.eigen_p$eig)+1))-
      dppG_logDensity(Y,dppGauss.eigen_c)+sum(log(dppGauss.eigen_c$eig/(1-dppGauss.eigen_c$eig)+1))+
      log(tau_p_ub-tau_p_lb) + log(sigma_p_ub-sigma_p_lb) - 
      log(min(200,tau_p+eps_tau)-max(50,tau_p-eps_tau)) - log(min(1/sqrt(pi*tau_list[t]),sigma_p+eps_sigma)-max(0.001,sigma_p-eps_sigma))
    if (log(runif(1)) <= min(0, log_alpha_right)){
      acceptance <- acceptance + 1
      tau_list[t+1] <- tau_p # update theta list
      sigma_list[t+1] <- sigma_p
    }else{
      tau_list[t+1] <- tau_list[t] # update theta list
      sigma_list[t+1] <- sigma_list[t]
    }
  }
  # stopCluster(cl)
  return(list(tau = tau_list, sigma = sigma_list, AcceptanceRate = acceptance/T))
}

#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------

# Noisy MH algorithm for determinantal point process Gaussian model with parallel computation
dppG_Noisy_E_kth_Ratio <- function(x, tau_p, sigma_p,dppGauss.eigen_p,dppGauss.eigen_c){ # Current state tau and sigma
  Z_p <- simulate(dppGauss(lambda=tau_p, alpha=sigma_p, d=2))
  X_p <- cbind(Z_p$x, Z_p$y) # Generate X' from likelihood( |(tau',sigma'))
  return(exp(dppG_logDensity(X_p,dppGauss.eigen_c)-dppG_logDensity(X_p,dppGauss.eigen_p)))
}

dppG_Parallel_Noisy_MH <- function(Y, tau0, sigma0, eps_tau, eps_sigma, K, T){ # tau0: initial tau, sigma0: initial sigma, 
  # Y: N X 2 observed data, the ith row is the ith point
  # tau0: initial tau
  # sigma0: initial sigma, 
  # eps_tau: Proposal epsilon of tau
  # eps_sigma: Proposal epsilon of sigma
  # T: total iterations
  # N: Number of auxiliary draws
  
  # Initialize parameter
  tau_list <- c(tau0)
  sigma_list <- c(sigma0)
  N_Y <- nrow(Y) # Number of vectors in Y
  acceptance <- 0
  require(spatstat)
  
  # # Set how many cores to parallel
  # cl <- parallel::makeCluster(detectCores()[1]-1)
  # clusterExport(cl=cl, list("simulate","dppGauss","Kest","logdet")) # In order to use this function for parallel running
  for (t in 1:T){
    if ((t%%1000) == 0){
      print(t)
    }
    # Propose the next states
    tau_p_lb <- max(50,tau_list[t]-eps_tau) # tau proposal lower bound
    tau_p_ub <- min(200,tau_list[t]+eps_tau) # tau proposal upper bound
    tau_p <- runif(1, tau_p_lb, tau_p_ub)
    sigma_p_lb <- max(0.001,sigma_list[t]-eps_sigma) # sigma proposal lower bound
    sigma_p_ub <- min(1/sqrt(pi*tau_p),sigma_list[t]+eps_sigma) # sigma proposal upper bound
    sigma_p <- runif(1, sigma_p_lb, sigma_p_ub)
    dppGauss.eigen_p <- dppeigen(dppGauss(lambda=tau_p, alpha=sigma_p, d=2),trunc=0.99, Wscale = c(1,1)) # Obtain the eigenvalues and eigenfunctions by the inbuild function dppeigen() from the spatstat package 
    dppGauss.eigen_c <- dppeigen(dppGauss(lambda=tau_list[t], alpha=sigma_list[t], d=2),trunc=0.99, Wscale = c(1,1))
    
    # Calculate the unbiased importance sampling estimator of the normalizing constant ratio
    noisy_E <- mean(parSapply(cl, 1:K, dppG_Noisy_E_kth_Ratio, tau_p=tau_p, sigma_p=sigma_p,
                              dppGauss.eigen_p=dppGauss.eigen_p,dppGauss.eigen_c=dppGauss.eigen_c))
    # Calculate the log acceptance ratio alpha_DPPG
    log_alpha_right <-  log(noisy_E)+dppG_logDensity(Y,dppGauss.eigen_p)-dppG_logDensity(Y,dppGauss.eigen_c) + 
      log(tau_p_ub-tau_p_lb) + log(sigma_p_ub-sigma_p_lb) - 
      log(min(200,tau_p+eps_tau)-max(50,tau_p-eps_tau)) - log(min(1/sqrt(pi*tau_list[t]),sigma_p+eps_sigma)-max(0.001,sigma_p-eps_sigma))
    if (log(runif(1)) <= min(0, log_alpha_right)){
      acceptance <- acceptance + 1
      tau_list[t+1] <- tau_p # update theta list
      sigma_list[t+1] <- sigma_p
    }else{
      tau_list[t+1] <- tau_list[t] # update theta list
      sigma_list[t+1] <- sigma_list[t]
    }
  }
  # stopCluster(cl)
  return(list(tau = tau_list, sigma = sigma_list, AcceptanceRate = acceptance/T))
}

#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------

# Approximate Noisy M-H algorithm for determinantal point process Gaussian model with parallel computation
Approx_dppG_Noisy_E_kth_Ratio <- function(x, tau_p, sigma_p,tau_c, sigma_c){ # Current state tau and sigma
  Z_p <- simulate(dppGauss(lambda=tau_p, alpha=sigma_p, d=2))
  X_p <- cbind(Z_p$x, Z_p$y) # Generate X' from likelihood( |(tau',sigma'))
  X_p.dist <- as.matrix(dist(X_p,diag=TRUE,upper=TRUE))
  return(exp(logdet(tau_c*exp(-(X_p.dist/sigma_c)^2))-logdet(tau_p*exp(-(X_p.dist/sigma_p)^2))))
}

Approx_dppG_Parallel_Noisy_MH <- function(Y, tau0, sigma0, eps_tau, eps_sigma, K, T){
  # Y: N X 2 observed data, the ith row is the ith point
  # tau0: initial tau
  # sigma0: initial sigma, 
  # eps_tau: Proposal epsilon of tau
  # eps_sigma: Proposal epsilon of sigma
  # T: total iterations
  # N: Number of auxiliary draws
  
  # Initialize parameter
  tau_list <- c(tau0)
  sigma_list <- c(sigma0)
  N_Y <- nrow(Y) # Number of vectors in Y
  dist.Y <- as.matrix(dist(Y,diag=TRUE,upper=TRUE)) # distance matrix of Y
  acceptance <- 0
  
  # # Set how many cores to parallel
  # cl <- parallel::makeCluster(detectCores()[1]-1)
  # clusterExport(cl=cl, list("simulate","dppGauss","Kest","logdet")) # In order to use this function for parallel running
  for (t in 1:T){
    if ((t%%1000) == 0){
      print(t)
    }
    # Propose the next states
    tau_p_lb <- max(50,tau_list[t]-eps_tau) # tau proposal lower bound
    tau_p_ub <- min(200,tau_list[t]+eps_tau) # tau proposal upper bound
    tau_p <- runif(1, tau_p_lb, tau_p_ub)
    sigma_p_lb <- max(0.001,sigma_list[t]-eps_sigma) # sigma proposal lower bound
    sigma_p_ub <- min(1/sqrt(pi*tau_p),sigma_list[t]+eps_sigma) # sigma proposal upper bound
    sigma_p <- runif(1, sigma_p_lb, sigma_p_ub)
    
    # Apply the approximate normalized likelihood to calculate the unbiased importance sampling estimator of the normalizing constant ratio
    noisy_E <- mean(parSapply(cl, 1:K, Approx_dppG_Noisy_E_kth_Ratio, tau_p=tau_p, sigma_p=sigma_p,tau_c=tau_list[t],sigma_c=sigma_list[t]))
    # Calculate the log approximate acceptance ratio \tilde{alpha}_DPPG
    log_alpha_right <-  log(noisy_E) + logdet(tau_p*exp(-(dist.Y/sigma_p)^2)) - logdet(tau_list[t]*exp(-(dist.Y/sigma_list[t])^2))
    log(tau_p_ub-tau_p_lb) + log(sigma_p_ub-sigma_p_lb) - 
      log(min(200,tau_p+eps_tau)-max(50,tau_p-eps_tau)) - log(min(1/sqrt(pi*tau_list[t]),sigma_p+eps_sigma)-max(0.001,sigma_p-eps_sigma))
    if (log(runif(1)) <= min(0, log_alpha_right)){
      acceptance <- acceptance + 1
      tau_list[t+1] <- tau_p # update theta list
      sigma_list[t+1] <- sigma_p
    }else{
      tau_list[t+1] <- tau_list[t] # update theta list
      sigma_list[t+1] <- sigma_list[t]
    }
  }
  # stopCluster(cl)
  return(list(tau = tau_list, sigma = sigma_list, AcceptanceRate = acceptance/T))
}

#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------

# Fearnhead & Prangle ABC-MCMC main algorithm for determinantal point process Gaussian model

F.P.ABC.MCMC.dppG <- function(Y, tau0, sigma0, eps_tau, eps_sigma, lmCoefTau, lmCoefSigma, Pilot.VarTau, Pilot.VarSigma, eps, r_M, T){
  # Y: Observation in point pattern, i.e. ppp()
  # tau0: initial tau
  # sigma0: initial sigma
  # eps_tau: Proposal epsilon of tau
  # eps_sigma: Proposal epsilon of sigma
  # lmCoefTau: linear regression coefficients for tau, i.e. logtau = a_tau + b_tau1*eta_1 + b_tau2r1*eta_2r1+b_tau2r2*eta_2r2+...+b_tau2rM*eta_2rM
  # lmCoefSIgma: linear regression coefficients for sigma, i.e. logsigma = a_sigma + b_sigma1*eta_1 + b_sigma2r1*eta_2r1+...+b_sigma2rM*eta_2rM
  # Pilot.VarTau: variance of pilot estimated logtau
  # Pilot.VarSigma: variance of pilot estimated logsigma
  # eps: epsilon at p* percentile of Pilot.psi
  # r_M: M equally spaced r's for sufficient statistic
  
  N_Y <- Y$n
  Kfunc_obs=as.function(Kest(Y, correction="isotropic"))
  Kfunc_obs_rM <- Kfunc_obs(r_M)
  tau_list <- c(tau0)
  sigma_list <- c(sigma0)
  acceptance <- 0
  
  for (t in 1:T){
    if ((t%%100) == 0){
      print(t)
    }
    #Define proposal bounds
    tau_p_lb <- max(50,tau_list[t]-eps_tau) # tau proposal lower bound
    tau_p_ub <- min(200,tau_list[t]+eps_tau) # tau proposal upper bound
    tau_p <- runif(1, tau_p_lb, tau_p_ub)
    sigma_p_lb <- max(0.001,sigma_list[t]-eps_sigma) # sigma proposal lower bound
    sigma_p_ub <- min(1/sqrt(pi*tau_p),sigma_list[t]+eps_sigma) # sigma proposal upper bound
    sigma_p <- runif(1, sigma_p_lb, sigma_p_ub)
    psi_p <- parSapply(cl, 1, F.P.ABC.dppG.fair.drawing,tau_p=tau_p,sigma_p=sigma_p,r_M=r_M,N_Y=N_Y,Kfunc_obs_rM=Kfunc_obs_rM,
                       lmCoefTau=lmCoefTau,lmCoefSigma=lmCoefSigma,Pilot.VarTau=Pilot.VarTau,Pilot.VarSigma=Pilot.VarSigma)
    
    if (psi_p<=eps){
      # Determine alpha ratio which is the ratio of proposal because priors are uniform distributions
      log_alpha_right <- log(tau_p_ub-tau_p_lb) + log(sigma_p_ub-sigma_p_lb) - 
        log(min(200,tau_p+eps_tau)-max(50,tau_p-eps_tau)) - log(min(1/sqrt(pi*tau_list[t]),sigma_p+eps_sigma)-max(0.001,sigma_p-eps_sigma))
      if (log(runif(1)) <= min(0, log_alpha_right)){
        acceptance <- acceptance + 1
        tau_list[t+1] <- tau_p # update theta list
        sigma_list[t+1] <- sigma_p
      }else{
        tau_list[t+1] <- tau_list[t] # update theta list
        sigma_list[t+1] <- sigma_list[t]
      }
    }else{
      tau_list[t+1] <- tau_list[t] # update theta list
      sigma_list[t+1] <- sigma_list[t]
    }
  }
  return(list(tau = tau_list, sigma = sigma_list, AcceptanceRate = acceptance/T))
}

F.P.ABC.dppG.fair.drawing <- function(x,tau_p,sigma_p,r_M,N_Y,Kfunc_obs_rM,lmCoefTau,lmCoefSigma,Pilot.VarTau,Pilot.VarSigma){ # Current state tau and sigma
  X_p <- simulate(dppGauss(lambda=tau_p, alpha=sigma_p, d=2))
  Kfunc_X_p <- as.function(Kest(X_p, correction="isotropic"))
  eta_p <- c(log(X_p$n)-log(N_Y),(sqrt(Kfunc_X_p(r_M))-sqrt(Kfunc_obs_rM))^2)
  psi_p <- (lmCoefTau[2:12]%*%eta_p)^2/Pilot.VarTau + (lmCoefSigma[2:12]%*%eta_p)^2/Pilot.VarSigma
  return(psi_p)
}

#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
###################################################
### Real Data Application Strauss point process ###
###################################################
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
# Noisy MH algorithm for Strauss point process with parallel computation
# The functions below are designed for real duke forest data application
# The only difference compared to simulation study 1 functions is the prior settings

Noisy_E_kth_Ratio <- function(x, beta_p, gamma_p, R, beta_c, gamma_c){ # Current state beta and gamma
  Z_p <- rStrauss(beta = beta_p, gamma = gamma_p, R = R, W = square(1))
  X_p <- cbind(Z_p$x, Z_p$y) # Generate X' from likelihood( |(beta',gamma'))
  return(((beta_c/beta_p)^(Z_p$n))*((gamma_c/gamma_p)^(sum(dist(X_p)<=R))))
}

df_SPP_Parallel_Noisy_MH <- function(Y, beta0, gamma0, eps_beta, eps_gamma, R, K, T){ # beta0: initial beta, gamma0: initial gamma, 
  # Y: N X 2 observed data, the ith row is the ith point
  # beta0: initial beta
  # gamma0: initial gamma, 
  # eps_beta: Proposal epsilon of beta
  # eps_gamma: Proposal epsilon of gamma
  # R: Radius in S_R function of SPP
  # T: total iterations
  # N: Number of auxiliary draws
  
  # Initialize parameter
  beta_list <- c(beta0)
  gamma_list <- c(gamma0)
  N_Y <- nrow(Y) # Number of vectors in Y
  s_R_Y <- sum(dist(Y)<=R) # calculate s_R(Y)
  acceptance <- 0
  
  # # Set how many cores to parallel
  # cl <- parallel::makeCluster(detectCores()[1]-1)
  # clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel running
  for (t in 1:T){
    if ((t%%1000) == 0){
      print(t)
    }
    # Propose the next states
    beta_p_lb <- max(50,beta_list[t]-eps_beta) # beta proposal lower bound
    beta_p_ub <- min(350,beta_list[t]+eps_beta) # beta proposal upper bound
    gamma_p_lb <- max(0,gamma_list[t]-eps_gamma) # gamma proposal lower bound
    gamma_p_ub <- min(1,gamma_list[t]+eps_gamma) # gamma proposal upper bound
    beta_p <- runif(1, beta_p_lb, beta_p_ub)
    gamma_p <- runif(1, gamma_p_lb, gamma_p_ub)
    
    # Find the noisy exchange sum for K
    noisy_E <- mean(parSapply(cl, 1:K, Noisy_E_kth_Ratio, beta_p = beta_p, gamma_p = gamma_p, R = R, beta_c = beta_list[t], gamma_c = gamma_list[t]))
    log_alpha_right <-  N_Y*(log(beta_p)-log(beta_list[t])) + s_R_Y*(log(gamma_p)-log(gamma_list[t])) + log(noisy_E) + 
      log(beta_p_ub-beta_p_lb) + log(gamma_p_ub-gamma_p_lb) - 
      log(min(350,beta_p+eps_beta)-max(50,beta_p-eps_beta)) - log(min(1,gamma_p+eps_gamma)-max(0,gamma_p-eps_gamma))
    if (log(runif(1)) <= min(0, log_alpha_right)){
      acceptance <- acceptance + 1
      beta_list[t+1] <- beta_p # update theta list
      gamma_list[t+1] <- gamma_p
    }else{
      beta_list[t+1] <- beta_list[t] # update theta list
      gamma_list[t+1] <- gamma_list[t]
    }
  }
  # stopCluster(cl)
  return(list(beta = beta_list, gamma = gamma_list, AcceptanceRate = acceptance/T))
}

#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------

# Fearnhead & Prangle ABC-MCMC main algorithm for Strauss point process
# The functions below are designed for real data application
# The only difference compared to SS1 is the prior range

df.F.P.ABC.MCMC.Strauss <- function(Y, beta0, gamma0, eps_beta, eps_gamma, lmCoefBeta, lmCoefGamma, Pilot.VarBeta, Pilot.VarGamma, eps, R, T){
  # Y: Observation in point pattern, i.e. ppp()
  # beta0: initial beta
  # gamma0: initial gamma
  # eps_beta: Proposal epsilon of beta
  # eps_gamma: Proposal epsilon of gamma
  # lmCoefBeta: linear regression coefficients for beta, i.e. logbeta = a_beta + b_beta1*eta_1 + b_beta2*eta_2
  # lmCoefGamma: linear regression coefficients for gamma, i.e. loggamma = a_gamma + b_gamma1*eta_1 + b_gamma2*eta_2
  # Pilot.VarBeta: variance of pilot estimated logbeta
  # Pilot.VarGamma: variance of pilot estimated loggamma
  # eps: epsilon at p* percentile of Pilot.psi
  # R: estimated radius of SPP, R_hat used for SPP
  
  N_Y <- Y$n
  Kfunc_obs=as.function(Kest(Y, correction="isotropic"))
  Kfunc_obs_R <- Kfunc_obs(R)
  beta_list <- c(beta0)
  gamma_list <- c(gamma0)
  acceptance <- 0
  
  for (t in 1:T){
    if ((t%%1000) == 0){
      print(t)
    }
    #Define proposal bounds
    beta_p_lb <- max(50,beta_list[t]-eps_beta) # beta proposal lower bound
    beta_p_ub <- min(350,beta_list[t]+eps_beta) # beta proposal upper bound
    gamma_p_lb <- max(0,gamma_list[t]-eps_gamma) # gamma proposal lower bound
    gamma_p_ub <- min(1,gamma_list[t]+eps_gamma) # gamma proposal upper bound
    beta_p <- runif(1, beta_p_lb, beta_p_ub)
    gamma_p <- runif(1, gamma_p_lb, gamma_p_ub)
    psi_p <- parSapply(cl, 1, F.P.ABC.Strauss.fair.drawing,beta_p=beta_p,gamma_p=gamma_p,R=R,N_Y=N_Y,Kfunc_obs_R=Kfunc_obs_R,
                       lmCoefBeta=lmCoefBeta,lmCoefGamma=lmCoefGamma,Pilot.VarBeta=Pilot.VarBeta,Pilot.VarGamma=Pilot.VarGamma)
    
    if (psi_p<=eps){
      # Determine alpha ratio which is the ratio of proposal because priors are uniform distributions
      log_alpha_right <- log(beta_p_ub-beta_p_lb) + log(gamma_p_ub-gamma_p_lb) - 
        log(min(350,beta_p+eps_beta)-max(50,beta_p-eps_beta)) - log(min(1,gamma_p+eps_gamma)-max(0,gamma_p-eps_gamma))
      if (log(runif(1)) <= min(0, log_alpha_right)){
        acceptance <- acceptance + 1
        beta_list[t+1] <- beta_p # update theta list
        gamma_list[t+1] <- gamma_p
      }else{
        beta_list[t+1] <- beta_list[t] # update theta list
        gamma_list[t+1] <- gamma_list[t]
      }
    }else{
      beta_list[t+1] <- beta_list[t] # update theta list
      gamma_list[t+1] <- gamma_list[t]
    }
  }
  return(list(beta = beta_list, gamma = gamma_list, AcceptanceRate = acceptance/T))
}
