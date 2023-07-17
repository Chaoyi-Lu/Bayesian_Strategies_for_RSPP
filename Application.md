# Simulation Studies and Real Data Application
This file illustrates the code and the corresponding applications associated to the outputs shown in the *Bayesian Strategies for Repulsive Spatial Point Processes* paper.
The code of both two simulation studies and the real data application is provided.

The source function code is included in the [`Algorithm_Functions_for_RSPP.R`] and can be loaded together with some required `R` packages by the following code.

``` r
rm(list=ls())
source("Algorithm_Functions_for_RSPP.R")
library(spatstat) # For point processes simulations and related application
library(doParallel) # For parallel computation
library(LaplacesDemon) # For ESS() and logdet() function
library(latex2exp) # For mathematics symbol in the plot
```

The explanations of the functions in the file [`Algorithm_Functions_for_RSPP.R`] are provided in the corresponding comments in the file.

## Strauss Point Process Simulation Study $1$

The function `Noisy_E_kth_Ratio()` corresponds to the $k$ th auxiliary draw of the noisy Metropolis-Hastings (noisy M-H) algorithm as well as the corresponding evaluation of the unnormalised likelihood ratio $\frac{q(x_k'|\theta^{(t-1)})}{q(x_k'|\theta')}$. 
The function `SPP_Parallel_Noisy_MH()` is the noisy M-H algorithm implemented for the Strauss point process (SPP) in this simulation study 1 (SS1).
The input $K$ is the fixed total number of auxiliary draws set by the practitioners.
Note that, by setting $K=1$, the noisy M-H algorithm becomes the exchange algorithm.
The parallel computation is implemented for the $K$ auxiliary draws.

The function `S.G.ABC.MCMC.Strauss.repeat.draws()` implements one round of proposed draw for the SPP in the `repeat` loop (step $1$ of the Algorithm $5$ in the paper) of the ABC-MCMC algorithm proposed by [Shirota and Gelfand (2017)](https://doi.org/10.1080/10618600.2017.1299627) and returns the proposed states as well as the indicator of whether the corresponding $\Psi(\hat{\theta}',\hat{a})$ is smaller than the acceptance threshold $\epsilon$.
The function `S.G.Parallel.ABC.MCMC.Strauss()` apply the implementation of the ABC-MCMC algorithm for SPP with the approximate parallel computation discussed in section $4$ of the paper.

We start from the process of generating the artificial data used in the SPP simulation study where we set $\beta = 200, \gamma = 0.1$ and $R = 0.05$ on $S = [0,1]\times[0,1]$.

``` r
## Generate from a Strauss point process
Z_observation <- rStrauss(beta = 200, gamma = 0.1, R = 0.05, W = square(1))
# utils::write.table(x=cbind(Z_observation$x,Z_observation$y),file="SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY.csv", sep="," , row.names = FALSE, col.names=FALSE)
```

All the point locations are stored in the file [`SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY.csv`].
The artificial dataset we applied in the paper can be called directly as follows.

``` r
## Simulation study 1 observation
SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY <- as.matrix(read.csv("SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY.csv",header=FALSE))
SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY  <- ppp(SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY[,1],SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY[,2]) # transform to point pattern class
```

Then we can obtain the estimated interaction radius $R$ by the profile pseudo-likelihood method.
It can be checked that the estimation $\hat{R}=0.0508$.

``` r
## profile pseudo-likelihood method, i.e. maximum pseudo-likelihood calculated at r
SS1_SPP_pplmStrauss <- profilepl(data.frame(r=seq(0.01,0.1, by=0.0001)), Strauss, SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY)
SS1_SPP_pplmStrauss$fit
SS1_SPP_R_hat <- SS1_SPP_pplmStrauss$fit$interaction$par$r # Store the estimated R
```

And the code below provides the plot of the point locations and the profile pseudo-likelihood plot shown as Figure $1$ in SPP simulation study of the paper.

``` r
par(mfrow=c(1,2),mai = c(0.55, 0.5, 0.25, 0.05),mgp=c(1.25,0.45,0))
plot(SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY$x,SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY$y,xlab = "",ylab = "",main="",cex.lab = 0.8)

plot(SS1_SPP_pplmStrauss$param[,1],SS1_SPP_pplmStrauss$prof,type = "l",xlab = "R",ylab = "log PL",cex.main=1,cex.lab = 1)
abline(v=SS1_SPP_pplmStrauss$fit$interaction$par$r,col = 2,lty = 2)
par(xpd=TRUE)
text(0.0508,210, TeX(r'($\hat{R}=0.0508$)'), pos = 4,col=2)
par(xpd=FALSE)
par(mfrow=c(1,1),mai = c(1.02, 0.82, 0.82, 0.42),mgp=c(3,1,0))
```

The ground truth implementation is to apply the exchange algorithm for $1,200,000$ iterations as follows.
The initial states are set as $\beta_0=190,\gamma_0=0.2$ and the proposal epsilons are tuned to be $\epsilon_{\beta}=65, \epsilon_{\gamma}=0.16$.

``` r
# Exchange Ground Truth
cl <- parallel::makeCluster(detectCores()[1]-1)
clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel computation
time_start <- Sys.time()
SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T1200000_1 <-
  SPP_Parallel_Noisy_MH(Y=cbind(SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY$x,SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY$y),
                                  beta0=190, gamma0=0.2,eps_beta=65, eps_gamma=0.16, R=SS1_SPP_R_hat, K=1, T=1200000)
time_end <- Sys.time()
SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T1200000_1_time <- time_end-time_start
# stopCluster(cl)
# Time difference of 2.459206 hours # This is the implementation time we show on the paper
```

Here we provide a reference of the time taken by the implementation.
The function above returns a list of $\beta$ chain and a list of $\gamma$ chain as well as the corresponding acceptance rate of the algorithm.
The outputs are stored in `SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T1200000_1` and thus we can apply the summary statistics on those outputs shown below.

``` r
# # Example summary statistics
# Acceptance rate
SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T1200000_1$AcceptanceRate
# Posterior trace plot
plot(SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T1200000_1$beta, type = "l")
plot(SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T1200000_1$gamma, type = "l")
# Posterior density plot
plot(density(SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T1200000_1$beta[200001:1200001]))
plot(density(SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T1200000_1$gamma[200001:1200001]))
# ESS/s
ESS(SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T1200000_1$beta[200001:1200001])/(SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T1200000_1_time[[1]]*3600)
ESS(SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T1200000_1$gamma[200001:1200001])/(SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T1200000_1_time[[1]]*3600)
# Average ESS/s
(ESS(SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T1200000_1$beta[200001:1200001])+ESS(SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T1200000_1$gamma[200001:1200001]))/(2*SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T1200000_1_time[[1]]*3600)
# Posterior mean
mean(SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T1200000_1$beta[200001:1200001])
mean(SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T1200000_1$gamma[200001:1200001])
# Posterior standard deviation
sd(SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T1200000_1$beta[200001:1200001])
sd(SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T1200000_1$gamma[200001:1200001])
```

Note here that the first element of the chain is the initial state $\theta^{(0)}$ and thus we need to drop the first $200,001$ iterations in order for the $200,000$ burn-in.

### The SS1 Implementation of the Exchange and Noisy M-H Algorithms

Similar implementations are applied for the algorithm comparisons of the exchange and noisy M-H algorithms implemented for $120000$ iterations.
The noisy M-H function `SPP_Parallel_Noisy_MH()` is implemented from $K=1$ to $K=8$ where the $K=1$ case is equivalent to the exchange algorithm.

``` r
# # Exchange == Noisy M-H K1 0.12 million iterations
# cl <- parallel::makeCluster(detectCores()[1]-1)
# clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel computation
time_start <- Sys.time()
SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T120000_1 <-
  SPP_Parallel_Noisy_MH(Y=cbind(SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY$x,SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY$y),
                                  beta0=190, gamma0=0.2,eps_beta=65, eps_gamma=0.16, R=SS1_SPP_R_hat, K=1, T=120000)
time_end <- Sys.time()
SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T120000_1_time <- time_end-time_start
# stopCluster(cl)
# # Time difference of 16.09556 mins

# # Noisy Exchange K2
# cl <- parallel::makeCluster(detectCores()[1]-1)
# clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel computation
time_start <- Sys.time()
SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K2_T120000_1 <-
  SPP_Parallel_Noisy_MH(Y=cbind(SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY$x,SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY$y),
                                  beta0=190, gamma0=0.2,eps_beta=65, eps_gamma=0.16, R=SS1_SPP_R_hat, K=2, T=120000)
time_end <- Sys.time()
SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K2_T120000_1_time <- time_end-time_start
# stopCluster(cl)
# # Time difference of 20.47789 mins

# # Noisy Exchange K3
# cl <- parallel::makeCluster(detectCores()[1]-1)
# clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel computation
time_start <- Sys.time()
SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K3_T120000_1 <-
  SPP_Parallel_Noisy_MH(Y=cbind(SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY$x,SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY$y),
                                  beta0=190, gamma0=0.2,eps_beta=65, eps_gamma=0.16, R=SS1_SPP_R_hat, K=3, T=120000)
time_end <- Sys.time()
SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K3_T120000_1_time <- time_end-time_start
# stopCluster(cl)
# # Time difference of 26.43284 mins

# # Noisy Exchange K4
# cl <- parallel::makeCluster(detectCores()[1]-1)
# clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel computation
time_start <- Sys.time()
SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K4_T120000_1 <-
  SPP_Parallel_Noisy_MH(Y=cbind(SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY$x,SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY$y),
                                  beta0=190, gamma0=0.2,eps_beta=65, eps_gamma=0.16, R=SS1_SPP_R_hat, K=4, T=120000)
time_end <- Sys.time()
SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K4_T120000_1_time <- time_end-time_start
# stopCluster(cl)
# # Time difference of 29.71931 mins

# # Noisy Exchange K5
# cl <- parallel::makeCluster(detectCores()[1]-1)
# clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel computation
time_start <- Sys.time()
SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K5_T120000_1 <-
  SPP_Parallel_Noisy_MH(Y=cbind(SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY$x,SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY$y),
                                  beta0=190, gamma0=0.2,eps_beta=65, eps_gamma=0.16, R=SS1_SPP_R_hat, K=5, T=120000)
time_end <- Sys.time()
SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K5_T120000_1_time <- time_end-time_start
# stopCluster(cl)
# # Time difference of 33.15239 mins

# # Noisy Exchange K6
# cl <- parallel::makeCluster(detectCores()[1]-1)
# clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel computation
time_start <- Sys.time()
SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K6_T120000_1 <-
  SPP_Parallel_Noisy_MH(Y=cbind(SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY$x,SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY$y),
                                  beta0=190, gamma0=0.2,eps_beta=65, eps_gamma=0.16, R=SS1_SPP_R_hat, K=6, T=120000)
time_end <- Sys.time()
SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K6_T120000_1_time <- time_end-time_start
# stopCluster(cl)
# # Time difference of 35.10859 mins

# # Noisy Exchange K7
# cl <- parallel::makeCluster(detectCores()[1]-1)
# clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel computation
time_start <- Sys.time()
SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K7_T120000_1 <-
  SPP_Parallel_Noisy_MH(Y=cbind(SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY$x,SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY$y),
                                  beta0=190, gamma0=0.2,eps_beta=65, eps_gamma=0.16, R=SS1_SPP_R_hat, K=7, T=120000)
time_end <- Sys.time()
SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K7_T120000_1_time <- time_end-time_start
# stopCluster(cl)
# # Time difference of 41.63192 mins

# # Noisy Exchange K8
# cl <- parallel::makeCluster(detectCores()[1]-1)
# clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel computation
time_start <- Sys.time()
SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K8_T120000_1 <-
  SPP_Parallel_Noisy_MH(Y=cbind(SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY$x,SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY$y),
                                  beta0=190, gamma0=0.2,eps_beta=65, eps_gamma=0.16, R=SS1_SPP_R_hat, K=8, T=120000)
time_end <- Sys.time()
SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K8_T120000_1_time <- time_end-time_start
stopCluster(cl)
# Time difference of 58.90153 mins
```

The corresponding reference implementation time is also provided above for each case.
The summarizing processes are similar to the ground truth case shown above and we propose not to put them here in order not to make this file become too lengthy.

### The SS1 Implementation of the ABC-MCMC Algorithm

As we discussed in section $4$ of the paper, the ABC-MCMC algorithm we make the comparisons with requires a pilot run to approximate the linear coefficients of the linear regression and to decide the acceptance thresholds.
We start from setting the $K$-function for the observation $\boldsymbol{y}$ with respect to $\hat{R}$, and setting the number of iterations in the pilot run.

``` r
# obtain Kfunc for Y
SS1_SPP_Kfunc_Obs=as.function(Kest(SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY, correction="isotropic"))
SS1_SPP_Kfunc_Obs_R_hat <- SS1_SPP_Kfunc_Obs(SS1_SPP_R_hat)
# Set L for the pilot run
SS1_SPP_Pilot.L <- 10000
```

Then we define a function to apply the pilot run and implement with parallel computation for it.
Here the settings follow what we introduced in the simulation study section $6.1$ of the paper, that is, $\pi(\beta)=\text{U}(50, 400), \pi(\gamma)=\text{U}(0,1)$ and $R=\hat{R}$ we obtained by the profile pseudo-likelihood method illustrated above.

``` r
# ABC-MCMC Pilot Draws Function
ABCMCMC_Pilot_lth_Draw_SPP <- function(x, N_Y, R, Kfunc_Obs_R_hat){ # Current state beta and gamma
  beta=runif(1,50,400)
  gamma=runif(1,0,1)
  X=rStrauss(beta,gamma,R,square(1))
  Kfunc_X=as.function(Kest(X, correction="isotropic"))
  eta <- c(log(X$n)-log(N_Y),(sqrt(Kfunc_X(R))-sqrt(Kfunc_Obs_R_hat))^2)
  return(list(beta=beta,gamma=gamma,X=X,eta=eta))
}
# Implement pilot run in parallel
cl <- parallel::makeCluster(detectCores()[1]-1)
clusterExport(cl=cl, list("rStrauss", "square","Kest")) # In order to use this function for parallel computation
time_start <- Sys.time()
SS1_SPP_Pilot <- parLapply(cl, 1:SS1_SPP_Pilot.L, ABCMCMC_Pilot_lth_Draw_SPP, N_Y=SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY$n, R=SS1_SPP_R_hat,
                           Kfunc_Obs_R_hat = SS1_SPP_Kfunc_Obs_R_hat)
time_end <- Sys.time()
SS1_SPP_Pilot.time <- time_end-time_start
# Time difference of 1.518383 mins
stopCluster(cl)
```

The corresponding reference implementation time of the pilot run is also provided.
However, it can be neglected if we compare to the time taken by the main algorithm of the ABC-MCMC algorithm and thus it was not counted in the comparisons.
Due to the fact that the parallel computation code returns a list each element of which further contains a list of outputs from each iteration of the pilot run.
We need to extract each single chain of the parameter by the following code.

``` r
# Transform pilot draws to single chains of parameters
SS1_SPP_Pilot.beta <- c()
SS1_SPP_Pilot.gamma <- c()
SS1_SPP_Pilot.X <- list()
SS1_SPP_Pilot.eta <- matrix(0,SS1_SPP_Pilot.L,2)
for(l in 1:SS1_SPP_Pilot.L){
  SS1_SPP_Pilot.beta[l]=SS1_SPP_Pilot[[l]]$beta
  SS1_SPP_Pilot.gamma[l]=SS1_SPP_Pilot[[l]]$gamma
  SS1_SPP_Pilot.X[[l]]=SS1_SPP_Pilot[[l]]$X
  SS1_SPP_Pilot.eta[l,] <- SS1_SPP_Pilot[[l]]$eta
}
```

Recall from the paper that the generalised linear regression under a multi-response Gaussian family is fit with lasso regression, and cross-validation is applied to determine the penalty parameter for the lasso.
This can be accomplished by the `cv.glmnet()` function in the `glmnet` package where `family="mgaussian"` corresponds to the multi-response Gaussian family, `alpha=1` corresponds to the lasso regression.

``` r
# apply glmnet for regression, i.e. glm with lasso and determine the penalty parameter for the lasso by cross-validation (cv)
library(glmnet)
SS1_SPP_Pilot.lmCoef <- coef(cv.glmnet(x=SS1_SPP_Pilot.eta,y=log(cbind(SS1_SPP_Pilot.beta,SS1_SPP_Pilot.gamma)),family="mgaussian",alpha=1),s="lambda.min")
```

Then we can extract the linear coefficients and calculate the sample variance of each estimated model parameter as well as the distance measures $\{ \Psi(\boldsymbol{\hat{\theta}}\_l, \boldsymbol{\hat{a}}) \}^{L}_{l = 1}$ for each iteration of the pilot run. 

``` r
# Linear coefficients for each model parameter 
SS1_SPP_Pilot.lmCoefBeta <- as.matrix(SS1_SPP_Pilot.lmCoef$SS1_SPP_Pilot.beta) # store the coefficients
SS1_SPP_Pilot.lmCoefGamma <- as.matrix(SS1_SPP_Pilot.lmCoef$SS1_SPP_Pilot.gamma)
# calculate variance of log(\hat{theta})
SS1_SPP_Pilot.VarBeta<-c(var(cbind(1,SS1_SPP_Pilot.eta)%*%SS1_SPP_Pilot.lmCoefBeta)) 
SS1_SPP_Pilot.VarGamma<-c(var(cbind(1,SS1_SPP_Pilot.eta)%*%SS1_SPP_Pilot.lmCoefGamma))
# Calculate Psi of Pilot run
SS1_SPP_Pilot.psi <- ((SS1_SPP_Pilot.eta%*%SS1_SPP_Pilot.lmCoefBeta[2:3])^2)/SS1_SPP_Pilot.VarBeta +
  ((SS1_SPP_Pilot.eta%*%SS1_SPP_Pilot.lmCoefGamma[2:3])^2)/SS1_SPP_Pilot.VarGamma
```

We specify three different percentiles for the acceptance thresholds $\epsilon$ in this simulation study.

``` r
# Take p percentile
SS1_SPP_Pilot.0.005eps <- quantile(SS1_SPP_Pilot.psi,probs=0.005)[[1]]
SS1_SPP_Pilot.0.01eps <- quantile(SS1_SPP_Pilot.psi,probs=0.01)[[1]]
SS1_SPP_Pilot.0.025eps <- quantile(SS1_SPP_Pilot.psi,probs=0.025)[[1]]
```

The main ABC-MCMC algorithm is implemented as following.
We start from the case where the $\epsilon$ is set as $2.5$ percent estimated percentile.
Recall here that the approximate parallel computation as we discussed in the section $4$ of the paper is implemented.

``` r
## ABC-MCMC algorithm with approximate parallel computation p0.025
NumCores <- detectCores()[1]-1
cl <- parallel::makeCluster(NumCores)
clusterExport(cl=cl, list("rStrauss", "square", "Kest")) # In order to use this function for parallel computation
time_start <- Sys.time()
SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.025_T120000_1 <-
  S.G.Parallel.ABC.MCMC.Strauss(Y = SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY, beta0=190, gamma0=0.2,eps_beta=65, eps_gamma=0.16,
                                lmCoefBeta = SS1_SPP_Pilot.lmCoefBeta, lmCoefGamma = SS1_SPP_Pilot.lmCoefGamma,
                                Pilot.VarBeta = SS1_SPP_Pilot.VarBeta, Pilot.VarGamma = SS1_SPP_Pilot.VarGamma,
                                eps = SS1_SPP_Pilot.0.025eps, R=SS1_SPP_R_hat, T=120000)
time_end <- Sys.time()
SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.025_T120000_1_time <- time_end-time_start
stopCluster(cl)
# Time difference of 2.401528 hours
```

Similar as the exchange and noisy M-H algorithms experiments, we can obtain the summarized statistics of the outputs as follows.
Note here that we further monitor the number of draws in the `repeat` loop until the acceptance condition is satisfied.
And we also monitor whether there are more than one draws which satisfy the the acceptance condition within one round of $J$ parallel draws.

``` r
# # Example summary statistics
# Acceptance rate
SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.025_T120000_1$AcceptanceRate
# Posterior trace plot
plot(SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.025_T120000_1$beta[20001:120001], type = "l")
plot(SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.025_T120000_1$gamma[20001:120001], type = "l")
# Posterior density plot
plot(density(SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.025_T120000_1$beta[20001:120001]))
plot(density(SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.025_T120000_1$gamma[20001:120001]))
# ESS/s
ESS(SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.025_T120000_1$beta[20001:120001])/(SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.025_T120000_1_time[[1]]*3600)
ESS(SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.025_T120000_1$gamma[20001:120001])/(SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.025_T120000_1_time[[1]]*3600)
# Posterior mean
mean(SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.025_T120000_1$beta[20001:120001])
mean(SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.025_T120000_1$gamma[20001:120001])
# Posterior standard deviation
sd(SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.025_T120000_1$beta[20001:120001])
sd(SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.025_T120000_1$gamma[20001:120001])
# check the number of draws in while/repeat loop until the acceptance condition is satisfied
mean(SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.025_T120000_1$NumOfDrawsUntilAcceptance[20001:120001])
plot(SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.025_T120000_1$NumOfDrawsUntilAcceptance[20001:120001],type="l")
median(SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.025_T120000_1$NumOfDrawsUntilAcceptance[20001:120001])
# check the number of accepted draws when the while/repeat loop stops
mean(SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.025_T120000_1$NumOfAcceptedDrawsInEachNumCoresDraws[20001:120001])
table(SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.025_T120000_1$NumOfAcceptedDrawsInEachNumCoresDraws[20001:120001])
```

Similar implementation and summary statistics can also be applied for the case where $p=0.01$ and $p=0.005$.

``` r
## ABC-MCMC algorithm with approximate parallel computation p0.01
NumCores <- detectCores()[1]-1
cl <- parallel::makeCluster(NumCores)
clusterExport(cl=cl, list("rStrauss", "square", "Kest")) # In order to use this function for parallel computation
time_start <- Sys.time()
SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.01_T120000_1 <-
  S.G.Parallel.ABC.MCMC.Strauss(Y = SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY, beta0=190, gamma0=0.2,eps_beta=65, eps_gamma=0.16,
                                lmCoefBeta = SS1_SPP_Pilot.lmCoefBeta, lmCoefGamma = SS1_SPP_Pilot.lmCoefGamma,
                                Pilot.VarBeta = SS1_SPP_Pilot.VarBeta, Pilot.VarGamma = SS1_SPP_Pilot.VarGamma,
                                eps = SS1_SPP_Pilot.0.01eps, R=SS1_SPP_R_hat, T=120000)
time_end <- Sys.time()
SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.01_T120000_1_time <- time_end-time_start
stopCluster(cl)
# Time difference of 3.880879 hours

#--------------------------------------------------------------------------------------------------------------------------------------------
## ABC-MCMC main algorithm with approximate parallel computation p0.005
NumCores <- detectCores()[1]-1
cl <- parallel::makeCluster(NumCores)
clusterExport(cl=cl, list("rStrauss", "square", "Kest")) # In order to use this function for parallel computation
time_start <- Sys.time()
SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.005_T120000_1 <-
  S.G.Parallel.ABC.MCMC.Strauss(Y = SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY, beta0=190, gamma0=0.2,eps_beta=65, eps_gamma=0.16,
                                lmCoefBeta = SS1_SPP_Pilot.lmCoefBeta, lmCoefGamma = SS1_SPP_Pilot.lmCoefGamma,
                                Pilot.VarBeta = SS1_SPP_Pilot.VarBeta, Pilot.VarGamma = SS1_SPP_Pilot.VarGamma,
                                eps = SS1_SPP_Pilot.0.005eps, R=SS1_SPP_R_hat, T=120000)
time_end <- Sys.time()
SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.005_T120000_1_time <- time_end-time_start
stopCluster(cl)
# Time difference of 6.113184 hours
```

The simulation study $1$ boxplots Figure $2$ can be recovered via the code provided below.

``` r
par(mfrow=c(1,2),mai = c(0.3, 0.3, 0.05, 0.01),mgp=c(0.45,0.3,0))
boxplot(SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T1200000_1$beta[200001:1200001],
        SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T120000_1$beta[20001:120001],
        SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K2_T120000_1$beta[20001:120001],
        SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K3_T120000_1$beta[20001:120001],
        SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K4_T120000_1$beta[20001:120001],
        SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K5_T120000_1$beta[20001:120001],
        SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K6_T120000_1$beta[20001:120001],
        SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K7_T120000_1$beta[20001:120001],
        SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K8_T120000_1$beta[20001:120001],
        SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.025_T120000_1$beta[20001:120001],
        SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.01_T120000_1$beta[20001:120001],
        SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.005_T120000_1$beta[20001:120001],
        xlab = "",ylab = "", main = "",cex.axis = 0.6)
title(xlab = "",ylab = TeX(r'($\beta$)'), main = "", mgp=c(0.45,0.3,0),cex.main=1,cex.lab = 0.8)
axis(1, at=c(1:12), labels = c("GT","Ex","K2","K3","K4","K5","K6","K7","K8","p2.5","p1","p0.5"),cex.axis=0.6)
abline(h=median(SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T1200000_1$beta[200001:1200001]),col = 2,lty = 2)

boxplot(SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T1200000_1$gamma[200001:1200001],
        SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T120000_1$gamma[20001:120001],
        SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K2_T120000_1$gamma[20001:120001],
        SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K3_T120000_1$gamma[20001:120001],
        SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K4_T120000_1$gamma[20001:120001],
        SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K5_T120000_1$gamma[20001:120001],
        SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K6_T120000_1$gamma[20001:120001],
        SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K7_T120000_1$gamma[20001:120001],
        SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K8_T120000_1$gamma[20001:120001],
        SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.025_T120000_1$gamma[20001:120001],
        SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.01_T120000_1$gamma[20001:120001],
        SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.005_T120000_1$gamma[20001:120001],
        xlab = "",ylab = "", main = "",cex.axis = 0.6)
title(xlab = "",ylab = TeX(r'($\gamma$)'), main = "", mgp=c(0.45,0.3,0),cex.main=1,cex.lab = 0.8)
axis(1, at=c(1:12), labels = c("GT","Ex","K2","K3","K4","K5","K6","K7","K8","p2.5","p1","p0.5"),cex.axis=0.6)
abline(h=median(SS1_SPP_Beta200_Gamma0.1_R0.05_NoisyMH_K1_T1200000_1$gamma[200001:1200001]),col = 2,lty = 2)
par(mfrow=c(1,1),mai = c(1.02, 0.82, 0.82, 0.42),mgp=c(3,1,0))
```

The code for the simulation study $1$ Figure $3$ is also shown here.

``` r
par(mfrow=c(1,3),mai = c(0.3, 0.3, 0.25, 0.05),mgp=c(1.25,0.45,0))
plot(SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.025_T120000_1$NumOfDrawsUntilAcceptance[20001:120001],type="l",xlab="",ylab="")
title(main = "ABC-MCMC p2.5", mgp=c(1,0.25,0),cex.main=1,cex.lab = 0.8)

plot(SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.01_T120000_1$NumOfDrawsUntilAcceptance[20001:120001],type="l",xlab="",ylab="")
title(main = "ABC-MCMC p1", mgp=c(1,0.25,0),cex.main=1,cex.lab = 0.8)

plot(SS1_SPP_Beta200_Gamma0.1_R0.05_ABCMCMC_p0.005_T120000_1$NumOfDrawsUntilAcceptance[20001:120001],type="l",xlab="",ylab="")
title(main = "ABC-MCMC p0.5", mgp=c(1,0.25,0),cex.main=1,cex.lab = 0.8)
par(mfrow=c(1,1),mai = c(1.02, 0.82, 0.82, 0.42),mgp=c(3,1,0))
```

## Determinantal Point Process with A Gaussian Kernel Simulation Study $2$

We illustrate the code and plots for our second determinantal point Process with a Gaussian Kernel (dppG) in this simulation study 2 (SS2).
Again we start from the generation of the artificial data from the dppG with the settings applied in the section $6.2$ of the paper, that is, $\tau = 100, \sigma = 0.05$.

``` r
## Generate from dppG
dppG_Y_observation <- simulate(dppGauss(lambda=100, alpha=0.05, d=2))
# utils::write.table(x=dppG_Y_observation,file="SS2_dppG_Tau100_Sigma0.05_ObsY.csv", sep="," , row.names = FALSE, col.names=FALSE)

## Simulation study 2 observation
SS2_dppG_Tau100_Sigma0.05_ObsY <- as.matrix(read.csv("SS2_dppG_Tau100_Sigma0.05_ObsY.csv",header=FALSE))
SS2_dppG_Tau100_Sigma0.05_ObsY <- ppp(SS2_dppG_Tau100_Sigma0.05_ObsY[,1],SS2_dppG_Tau100_Sigma0.05_ObsY[,2]) # transform to point pattern
```

The `dppG_logDensity()` function in the [`Algorithm_Functions_for_RSPP.R`] evaluates the log density of the $\hat{X}_S$ without the normalising constant.
The `dppG_MH()` function implements the Metropolis-Hastings algorithm for the dppG due to the tractability of the likelihood normalising term of the $\hat{X}_S$.
The function `dppG_Noisy_E_kth_Ratio()`, which is similar as that in SPP cases, calculates the unnormalised likelihood ratio $\frac{q(x_k'|\theta^{(t-1)})}{q(x_k'|\theta')}$ for noisy M-H algorithm.
The function `dppG_Parallel_Noisy_MH()` implements the exchange or the noisy M-H algorithm for dppG with parallel computation.

Recall here that we propose to apply an approximation of the unnormalised likelihood function for $\hat{X}_S$ in order to improve the efficiency.
The functions `Approx_dppG_Noisy_E_kth_Ratio()` and `Approx_dppG_Parallel_Noisy_MH()` corresponds to the approximate exchange and noisy M-H algorithms.
The ABC-MCMC functions for dppG, `S.G.ABC.MCMC.dppG.repeat.draws()` and `S.G.Parallel.ABC.MCMC.dppG()`, are similar as those of SPP cases and the only difference is that the summary statistic $\boldsymbol{\eta_2}$ is now evaluated at $10$ equally spaced $r_i$'s from $i=1$ to $i=10$ as we discussed in the end of section $5$ of the paper.

The ground truth in this dppG simulation study is to implement the M-H algorithm for $120,000$ iterations with $20,000$ burn-in.
The initial states are set as $\tau_0=125,\sigma_0=0.04$ and the proposal epsilons are tuned to be $\epsilon_{\tau}=32, \epsilon_{\sigma}=0.015$.

``` r
# MH algorithm dppG Ground Truth
time_start <- Sys.time()
SS2_dppG_Tau100_Sigma0.05_MH_T120000_1 <-
  dppG_MH(Y=cbind(SS2_dppG_Tau100_Sigma0.05_ObsY$x,SS2_dppG_Tau100_Sigma0.05_ObsY$y),
                tau0=125, sigma0=0.04,eps_tau=32, eps_sigma=0.015, T=120000)
time_end <- Sys.time()
SS2_dppG_Tau100_Sigma0.05_MH_T120000_1_time <- time_end-time_start
# Time difference of 1.980805 days
```

It can be seen that the implementation time is much longer than the SPP cases even though the number of iterations is much less than that of the SS1.

### The SS2 Implementation of the M-H Algorithm

Similarly the M-H algorithm can also be implemented for $12,000$ iterations with $2000$ iteration burn-in for the algorithm comparisons.

``` r
# MH algorithm dppG 12000 iteration
time_start <- Sys.time()
SS2_dppG_Tau100_Sigma0.05_MH_T12000_1 <-
  dppG_MH(Y=cbind(SS2_dppG_Tau100_Sigma0.05_ObsY$x,SS2_dppG_Tau100_Sigma0.05_ObsY$y),
                tau0=125, sigma0=0.04,eps_tau=32, eps_sigma=0.015, T=12000)
time_end <- Sys.time()
SS2_dppG_Tau100_Sigma0.05_MH_T12000_1_time <- time_end-time_start
# Time difference of 5.498604 hours
```

Due to the fact that the code for summary statistics are similar as that of the SPP cases, we propose not to show amount of the repeated code of the summary statistics for all the following implementations.
The implementation code for the exchange, noisy M-H, approximate exchange, approximate noisy M-H algorithms are also similar as those of SPP cases as shown below.

### The SS2 Implementations of the Exchange and Noisy M-H Algorithms

``` r
#--------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------

# Exchange algorithm == Noisy MH K1 algorithm for dppG
cl <- parallel::makeCluster(detectCores()[1]-1)
clusterExport(cl=cl, list("simulate", "dppGauss","Kest","dppG_logDensity","fourierbasisraw","logdet"))
time_start <- Sys.time()
SS2_dppG_Tau100_Sigma0.05_NoisyMH_K1_T12000_1 <-
  dppG_Parallel_Noisy_MH(Y=cbind(SS2_dppG_Tau100_Sigma0.05_ObsY$x,SS2_dppG_Tau100_Sigma0.05_ObsY$y),
                                     tau0=125, sigma0=0.04,eps_tau=32, eps_sigma=0.015, K=1, T=12000)
time_end <- Sys.time()
SS2_dppG_Tau100_Sigma0.05_NoisyMH_K1_T12000_1_time <- time_end-time_start
stopCluster(cl)
# Time difference of 11.79334 hours

# Noisy MH K2 algorithm for dppG
cl <- parallel::makeCluster(detectCores()[1]-1)
clusterExport(cl=cl, list("simulate", "dppGauss","Kest","dppG_logDensity","fourierbasisraw","logdet"))
time_start <- Sys.time()
SS2_dppG_Tau100_Sigma0.05_NoisyMH_K2_T12000_1 <-
  dppG_Parallel_Noisy_MH(Y=cbind(SS2_dppG_Tau100_Sigma0.05_ObsY$x,SS2_dppG_Tau100_Sigma0.05_ObsY$y),
                                     tau0=125, sigma0=0.04,eps_tau=32, eps_sigma=0.015, K=2, T=12000)
time_end <- Sys.time()
SS2_dppG_Tau100_Sigma0.05_NoisyMH_K2_T12000_1_time <- time_end-time_start
stopCluster(cl)
# Time difference of 12.95519 hours
```

### The SS2 Implementations of the Approximate Exchange and the Approximate Noisy M-H Algorithms

``` r
# Approximate exchange == Noisy MH K1 algorithm for dppG
cl <- parallel::makeCluster(detectCores()[1]-1)
clusterExport(cl=cl, list("simulate", "dppGauss","Kest","Approx_dppG_Noisy_E_Nth_Ratio","logdet")) 
time_start <- Sys.time()
Approx_SS2_dppG_Tau100_Sigma0.05_NoisyMH_K1_T12000_1 <-
  Approx_dppG_Parallel_Noisy_MH(Y=cbind(SS2_dppG_Tau100_Sigma0.05_ObsY$x,SS2_dppG_Tau100_Sigma0.05_ObsY$y), 
                               tau0=125, sigma0=0.04,eps_tau=32, eps_sigma=0.015, K=1, T=12000)
time_end <- Sys.time()
Approx_SS2_dppG_Tau100_Sigma0.05_NoisyMH_K1_T12000_1_time <- time_end-time_start
# stopCluster(cl)
# # Time difference of 1.652786 hours

# # Approximate Noisy MH K2 algorithm for dppG
# cl <- parallel::makeCluster(detectCores()[1]-1)
# clusterExport(cl=cl, list("simulate", "dppGauss","Kest","Approx_dppG_Noisy_E_Nth_Ratio","logdet")) 
time_start <- Sys.time()
Approx_SS2_dppG_Tau100_Sigma0.05_NoisyMH_K2_T12000_1 <-
  Approx_dppG_Parallel_Noisy_MH(Y=cbind(SS2_dppG_Tau100_Sigma0.05_ObsY$x,SS2_dppG_Tau100_Sigma0.05_ObsY$y), 
                               tau0=125, sigma0=0.04,eps_tau=32, eps_sigma=0.015, K=2, T=12000)
time_end <- Sys.time()
Approx_SS2_dppG_Tau100_Sigma0.05_NoisyMH_K2_T12000_1_time <- time_end-time_start
# stopCluster(cl)
# # Time difference of 2.074797 hours

# # Approximate Noisy MH K3 algorithm for dppG
# cl <- parallel::makeCluster(detectCores()[1]-1)
# clusterExport(cl=cl, list("simulate", "dppGauss","Kest","Approx_dppG_Noisy_E_Nth_Ratio","logdet")) 
time_start <- Sys.time()
Approx_SS2_dppG_Tau100_Sigma0.05_NoisyMH_K3_T12000_1 <-
  Approx_dppG_Parallel_Noisy_MH(Y=cbind(SS2_dppG_Tau100_Sigma0.05_ObsY$x,SS2_dppG_Tau100_Sigma0.05_ObsY$y), 
                               tau0=125, sigma0=0.04,eps_tau=32, eps_sigma=0.015, K=3, T=12000)
time_end <- Sys.time()
Approx_SS2_dppG_Tau100_Sigma0.05_NoisyMH_K3_T12000_1_time <- time_end-time_start
# stopCluster(cl)
# # Time difference of 2.364647 hours

# # Approximate Noisy MH K4 algorithm for dppG
# cl <- parallel::makeCluster(detectCores()[1]-1)
# clusterExport(cl=cl, list("simulate", "dppGauss","Kest","Approx_dppG_Noisy_E_Nth_Ratio","logdet")) 
time_start <- Sys.time()
Approx_SS2_dppG_Tau100_Sigma0.05_NoisyMH_K4_T12000_1 <-
  Approx_dppG_Parallel_Noisy_MH(Y=cbind(SS2_dppG_Tau100_Sigma0.05_ObsY$x,SS2_dppG_Tau100_Sigma0.05_ObsY$y), 
                               tau0=125, sigma0=0.04,eps_tau=32, eps_sigma=0.015, K=4, T=12000)
time_end <- Sys.time()
Approx_SS2_dppG_Tau100_Sigma0.05_NoisyMH_K4_T12000_1_time <- time_end-time_start
# stopCluster(cl)
# # Time difference of 2.852505 hours

# # Approximate Noisy MH K5 algorithm for dppG
# cl <- parallel::makeCluster(detectCores()[1]-1)
# clusterExport(cl=cl, list("simulate", "dppGauss","Kest","Approx_dppG_Noisy_E_Nth_Ratio","logdet")) 
time_start <- Sys.time()
Approx_SS2_dppG_Tau100_Sigma0.05_NoisyMH_K5_T12000_1 <-
  Approx_dppG_Parallel_Noisy_MH(Y=cbind(SS2_dppG_Tau100_Sigma0.05_ObsY$x,SS2_dppG_Tau100_Sigma0.05_ObsY$y), 
                               tau0=125, sigma0=0.04,eps_tau=32, eps_sigma=0.015, K=5, T=12000)
time_end <- Sys.time()
Approx_SS2_dppG_Tau100_Sigma0.05_NoisyMH_K5_T12000_1_time <- time_end-time_start
# stopCluster(cl)
# # Time difference of 3.27993 hours

# # Approximate Noisy MH K6 algorithm for dppG
# cl <- parallel::makeCluster(detectCores()[1]-1)
# clusterExport(cl=cl, list("simulate", "dppGauss","Kest","Approx_dppG_Noisy_E_Nth_Ratio","logdet"))
time_start <- Sys.time()
Approx_SS2_dppG_Tau100_Sigma0.05_NoisyMH_K6_T12000_1 <-
  Approx_dppG_Parallel_Noisy_MH(Y=cbind(SS2_dppG_Tau100_Sigma0.05_ObsY$x,SS2_dppG_Tau100_Sigma0.05_ObsY$y), 
                               tau0=125, sigma0=0.04,eps_tau=32, eps_sigma=0.015, K=6, T=12000)
time_end <- Sys.time()
Approx_SS2_dppG_Tau100_Sigma0.05_NoisyMH_K6_T12000_1_time <- time_end-time_start
stopCluster(cl)
# Time difference of 3.971992 hours
```

### The SS2 Implementation of the ABC-MCMC Algorithm

The code for the pilot run of the ABC-MCMC algorithm for dppG is shown as below.
Note here that the pilot sampling of the parameter $\tau$ follows the prior distribution $\pi(\tau)=\text{U}(50, 200)$.
Though the prior distribution of $\sigma$ is proposed to be $\pi(\sigma)=\text{U}(0.001,1/\sqrt{10\pi})$, the existance of the dppG restricts the parameter $\sigma$ to be bounded by $1/\sqrt{\tau\pi}$.
Moreover, the $10$ equally spaced $r_i$'s from $i=1$ to $i=10$ are specified here by the code `r_M <- seq(0.01,0.1,0.01)`.

``` r
# obtain Kfunc for Y
Kfunc_Obs=as.function(Kest(SS2_dppG_Tau100_Sigma0.05_ObsY, correction="isotropic"))
r_M <- seq(0.01,0.1,0.01) # i.e. M = 10
SS2_dppG_Kfunc_Obs_rM <- Kfunc_Obs(r_M) # M = 10 equally spaced values from 0.01 to 0.1
SS2_dppG_Pilot.L <- 10000

# ABC-MCMC Pilot Draws Function with r_M
ABCMCMC_Pilot_lth_Draw_MdppG <- function(x, N_Y, r_M, SS2_dppG_Kfunc_Obs_rM){ # Current state beta and gamma
  tau=runif(1,50,200)
  sigma=runif(1,0.001,1/sqrt(pi*tau))
  X=simulate(dppGauss(lambda=tau, alpha=sigma, d=2))
  Kfunc_X=as.function(Kest(X, correction="isotropic"))
  eta <- c(log(X$n)-log(N_Y),(sqrt(Kfunc_X(r_M))-sqrt(SS2_dppG_Kfunc_Obs_rM))^2)
  return(list(tau=tau,sigma=sigma,X=X,eta=eta))
}
# Implement pilot run by parallel computation
cl <- parallel::makeCluster(detectCores()[1]-1)
clusterExport(cl=cl, list("simulate", "dppGauss","Kest")) # In order to use this function for parallel computation
time_start <- Sys.time()
SS2_dppG_Pilot <- parLapply(cl, 1:SS2_dppG_Pilot.L, ABCMCMC_Pilot_lth_Draw_MdppG, N_Y=SS2_dppG_Tau100_Sigma0.05_ObsY$n,
                            r_M=r_M, SS2_dppG_Kfunc_Obs_rM = SS2_dppG_Kfunc_Obs_rM)
time_end <- Sys.time()
SS2_dppG_Pilot.time <- time_end-time_start
# Time difference of 38.05565 mins # Note that we didn't count this significant pilot time when measuring the efficiency
stopCluster(cl)
# Transform pilot draws
SS2_dppG_Pilot.tau <- c()
SS2_dppG_Pilot.sigma <- c()
SS2_dppG_Pilot.X <- list()
SS2_dppG_Pilot.eta <- matrix(0,SS2_dppG_Pilot.L,11)
for(l in 1:SS2_dppG_Pilot.L){
  SS2_dppG_Pilot.tau[l]=SS2_dppG_Pilot[[l]]$tau
  SS2_dppG_Pilot.sigma[l]=SS2_dppG_Pilot[[l]]$sigma
  SS2_dppG_Pilot.X[[l]]=SS2_dppG_Pilot[[l]]$X
  SS2_dppG_Pilot.eta[l,] <- SS2_dppG_Pilot[[l]]$eta
}

# apply glmnet for regression, i.e. glm with lasso and determine the penalty parameter for the lasso by cross-validation (cv)
library(glmnet)
SS2_dppG_Pilot.lmCoef <- 
  coef(cv.glmnet(x=SS2_dppG_Pilot.eta,y=log(cbind(SS2_dppG_Pilot.tau,SS2_dppG_Pilot.sigma)),family="mgaussian",alpha=1),s="lambda.min")
SS2_dppG_Pilot.lmCoefTau <- as.matrix(SS2_dppG_Pilot.lmCoef$SS2_dppG_Pilot.tau) # store the coefficients
SS2_dppG_Pilot.lmCoefSigma <- as.matrix(SS2_dppG_Pilot.lmCoef$SS2_dppG_Pilot.sigma)
SS2_dppG_Pilot.VarTau=c(var(cbind(1,SS2_dppG_Pilot.eta)%*%SS2_dppG_Pilot.lmCoefTau)) # calculate variance of log(theta)^hat
SS2_dppG_Pilot.VarSigma=c(var(cbind(1,SS2_dppG_Pilot.eta)%*%SS2_dppG_Pilot.lmCoefSigma))
# Calculate Psi of Pilot run
SS2_dppG_Pilot.psi <- ((SS2_dppG_Pilot.eta%*%SS2_dppG_Pilot.lmCoefTau[2:12])^2)/SS2_dppG_Pilot.VarTau + 
  ((SS2_dppG_Pilot.eta%*%SS2_dppG_Pilot.lmCoefSigma[2:12])^2)/SS2_dppG_Pilot.VarSigma
# Take p percentile
SS2_dppG_Pilot.0.005eps <- quantile(SS2_dppG_Pilot.psi,probs=0.005)[[1]]
SS2_dppG_Pilot.0.01eps <- quantile(SS2_dppG_Pilot.psi,probs=0.01)[[1]]
SS2_dppG_Pilot.0.025eps <- quantile(SS2_dppG_Pilot.psi,probs=0.025)[[1]]
```

The main implementation code of the ABC-MCMC algorithm with approximate parallel computation is following for the two cases $p=0.01$ and $p=0.005$.

``` r
## ABC-MCMC algorithm for dppG with approximate parallel computation p0.01
NumCores <- detectCores()[1]-1
cl <- parallel::makeCluster(NumCores)
clusterExport(cl=cl, list("simulate", "dppGauss","Kest")) # In order to use this function for parallel computation
time_start <- Sys.time()
SS2_dppG_Tau100_Sigma0.05_ABCMCMC_p0.01_T12000_1 <- 
  S.G.Parallel.ABC.MCMC.dppG(Y = SS2_dppG_Tau100_Sigma0.05_ObsY, tau0=125, sigma0=0.04,eps_tau=32, eps_sigma=0.015, 
                             lmCoefTau = SS2_dppG_Pilot.lmCoefTau, lmCoefSigma = SS2_dppG_Pilot.lmCoefSigma, 
                             Pilot.VarTau = SS2_dppG_Pilot.VarTau, Pilot.VarSigma = SS2_dppG_Pilot.VarSigma, 
                             eps = SS2_dppG_Pilot.0.01eps, r_M=seq(0.01,0.1,0.01), T=12000)
time_end <- Sys.time()
SS2_dppG_Tau100_Sigma0.05_ABCMCMC_p0.01_T12000_1_time <- time_end-time_start
stopCluster(cl)
# Time difference of 11.67918 hours

## ABC-MCMC algorithm for dppG with approximate parallel computation p0.005
NumCores <- detectCores()[1]-1
cl <- parallel::makeCluster(NumCores)
clusterExport(cl=cl, list("simulate", "dppGauss","Kest")) # In order to use this function for parallel computation
time_start <- Sys.time()
SS2_dppG_Tau100_Sigma0.05_ABCMCMC_p0.005_T12000_1 <- 
  S.G.Parallel.ABC.MCMC.dppG(Y = SS2_dppG_Tau100_Sigma0.05_ObsY, tau0=125, sigma0=0.04,eps_tau=32, eps_sigma=0.015, 
                             lmCoefTau = SS2_dppG_Pilot.lmCoefTau, lmCoefSigma = SS2_dppG_Pilot.lmCoefSigma, 
                             Pilot.VarTau = SS2_dppG_Pilot.VarTau, Pilot.VarSigma = SS2_dppG_Pilot.VarSigma, 
                             eps = SS2_dppG_Pilot.0.005eps, r_M=seq(0.01,0.1,0.01), T=12000)
time_end <- Sys.time()
SS2_dppG_Tau100_Sigma0.05_ABCMCMC_p0.005_T12000_1_time <- time_end-time_start
stopCluster(cl)
# Time difference of 21.44336 hours
```

The plot of the point locations of the SS2 aritificial dataset shown as Figure $4$ in section $6.2$ can be recovered by the following code.

```r
par(mfrow=c(1,1),mai = c(0.5, 0.5, 0.5, 0.5),mgp=c(1.25,0.45,0))
plot(SS2_dppG_Tau100_Sigma0.05_ObsY$x,SS2_dppG_Tau100_Sigma0.05_ObsY$y,xlab = "",ylab = "")
title(main = "", mgp=c(1,0.25,0),cex.main=1,cex.lab = 0.8)
par(mfrow=c(1,1),mai = c(1.02, 0.82, 0.82, 0.42),mgp=c(3,1,0))
```

The trace plots of the outputs for the algorithms shown as Figure $5$ can be recovered by:

```r
par(mfrow=c(2,6),mai = c(0.15, 0.15, 0.15, 0.01),mgp=c(0.75,0.25,0))
plot(SS2_dppG_Tau100_Sigma0.05_MH_T12000_1$tau[2001:12001], type = "l",xlab = "",ylab = "", main = "MH",cex.axis = 0.9,cex.main=1, ylim=c(65,145))
plot(SS2_dppG_Tau100_Sigma0.05_NoisyMH_K1_T12000_1$tau[2001:12001], type = "l",xlab = "",ylab = "", main = "Ex",cex.axis = 0.9,cex.main=1, ylim=c(65,145))
plot(SS2_dppG_Tau100_Sigma0.05_NoisyMH_K2_T12000_1$tau[2001:12001], type = "l",xlab = "",ylab = "", main = "NMH K2",cex.axis = 0.9,cex.main=1, ylim=c(65,145))
plot(Approx_SS2_dppG_Tau100_Sigma0.05_NoisyMH_K1_T12000_1$tau[2001:12001], type = "l",xlab = "",ylab = "", main = TeX(r'(Ex$^{app}$)',bold = TRUE),cex.axis = 0.9,cex.main=1, ylim=c(65,145))
plot(Approx_SS2_dppG_Tau100_Sigma0.05_NoisyMH_K2_T12000_1$tau[2001:12001], type = "l",xlab = "",ylab = "", main = TeX(r'(NMH$^{app}$ K2)',bold = TRUE),cex.axis = 0.9,cex.main=1, ylim=c(65,145))
plot(SS2_dppG_Tau100_Sigma0.05_ABCMCMC_p0.005_T12000_1$tau[2001:12001], type = "l",xlab = "",ylab = "", main = "ABC p0.5",cex.axis = 0.9,cex.main=1, ylim=c(65,145))

plot(SS2_dppG_Tau100_Sigma0.05_MH_T12000_1$sigma[2001:12001], type = "l",xlab = "",ylab = "", main = "MH",cex.axis = 0.9,cex.main=1, ylim=c(0.018,0.063))
plot(SS2_dppG_Tau100_Sigma0.05_NoisyMH_K1_T12000_1$sigma[2001:12001], type = "l",xlab = "",ylab = "", main = "Ex",cex.axis = 0.9,cex.main=1, ylim=c(0.018,0.063))
plot(SS2_dppG_Tau100_Sigma0.05_NoisyMH_K2_T12000_1$sigma[2001:12001], type = "l",xlab = "",ylab = "", main = "NMH K2",cex.axis = 0.9,cex.main=1, ylim=c(0.018,0.063))
plot(Approx_SS2_dppG_Tau100_Sigma0.05_NoisyMH_K1_T12000_1$sigma[2001:12001], type = "l",xlab = "",ylab = "", main = TeX(r'(Ex$^{app}$)',bold = TRUE),cex.axis = 0.9,cex.main=1, ylim=c(0.018,0.063))
plot(Approx_SS2_dppG_Tau100_Sigma0.05_NoisyMH_K2_T12000_1$sigma[2001:12001], type = "l",xlab = "",ylab = "", main = TeX(r'(NMH$^{app}$ K2)',bold = TRUE),cex.axis = 0.9,cex.main=1, ylim=c(0.018,0.063))
plot(SS2_dppG_Tau100_Sigma0.05_ABCMCMC_p0.005_T12000_1$sigma[2001:12001], type = "l",xlab = "",ylab = "", main = "ABC p0.5",cex.axis = 0.9,cex.main=1, ylim=c(0.018,0.063))
par(mfrow=c(1,1),mai = c(1.02, 0.82, 0.82, 0.42),mgp=c(3,1,0))
```

The posterior density plot Figure $6$ can be recovered by:

```r
par(mfrow=c(1,2),mai = c(0.25, 0.25, 0.25, 0.05),mgp=c(1.25,0.25,0))
plot(density(SS2_dppG_Tau100_Sigma0.05_MH_T120000_1$tau[20001:120001],bw = 4),xlab = "",ylab="", main = TeX(r'($\tau$ Posterior Density)'),cex.main=0.8,cex.lab = 0.8,cex.axis = 0.6)
lines(density(SS2_dppG_Tau100_Sigma0.05_MH_T12000_1$tau[2001:12001],bw = 4), col = 2)
lines(density(SS2_dppG_Tau100_Sigma0.05_NoisyMH_K1_T12000_1$tau[2001:12001],bw = 4), col = 3)
lines(density(SS2_dppG_Tau100_Sigma0.05_NoisyMH_K2_T12000_1$tau[2001:12001],bw = 4), col = 4)
lines(density(Approx_SS2_dppG_Tau100_Sigma0.05_NoisyMH_K1_T12000_1$tau[2001:12001],bw = 4), col = 5)
lines(density(Approx_SS2_dppG_Tau100_Sigma0.05_NoisyMH_K2_T12000_1$tau[2001:12001],bw = 4), col = 6)
lines(density(SS2_dppG_Tau100_Sigma0.05_ABCMCMC_p0.01_T12000_1$tau[2001:12001],bw = 4), col = 7)
lines(density(SS2_dppG_Tau100_Sigma0.05_ABCMCMC_p0.005_T12000_1$tau[2001:12001],bw = 4), col = 8)
legend("topright", legend=c("GT","MH","Ex","NMH K2",TeX(r'(Ex$^{app}$)'),TeX(r'(NMH$^{app}$ K2)'),"ABC p1","ABC p0.5"),
       col=c(1:8), lty = 1, cex=0.6)

plot(density(SS2_ApproxdppG_Tau100_Sigma0.05_MH_T120000_1$sigma[20001:120001],bw = 0.002),ylim=c(0,90),xlab = "",ylab="", main = TeX(r'($\sigma$ Posterior Density)'),cex.main=0.8,cex.lab = 0.8,cex.axis = 0.6)
lines(density(SS2_dppG_Tau100_Sigma0.05_MH_T12000_1$sigma[2001:12001],bw = 0.002), col = 2)
lines(density(SS2_dppG_Tau100_Sigma0.05_NoisyMH_K1_T12000_1$sigma[2001:12001],bw = 0.002), col = 3)
lines(density(SS2_dppG_Tau100_Sigma0.05_NoisyMH_K2_T12000_1$sigma[2001:12001],bw = 0.002), col = 4)
lines(density(Approx_SS2_dppG_Tau100_Sigma0.05_NoisyMH_K1_T12000_1$sigma[2001:12001],bw = 0.002), col = 5)
lines(density(Approx_SS2_dppG_Tau100_Sigma0.05_NoisyMH_K2_T12000_1$sigma[2001:12001],bw = 0.002), col = 6)
lines(density(SS2_dppG_Tau100_Sigma0.05_ABCMCMC_p0.01_T12000_1$sigma[2001:12001],bw = 0.002), col = 7)
lines(density(SS2_dppG_Tau100_Sigma0.05_ABCMCMC_p0.005_T12000_1$sigma[2001:12001],bw = 0.002), col = 8)
legend("topleft", legend=c("GT","MH","Ex","NMH K2",TeX(r'(Ex$^{app}$)'),TeX(r'(NMH$^{app}$ K2)'),"ABC p1","ABC p0.5"),
       col=c(1:8), lty = 1, cex=0.6)
par(mfrow=c(1,1),mai = c(1.02, 0.82, 0.82, 0.42),mgp=c(3,1,0))
```

## Real Data Application

Recall here that we apply the algorithm comparisons by fitting the Strauss point process model to the Duke Forest dataset processed by [Shirota and Gelfand (2017)](https://doi.org/10.1080/10618600.2017.1299627).
The dataset can be found in the link above as well as the `RealF_Data.csv` provided with this GitHub page.
The data can be loaded by the code:

```r
# Real Duke Forest dataset processed by Shirota and Gelfand (2017)
duke_forest<-data.frame(read.csv("RealF_Data.csv",header=T))
colnames(duke_forest) <- c("x","y")
```

The profile pseudo-likelihood method is applied in the similar way as in [Shirota and Gelfand (2017)](https://doi.org/10.1080/10618600.2017.1299627).
The estimated $\hat{R}=0.053$.

```r
## profile pseudo-likelihood method, i.e. maximum pseudo-likelihood calculated at r
RDA_SPP_pplmStrauss <- profilepl(data.frame(r=seq(0.01,0.1, by=0.001)), Strauss, ppp(duke_forest$x,duke_forest$y)) # The same setting as Shirota & Gelfand (2017)
RDA_SPP_pplmStrauss$fit
RDA_SPP_R_hat <- RDA_SPP_pplmStrauss$fit$interaction$par$r
plot(RDA_SPP_pplmStrauss$param[,1],RDA_SPP_pplmStrauss$prof,type = "l",xlab = "R",ylab = "log PL")
abline(v=RDA_SPP_pplmStrauss$fit$interaction$par$r,col = 2,lty = 2)
```

The plots of the tree positions of the real dataset and the profile pseudo-likelihood shown as Figure $7$ in real data application (RDA) section $7$ of the paper can be recovered by the following code.

```r
par(mfrow=c(1,2),mai = c(0.5, 0.5, 0.25, 0.05),mgp=c(1.25,0.45,0))
plot(duke_forest$x,duke_forest$y,xlab = "",ylab = "")
title(main = "", mgp=c(1,0.25,0),cex.main=1,cex.lab = 0.8)

plot(RDA_SPP_pplmStrauss$param[,1],RDA_SPP_pplmStrauss$prof,type = "l",xlab = "R",ylab = "log PL")
abline(v=RDA_SPP_pplmStrauss$fit$interaction$par$r,col = 2,lty = 2)
title(main = "", mgp=c(1,0.25,0),cex.main=1,cex.lab = 0.6)
par(xpd=TRUE)
text(0.053,204, TeX(r'($\hat{R}=0.053$)'), pos = 4,col=2)
par(xpd=FALSE)
par(mfrow=c(1,1),mai = c(1.02, 0.82, 0.82, 0.42),mgp=c(3,1,0))
```

In this real data application we apply the same experiments as we implemented in the SPP simulation study.
In order to make everything clear enough, we modify the functions used in SPP simulation study and obtain the corresponding specific functions for the real data applications.
The function `df_SPP_Parallel_Noisy_MH()` is the exchange or noisy M-H algorithm implementation of fitting SPP model to the real Duke Forest (df) dataset with parallel computation.
The function `df.S.G.Parallel.ABC.MCMC.Strauss()` is the ABC-MCMC implementation with approximate parallel computation.
If we compare with the functions `SPP_Parallel_Noisy_MH()` and `S.G.ABC.MCMC.Strauss.repeat.draws()`, the only difference is the prior and bound proposal settings, that is, $\pi(\beta)=\text{U}(50,350)$ and $\pi(\gamma)=\text{U}(0,1)$.

The ground truth is to implement exchange algorithm for $1,200,000$ iterations with $200,000$-iteration burn-in.
The initial states of all the implementations below are set as $\beta_0=190,\gamma_0=0.2$, and the proposal epsilons are tuned to be $\epsilon_{\beta}=50, \epsilon_{\gamma}=0.23$.

```r
# Exchange Ground truth
cl <- parallel::makeCluster(detectCores()[1]-1)
clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel computation
time_start <- Sys.time()
RDA_SPP_NoisyMH_K1_T1200000_1 <-
  df_Strauss_Parallel_Noisy_MH(Y=duke_forest,beta0=190, gamma0=0.2,eps_beta=50, eps_gamma=0.23, R=RDA_SPP_R_hat, K=1, T=1200000)
time_end <- Sys.time()
RDA_SPP_NoisyMH_K1_T1200000_1_time <- time_end-time_start
# stopCluster(cl)
# # Time difference of 1.411561 hours
```

### The RDA Implementations of the Exchange and Noisy M-H Algorithms

The implementations of the algorithm comparisons for the exchange and noisy M-H algorithms are shown below by implementing the function `df_SPP_Parallel_Noisy_MH()` for $120,000$ iterations from $K=1$ to $K=8$.

```r
# # Exchange == Noisy MH K1 0.12 million iterations
# cl <- parallel::makeCluster(detectCores()[1]-1)
# clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel computation
time_start <- Sys.time()
RDA_SPP_NoisyMH_K1_T120000_1 <-
  df_SPP_Parallel_Noisy_MH(Y=duke_forest,beta0=190, gamma0=0.2,eps_beta=50, eps_gamma=0.23, R=RDA_SPP_R_hat, K=1, T=120000)
time_end <- Sys.time()
RDA_SPP_NoisyMH_K1_T120000_1_time <- time_end-time_start
# stopCluster(cl)
# # Time difference of 7.484524 mins

# # Noisy MH K2
# cl <- parallel::makeCluster(detectCores()[1]-1)
# clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel computation
time_start <- Sys.time()
RDA_SPP_NoisyMH_K2_T120000_1 <-
  df_SPP_Parallel_Noisy_MH(Y=duke_forest,beta0=190, gamma0=0.2,eps_beta=50, eps_gamma=0.23, R=RDA_SPP_R_hat, K=2, T=120000)
time_end <- Sys.time()
RDA_SPP_NoisyMH_K2_T120000_1_time <- time_end-time_start
# stopCluster(cl)
# # Time difference of 11.58724 mins

# # Noisy MH K3
# cl <- parallel::makeCluster(detectCores()[1]-1)
# clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel computation
time_start <- Sys.time()
RDA_SPP_NoisyMH_K3_T120000_1 <-
  df_SPP_Parallel_Noisy_MH(Y=duke_forest,beta0=190, gamma0=0.2,eps_beta=50, eps_gamma=0.23, R=RDA_SPP_R_hat, K=3, T=120000)
time_end <- Sys.time()
RDA_SPP_NoisyMH_K3_T120000_1_time <- time_end-time_start
# stopCluster(cl)
# # Time difference of 14.10604 mins

# # Noisy MH K4
# cl <- parallel::makeCluster(detectCores()[1]-1)
# clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel computation
time_start <- Sys.time()
RDA_SPP_NoisyMH_K4_T120000_1 <-
  df_SPP_Parallel_Noisy_MH(Y=duke_forest,beta0=190, gamma0=0.2,eps_beta=50, eps_gamma=0.23, R=RDA_SPP_R_hat, K=4, T=120000)
time_end <- Sys.time()
RDA_SPP_NoisyMH_K4_T120000_1_time <- time_end-time_start
# stopCluster(cl)
# # Time difference of 16.94333 mins

# # Noisy MH K5
# cl <- parallel::makeCluster(detectCores()[1]-1)
# clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel computation
time_start <- Sys.time()
RDA_SPP_NoisyMH_K5_T120000_1 <-
  df_SPP_Parallel_Noisy_MH(Y=duke_forest,beta0=190, gamma0=0.2,eps_beta=50, eps_gamma=0.23, R=RDA_SPP_R_hat, K=5, T=120000)
time_end <- Sys.time()
RDA_SPP_NoisyMH_K5_T120000_1_time <- time_end-time_start
# stopCluster(cl)
# # Time difference of 18.52944 mins

# # Noisy MH K6
# cl <- parallel::makeCluster(detectCores()[1]-1)
# clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel computation
time_start <- Sys.time()
RDA_SPP_NoisyMH_K6_T120000_1 <-
  df_SPP_Parallel_Noisy_MH(Y=duke_forest,beta0=190, gamma0=0.2,eps_beta=50, eps_gamma=0.23, R=RDA_SPP_R_hat, K=6, T=120000)
time_end <- Sys.time()
RDA_SPP_NoisyMH_K6_T120000_1_time <- time_end-time_start
# stopCluster(cl)
# # Time difference of 20.82985 mins

# # Noisy MH K7
# cl <- parallel::makeCluster(detectCores()[1]-1)
# clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel computation
time_start <- Sys.time()
RDA_SPP_NoisyMH_K7_T120000_1 <-
  df_SPP_Parallel_Noisy_MH(Y=duke_forest,beta0=190, gamma0=0.2,eps_beta=50, eps_gamma=0.23, R=RDA_SPP_R_hat, K=7, T=120000)
time_end <- Sys.time()
RDA_SPP_NoisyMH_K7_T120000_1_time <- time_end-time_start
# stopCluster(cl)
# # Time difference of 22.83759 mins

# # Noisy MH K8
# cl <- parallel::makeCluster(detectCores()[1]-1)
# clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel computation
time_start <- Sys.time()
RDA_SPP_NoisyMH_K8_T120000_1 <-
  df_SPP_Parallel_Noisy_MH(Y=duke_forest,beta0=190, gamma0=0.2,eps_beta=50, eps_gamma=0.23, R=RDA_SPP_R_hat, K=8, T=120000)
time_end <- Sys.time()
RDA_SPP_NoisyMH_K8_T120000_1_time <- time_end-time_start
stopCluster(cl)
# Time difference of 39.15605 mins
```

### The RDA Implementation of the ABC-MCMC Algorithm

The pilot run for the real data applications is shown below.
The process is exactly the same as SS1 pilot run except the prior settings.
Here we consider $p=0.025, 0.01$ and $0.005$ again in this experiment.

```r
# obtain Kfunc for Y
RDA_SPP_Kfunc_Obs=as.function(Kest(ppp(duke_forest$x,duke_forest$y), correction="isotropic"))
RDA_SPP_R_hat <- RDA_SPP_pplmStrauss$fit$interaction$par$r
RDA_SPP_Kfunc_Obs_R_hat <- RDA_SPP_Kfunc_Obs(RDA_SPP_R_hat)
RDA_SPP_Pilot.L <- 10000

# ABC-MCMC Pilot Draws Function
df_ABCMCMC_Pilot_lth_Draw_SPP <- function(x, N_Y, R, Kfunc_Obs_R_hat){ # Current state beta and gamma
  beta=runif(1,50,350)
  gamma=runif(1,0,1)
  X=rStrauss(beta,gamma,R,square(1))
  Kfunc_X=as.function(Kest(X, correction="isotropic"))
  eta <- c(log(X$n)-log(N_Y),(sqrt(Kfunc_X(R))-sqrt(Kfunc_Obs_R_hat))^2)
  return(list(beta=beta,gamma=gamma,X=X,eta=eta))
}

# Implement pilot run by parallel computation
cl <- parallel::makeCluster(detectCores()[1]-1)
clusterExport(cl=cl, list("rStrauss", "square","Kest")) # In order to use this function for parallel computation
time_start <- Sys.time()
RDA_SPP_Pilot <- parLapply(cl, 1:RDA_SPP_Pilot.L, df_ABCMCMC_Pilot_lth_Draw_SPP, N_Y=ppp(duke_forest$x,duke_forest$y)$n, R=RDA_SPP_R_hat,
                           Kfunc_Obs_R_hat = RDA_SPP_Kfunc_Obs_R_hat)
time_end <- Sys.time()
RDA_SPP_Pilot.time <- time_end-time_start
# Time difference of 37.5737 secs
stopCluster(cl)
# Transform pilot draws
RDA_SPP_Pilot.beta <- c()
RDA_SPP_Pilot.gamma <- c()
RDA_SPP_Pilot.X <- list()
RDA_SPP_Pilot.eta <- matrix(0,RDA_SPP_Pilot.L,2)
for(l in 1:RDA_SPP_Pilot.L){
  RDA_SPP_Pilot.beta[l]=RDA_SPP_Pilot[[l]]$beta
  RDA_SPP_Pilot.gamma[l]=RDA_SPP_Pilot[[l]]$gamma
  RDA_SPP_Pilot.X[[l]]=RDA_SPP_Pilot[[l]]$X
  RDA_SPP_Pilot.eta[l,] <- RDA_SPP_Pilot[[l]]$eta
}

# apply glmnet for regression, i.e. glm with lasso and determine the penalty parameter for the lasso by cross-validation (cv)
library(glmnet)
RDA_SPP_Pilot.lmCoef <-
  coef(cv.glmnet(x=RDA_SPP_Pilot.eta,y=log(cbind(RDA_SPP_Pilot.beta,RDA_SPP_Pilot.gamma)),family="mgaussian",alpha=1),s="lambda.min")
RDA_SPP_Pilot.lmCoefBeta <- as.matrix(RDA_SPP_Pilot.lmCoef$RDA_SPP_Pilot.beta) # store the coefficients
RDA_SPP_Pilot.lmCoefGamma <- as.matrix(RDA_SPP_Pilot.lmCoef$RDA_SPP_Pilot.gamma)
RDA_SPP_Pilot.VarBeta=c(var(cbind(1,RDA_SPP_Pilot.eta)%*%RDA_SPP_Pilot.lmCoefBeta)) # calculate variance of log(theta)^hat
RDA_SPP_Pilot.VarGamma=c(var(cbind(1,RDA_SPP_Pilot.eta)%*%RDA_SPP_Pilot.lmCoefGamma))
RDA_SPP_Pilot.psi <- ((RDA_SPP_Pilot.eta%*%RDA_SPP_Pilot.lmCoefBeta[2:3])^2)/RDA_SPP_Pilot.VarBeta +
  ((RDA_SPP_Pilot.eta%*%RDA_SPP_Pilot.lmCoefGamma[2:3])^2)/RDA_SPP_Pilot.VarGamma
# Take p percentile
RDA_SPP_Pilot.0.005eps <- quantile(RDA_SPP_Pilot.psi,probs=0.005)[[1]]
RDA_SPP_Pilot.0.01eps <- quantile(RDA_SPP_Pilot.psi,probs=0.01)[[1]]
RDA_SPP_Pilot.0.025eps <- quantile(RDA_SPP_Pilot.psi,probs=0.025)[[1]]
```

The implementations of the main ABC-MCMC algorithm for all the three different $p$ cases are following.

```r
## ABC-MCMC algorithm with approximate parallel computation p0.025
NumCores <- detectCores()[1]-1
cl <- parallel::makeCluster(NumCores)
clusterExport(cl=cl, list("rStrauss", "square", "Kest")) # In order to use this function for parallel computation
time_start <- Sys.time()
RDA_SPP_ABCMCMC_p0.025_T120000_1 <-
  df.S.G.Parallel.ABC.MCMC.Strauss(Y = ppp(duke_forest$x,duke_forest$y), beta0=190, gamma0=0.2,eps_beta=50, eps_gamma=0.23,
                                   lmCoefBeta = RDA_SPP_Pilot.lmCoefBeta, lmCoefGamma = RDA_SPP_Pilot.lmCoefGamma,
                                   Pilot.VarBeta = RDA_SPP_Pilot.VarBeta, Pilot.VarGamma = RDA_SPP_Pilot.VarGamma,
                                   eps = RDA_SPP_Pilot.0.025eps, R=RDA_SPP_R_hat, T=120000)
time_end <- Sys.time()
RDA_SPP_ABCMCMC_p0.025_T120000_1_time <- time_end-time_start
stopCluster(cl)
# Time difference of 2.967832 hours

## ABC-MCMC algorithm with approximate parallel computation p0.01
NumCores <- detectCores()[1]-1
cl <- parallel::makeCluster(NumCores)
clusterExport(cl=cl, list("rStrauss", "square", "Kest")) # In order to use this function for parallel computation
time_start <- Sys.time()
RDA_SPP_ABCMCMC_p0.01_T120000_1 <-
  df.S.G.Parallel.ABC.MCMC.Strauss(Y = ppp(duke_forest$x,duke_forest$y), beta0=190, gamma0=0.2,eps_beta=50, eps_gamma=0.23,
                                   lmCoefBeta = RDA_SPP_Pilot.lmCoefBeta, lmCoefGamma = RDA_SPP_Pilot.lmCoefGamma,
                                   Pilot.VarBeta = RDA_SPP_Pilot.VarBeta, Pilot.VarGamma = RDA_SPP_Pilot.VarGamma,
                                   eps = RDA_SPP_Pilot.0.01eps, R=RDA_SPP_R_hat, T=120000)
time_end <- Sys.time()
RDA_SPP_ABCMCMC_p0.01_T120000_1_time <- time_end-time_start
stopCluster(cl)
# Time difference of 6.25194 hours

## ABC-MCMC algorithm with approximate parallel computation p0.005
NumCores <- detectCores()[1]-1
cl <- parallel::makeCluster(NumCores)
clusterExport(cl=cl, list("rStrauss", "square", "Kest")) # In order to use this function for parallel computation
time_start <- Sys.time()
RDA_SPP_ABCMCMC_p0.005_T120000_1 <-
  df.S.G.Parallel.ABC.MCMC.Strauss(Y = ppp(duke_forest$x,duke_forest$y), beta0=190, gamma0=0.2,eps_beta=50, eps_gamma=0.23,
                                   lmCoefBeta = RDA_SPP_Pilot.lmCoefBeta, lmCoefGamma = RDA_SPP_Pilot.lmCoefGamma,
                                   Pilot.VarBeta = RDA_SPP_Pilot.VarBeta, Pilot.VarGamma = RDA_SPP_Pilot.VarGamma,
                                   eps = RDA_SPP_Pilot.0.005eps, R=RDA_SPP_R_hat, T=120000)
time_end <- Sys.time()
RDA_SPP_ABCMCMC_p0.005_T120000_1_time <- time_end-time_start
stopCluster(cl)
# Time difference of 11.55707 hours
```

The box plots Figure $8$ in section $7$ can be recovered by the following code.

```r
par(mfrow=c(1,2),mai = c(0.3, 0.3, 0.05, 0.01),mgp=c(0.45,0.3,0))
boxplot(RDA_SPP_NoisyMH_K1_T1200000_1$beta[200001:1200001],
        RDA_SPP_NoisyMH_K1_T120000_1$beta[20001:120001],
        RDA_SPP_NoisyMH_K2_T120000_1$beta[20001:120001],
        RDA_SPP_NoisyMH_K3_T120000_1$beta[20001:120001],
        RDA_SPP_NoisyMH_K4_T120000_1$beta[20001:120001],
        RDA_SPP_NoisyMH_K5_T120000_1$beta[20001:120001],
        RDA_SPP_NoisyMH_K6_T120000_1$beta[20001:120001],
        RDA_SPP_NoisyMH_K7_T120000_1$beta[20001:120001],
        RDA_SPP_NoisyMH_K8_T120000_1$beta[20001:120001],
        RDA_SPP_ABCMCMC_p0.025_T120000_1$beta[20001:120001],
        RDA_SPP_ABCMCMC_p0.01_T120000_1$beta[20001:120001],
        RDA_SPP_ABCMCMC_p0.005_T120000_1$beta[20001:120001],
        xlab = "",ylab = "", main = "",cex.axis = 0.6)
title(xlab = "",ylab = TeX(r'($\beta$)'), main = "", mgp=c(0.45,0.3,0),cex.main=1,cex.lab = 0.8)
axis(1, at=c(1:12), labels = c("GT","Ex","K2","K3","K4","K5","K6","K7","K8","p2.5","p1","p0.5"),cex.axis=0.6)
abline(h=median(RDA_SPP_NoisyMH_K1_T1200000_1$beta[200001:1200001]),col = 2,lty = 2)

boxplot(RDA_SPP_NoisyMH_K1_T1200000_1$gamma[200001:1200001],
        RDA_SPP_NoisyMH_K1_T120000_1$gamma[20001:120001],
        RDA_SPP_NoisyMH_K2_T120000_1$gamma[20001:120001],
        RDA_SPP_NoisyMH_K3_T120000_1$gamma[20001:120001],
        RDA_SPP_NoisyMH_K4_T120000_1$gamma[20001:120001],
        RDA_SPP_NoisyMH_K5_T120000_1$gamma[20001:120001],
        RDA_SPP_NoisyMH_K6_T120000_1$gamma[20001:120001],
        RDA_SPP_NoisyMH_K7_T120000_1$gamma[20001:120001],
        RDA_SPP_NoisyMH_K8_T120000_1$gamma[20001:120001],
        RDA_SPP_ABCMCMC_p0.025_T120000_1$gamma[20001:120001],
        RDA_SPP_ABCMCMC_p0.01_T120000_1$gamma[20001:120001],
        RDA_SPP_ABCMCMC_p0.005_T120000_1$gamma[20001:120001],
        xlab = "",ylab = "", main = "",cex.axis = 0.6)
title(xlab = "",ylab = TeX(r'($\gamma$)'), main = "", mgp=c(0.45,0.3,0),cex.main=1,cex.lab = 0.8)
axis(1, at=c(1:12), labels = c("GT","Ex","K2","K3","K4","K5","K6","K7","K8","p2.5","p1","p0.5"),cex.axis=0.6)
abline(h=median(RDA_SPP_NoisyMH_K1_T1200000_1$gamma[200001:1200001]),col = 2,lty = 2)
par(mfrow=c(1,1),mai = c(1.02, 0.82, 0.82, 0.42),mgp=c(3,1,0))
```

The corresponding density plots Figure $9$ of the paper can be recovered by:

```r
par(mfrow=c(1,2),mai = c(0.25, 0.25, 0.25, 0.05),mgp=c(1.25,0.25,0))
plot(density(RDA_SPP_NoisyMH_K1_T1200000_1$beta[200001:1200001],bw = 5),xlab = "",ylab="",xlim=c(50,250),ylim=c(0,0.018), main = TeX(r'($\beta$ Posterior Density)'),cex.main=0.8,cex.lab = 0.8,cex.axis = 0.6)
lines(density(RDA_SPP_NoisyMH_K1_T120000_1$beta[20001:120001],bw = 5),col=2)
lines(density(RDA_SPP_NoisyMH_K2_T120000_1$beta[20001:120001],bw = 5),col=3)
lines(density(RDA_SPP_ABCMCMC_p0.005_T120000_1$beta[20001:120001],bw = 5),col = 4)
legend("topright", legend=c("GT","Ex","NMH K2","ABC p0.5"),
       col=c(1:4), lty = 1, cex=0.6)

plot(density(RDA_SPP_NoisyMH_K1_T1200000_1$gamma[200001:1200001],bw = 0.015),xlab = "",ylab="",ylim=c(0,4), main = TeX(r'($\gamma$ Posterior Density)'),cex.main=0.8,cex.lab = 0.8,cex.axis = 0.6)
lines(density(RDA_SPP_NoisyMH_K1_T120000_1$gamma[20001:120001],bw = 0.015),col=2)
lines(density(RDA_SPP_NoisyMH_K2_T120000_1$gamma[20001:120001],bw = 0.015),col=3)
lines(density(RDA_SPP_ABCMCMC_p0.005_T120000_1$gamma[20001:120001],bw = 0.015),col = 4)
legend("topright", legend=c("GT","Ex","NMH K2","ABC p0.5"),
       col=c(1:4), lty = 1, cex=0.6)
par(mfrow=c(1,1),mai = c(1.02, 0.82, 0.82, 0.42),mgp=c(3,1,0))
```

If we construct the Table $3$ shown in section $7$ of the paper in the `R` code, and store it as `RDA_SPP_comparison_table` in the enviroment, the Figure $10$ of the paper can be recovered by the following code.

```r
par(mfrow=c(1,2),mai = c(0.25, 0.25, 0.25, 0.05),mgp=c(1.25,0.25,0))
plot(RDA_SPP_comparison_table[1,5:12],type = "b",ylim=c(0,9000), pch = 20,xaxt="n",xlab = "",ylab = "")
lines(RDA_SPP_comparison_table[8,5:12],type = "b",col=2, pch = 20)
lines(RDA_SPP_comparison_table[9,5:12],type = "b",col=3, pch = 20)
title(xlab = "",ylab = "", main = "Time and ESS plots of Ex and NMH Cases", mgp=c(0.25,0.25,0),cex.main=0.8,cex.lab = 0.6)
axis(1, at=c(1:8), labels = c("Ex","K2","K3","K4","K5","K6","K7","K8"),cex.axis=0.5)
legend("topleft", legend=c("Time", TeX(r'(ESS($\beta$))'), TeX(r'(ESS($\gamma$))')),col=1:3, lty=1, cex=0.6)
# The above line is correct but the Rstudio could not recognize

plot(RDA_SPP_comparison_table[10,5:12],type = "b", pch = 20,ylim=c(0,11),xaxt="n",xlab = "",ylab = "")
abline(h=RDA_SPP_comparison_table[10,2],col = 2,lty = 2)
abline(h=RDA_SPP_comparison_table[10,3],col = 3,lty = 2)
abline(h=RDA_SPP_comparison_table[10,4],col = 4,lty = 2)
title(xlab = "",ylab = "", main = "ESS(Ave)/s Comparison", mgp=c(0.25,0.25,0),cex.main=0.8,cex.lab = 0.6)
axis(1, at=c(1:8), labels = c("Ex","K2","K3","K4","K5","K6","K7","K8"),cex.axis=0.5)
legend("topright", legend=c("NMH Different K Cases", "ABC-MCMC p2.5", "ABC-MCMC p1", "ABC-MCMC p0.5"),
       col=1:4, lty=c(1,2,2,2), cex=0.6)
par(mfrow=c(1,1),mai = c(1.02, 0.82, 0.82, 0.42),mgp=c(3,1,0))
```

Since the code for constructing the table `RDA_SPP_comparison_table` requires hundreds of lines to summarize the outputs and to integrate all the summary statistics together.
We propose not to provide more details here in order not to make this page too lengthy.





