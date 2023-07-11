# Simulation Studies and Real Data Application
This file illustrates the code and the corresponding applications associated to the outputs shown in the *Bayesian Strategies for Repulsive Spatial Point Processes* paper.
The code of both two simulation studies and the real data application is provided.

The source code is included in the [`Algorithm_Functions_for_RSPP.R`] and can be loaded together with some required `R` packages by the following code.

``` r
rm(list=ls())
source("Algorithm_Functions_for_RSPP.R")
library(spatstat) # For point processes simulations and related application
library(doParallel) # For parallel computation
library(LaplacesDemon) # For ESS() and logdet() function
```

The explanations of each function and almost each line of the code in [`Algorithm_Functions_for_RSPP.R`] are provided in the corresponding comments in the file.

## Strauss Point Process Simulation Study

Note that the function `Noisy_E_kth_Ratio()` corresponds to the $k$' th auxiliary draw of the noisy Metropolis-Hastings (noisy M-H) algorithm as well as the corresponding evaluation of the unnormalised likelihood ratio $\frac{q(x_n'|\theta^{(t-1)})}{q(x_n'|\theta')}$.

The function `SPP_Parallel_Noisy_MH()` is the noisy M-H algorithm implemented for the Strauss point process (SPP) in the simulation study.
Note further that, by setting $K=1$, the algorithm becomes the exchange algorithm.
The parallel computation is implemented for the $N$ auxiliary draws.

The function `S.G.ABC.MCMC.Strauss.repeat.draws()` implements one round of proposed draw for the SPP in the `repeat` loop of the ABC-MCMC algorithm proposed by [Shirota and Gelfand (2017)](https://doi.org/10.1080/10618600.2017.1299627) and returns the proposed states as well as the indicator of whether the corresponding $\Psi(\hat{\theta}',\hat{a})$ is smaller than the acceptance threshold $\epsilon$.
The function `S.G.Parallel.ABC.MCMC.Strauss()` apply the implementation of the ABC-MCMC algorithm with the approximate parallel computation discussed in section $4$ of the paper.

We start from the process of generating the artificial data used in the SPP simulation study where we set $\beta = 200, \gamma = 0.1$ and $R = 0.05$ on $S = [0,1]\times[0,1]$.

``` r
## Generate from a Strauss point process
Z_observation <- rStrauss(beta = 200, gamma = 0.1, R = 0.05, W = square(1))
# utils::write.table(x=cbind(Z_observation$x,Z_observation$y),file="SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY.csv", sep="," , row.names = FALSE, col.names=FALSE)
```

All the point locations are stored in the file [`SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY.csv`] and can be called directly as follows.

``` r
## Simulation study 1 observation
SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY <- as.matrix(read.csv("SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY.csv",header=FALSE))
SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY  <- ppp(SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY[,1],SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY[,2]) # transform to point pattern class
```

Then we can obtain the estimated interaction radius $R$ by the profile pseudo-likelihood method

``` r
## profile pseudo-likelihood method, i.e. maximum pseudo-likelihood calculated at r
SS1_SPP_pplmStrauss <- profilepl(data.frame(r=seq(0.01,0.1, by=0.0001)), Strauss, SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY)
SS1_SPP_pplmStrauss$fit
SS1_SPP_R_hat <- SS1_SPP_pplmStrauss$fit$interaction$par$r # Store the estimated R
```

And the code below provides the plot of the point pocations and the profile pseudo-likelihood plot shown in SPP simulation study of the paper.

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

The ground truth implementation is to apply the exchange algorithm for $1200000$ iterations as follows.

``` r
# Exchange Ground Truth
cl <- parallel::makeCluster(detectCores()[1]-1)
clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel running
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

Note here that the first element of the chain is the initial state $\theta^{(0)}$ and thus we need to drop the first $200001$ iterations in order for the $200000$ burn-in.

### The Implementation of the Exchange and Noisy M-H Algorithms

Similar implementations are applied for the exchange and noisy M-H algorithms with $120000$ iterations.
The noisy M-H algorithms are implemented from $k=2$ to $k=8$ where the $k=1$ case is equivalent to the exchange algorithm.

``` r
# # Exchange == Noisy M-H K1 0.12 million iterations
# cl <- parallel::makeCluster(detectCores()[1]-1)
# clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel running
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
# clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel running
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
# clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel running
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
# clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel running
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
# clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel running
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
# clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel running
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
# clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel running
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
# clusterExport(cl=cl, list("rStrauss", "square")) # In order to use this function for parallel running
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

### The Implementation of the ABC-MCMC algorithm



