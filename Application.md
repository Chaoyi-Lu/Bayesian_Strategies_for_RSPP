# Simulation Studies and Real Data Application
This file illustrates the code and the corresponding applications associated to the outputs shown in the *Bayesian Strategies for Repulsive Spatial Point Processes* paper.
The code of both two simulation studies and the real data application is provided.

The source code is included in the [`Algorithm_Functions_for_RSPP.R`] and can be loaded by the following code.

``` r
rm(list=ls())

source("Algorithm_Functions_for_RSPP.R")
```

The explanations of each function and almost each line of the code in [`Algorithm_Functions_for_RSPP.R`] are provided in the corresponding comments in the file.
Note that the function `Noisy_E_nth_Ratio()` corresponds to the $n$' th auxiliary draw of the noisy Metropolis-Hastings (noisy M-H) algorithm as well as the corresponding evaluation of the unnormalised likelihood ratio $\frac{q(x_n'|\theta^{(t-1)})}{q(x_n'|\theta')}$ where we use notation $N$ instead of $K$ to denote the number of auxiliary draws for the noisy M-H algorithm in the code.

The function `SPP_Parallel_Noisy_MH()` is the noisy M-H algorithm implemented for the Strauss point process in the simulation study.
Note further that, by setting $N=1$, the algorithm becomes the exchange algorithm.
The parallel computation is implemented for the $N$ auxiliary draws.

The function `S.G.ABC.MCMC.Strauss.repeat.draws()` implements one round of draw in the `repeat` loop of the ABC-MCMC algorithm proposed by [Shirota and Gelfand (2017)](https://doi.org/10.1080/10618600.2017.1299627) and returns the corresponding indicator of whether the $\Psi(\hat{\theta}',\hat{a})$ is smaller than the acceptance threshold $\epsilon$ as well as the corresponding proposed states.

