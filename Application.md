# Simulation Studies and Real Data Application
This file illustrates the code and the corresponding applications associated to the outputs shown in the *Bayesian Strategies for Repulsive Spatial Point Processes* paper.
The code of both two simulation studies and the real data application is provided.

The source code is included in the [`Algorithm_Functions_for_RSPP.R`] and can be called by the following code.

``` r
rm(list=ls())

source("Algorithm_Functions_for_RSPP.R")
```

The explanations of each function and almost each line of the code in [`Algorithm_Functions_for_RSPP.R`] are provided in the corresponding comments in the file.
Note that the function `Noisy_E_nth_Ratio()` corresponds to the $`N`$th auxiliary draw of the noisy Metropolis-Hastings algorithm as well as the corresponding evaluation of the unnormalised likelihood ratio $`\frac{q(x_n'|\theta^{(t-1)})}{q(x_n'|\theta')}`$. 
