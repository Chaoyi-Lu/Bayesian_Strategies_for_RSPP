# Bayesian Strategies for Repulsive Spatial Point Processes

This repository is a complementary material for the paper [**Bayesian Strategies for Repulsive Spatial Point Processes**](https://arxiv.org/abs/2404.15133) to provide the code for all the implementations including two simulation studies and one real data application therein.
This paper explores the **exchange algorithm** and the **noisy Metropolis-Hastings algorithm** implemented on **repulsive spatial point processes**.
We also develop a **noval ABC-MCMC algorithm** based on the correction of the [**Shirota and Gelfand (2017)**](https://www.tandfonline.com/doi/full/10.1080/10618600.2017.1299627) **ABC-MCMC algorithm**, and we explore the performance of such a newly proposed algorithm by comparing to the [**Fearnhead and Prangle (2012)**](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2011.01010.x) **ABC-MCMC algorithm**.
The comparisons between the candidate algorithms outlined about are also provided.

The file [`Application.md`] is the main tutorial notes for the simulation studies and the real data applications.
The file [`Algorithm_Functions_for_RSPP.R`] contains all the [`R`] functions we need to use.
The files [`SS1_SPP_Beta200_Gamma0.1_R0.05_ObsY.csv`], [`SS2_dppG_Tau100_Sigma0.05_ObsY.csv`] and [`RealF_Data.csv`], respectively, contains the dataset we implement on for the simulation studies $1$, $2$ and the real data application.
