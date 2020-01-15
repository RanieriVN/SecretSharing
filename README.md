# SecretSharing 

### Codes accompanying the paper "Device-independent secret sharing and a stronger form of Bell nonlocality" ([arXiv:1909.11785 [quant-ph]](https://arxiv.org/abs/1909.11785))

**FindQuantumBehavior.py** computes an upper bound for Eve's quantum guessing probability as a function of the violation of a given tripartite Bell-like inequality. This is done by approximating the problem by an SDP through the NPA hierarchy. It is currently configured to use the level 2 of the hierarchy with extra projections ABC, ABD, BCD and ABCD and Sveltichny's inequality as the Bell-like inequality.

**OptimGuess_QuantumReduced_Visibility.py** computes the upper bound by considering a specific marginal distribution for Alice, Bob and Charlie. The guessing probability is then given as a function of the visibility of the state used to compute the marginal distribution. Here also the code is configured to use the level 2 of the NPA hierarchy with extra projections ABC, ABD, BCD and ABCD.

Both codes use Peter Wittek's ncpol2sdpa library (found [here](https://github.com/peterwittek/ncpol2sdpa)) and are currently configured to use mosek (found [here](https://www.mosek.com/)) as solver for the SDPs.
