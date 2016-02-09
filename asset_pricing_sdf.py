import numpy as np
from mult_functional import MultiplicativeFunctional, _Result


class AssetPricingSDF(object):
    """
    Class representing the asset pricing model with stochastic discount
    factors.

    Parameters
    ----------
    mc : MarkovChain
        MarkovChain instance with n states representing the `X` process.

    G_s : array_like(float, ndim=2)
        Discount rate matrix. Must be of shape n x n.

    G_d : array_like(float, ndim=2)
        Growth rate matrix for the dividend. Must be of shape n x n.

    d_inits : array_like(float, ndim=1), optional(default=None)
        Array containing the initial values of the dividend, one for
        each state. Must be of length n. If not specified, default to
        the vector of all ones.

    Attributes
    ----------
    mc : MarkovChain
        See Parameters.

    G_s, G_d : ndarray(float, ndim=2)
        See Parameters.

    d_inits: ndarray(float, ndim=1)
        See Parameters.

    n : scalar(int)
        Number of the state.

    P : ndarray(float, ndim=2)
        Transition probability matrix of `mc`.

    mf_s : MultiplicativeFunctional
        MultiplicativeFunctional instance for the stochastic discount
        factor process.

    mf_d : MultiplicativeFunctional
        MultiplicativeFunctional instance for the dividend process.

    v : ndarray(float, ndim=1)
        Dividend-price ratios.

    """
    def __init__(self, mc, G_s, G_d, d_inits=None):
        self.mc = mc
        self.n = mc.n
        self.P = mc.P

        self.mf_s = MultiplicativeFunctional(self.mc, G_s)
        self.mf_d = MultiplicativeFunctional(self.mc, G_d, M_inits=d_inits)

        self.P_check = self.P * self.mf_s.M_matrix
        self.P_tilde = self.P_check * self.mf_d.M_matrix

        # Solve the linear equation v = P_tilde v + P_check 1
        A = np.identity(self.n) - self.P_tilde
        b = self.P_check.dot(np.ones(self.n))
        self.v = np.linalg.solve(A, b)

    def simulate(self, ts_length, X_init=None,
                 num_reps=None, random_state=None):
        """
        Simulate the discount factor and dividend processes.

        Parameters
        ----------
        ts_length : scalar(int)
            Length of each simulation.

        X_init : scalar(int), optional(default=None)
            Initial state of the `X` process. If None, the initial state
            is randomly drawn.

        num_reps : scalar(int), optional(default=None)
            Number of repetitions of simulation.

        random_state : scalar(int) or np.random.RandomState,
                       optional(default=None)
            Random seed (integer) or np.random.RandomState instance to
            set the initial state of the random number generator for
            reproducibility. If None, a randomly initialized RandomState
            is used.

        Returns
        -------
        res: APSDFSimulateResult
            Simulation result represetned as a `APSDFSimulateResult`.
            See `APSDFSimulateResult` for details. The array for each
            attribute is of shape `(ts_length,)` if `num_reps=None`, or
            of shape `(num_reps, ts_length)` otherwise.

        """
        X = self.mc.simulate(ts_length, init=X_init,
                             num_reps=num_reps, random_state=random_state)
        return self.generate_paths(X)

    def generate_paths(self, X):
        """
        Given a simulation of the `X` process, generate sample paths of
        the discount factor and dividend processes.

        Parameters
        ----------
        X : array_like(int)
            Array containing the sample path(s) of the `X` process.

        Returns
        -------
        res: APSDFSimulateResult
            Simulation result represetned as a `APSDFSimulateResult`.
            See `APSDFSimulateResult` for details. The array for each
            attribute is of the same shape as `X`.

        """
        res_s = self.mf_s.generate_paths(X)
        res_d = self.mf_d.generate_paths(X)
        s, s_tilde = res_s.M, res_s.M_tilde
        d, d_tilde = res_d.M, res_d.M_tilde
        p = d * self.v[X]
        res = APSDFSimulateResult(X=X,
                                  s=s,
                                  s_tilde=s_tilde,
                                  d=d,
                                  d_tilde=d_tilde,
                                  p=p)
        return res


class APSDFSimulateResult(_Result):
    """
    Contain the information about the simulation result for
    `AssetPricingSDF`.

    Attributes
    ----------
    s : ndarray(float, ndim=1 or 2)
        Array containing the sample path(s) of the discount factor
        process.

    s_tilde : ndarray(float, ndim=1 or 2)
        Array containing the sample path(s) of the multiplicative
        martingale component of the discount factor process.

    d : ndarray(float, ndim=1 or 2)
        Array containing the sample path(s) of the dividend process.

    d_tilde : ndarray(float, ndim=1 or 2)
        Array containing the sample path(s) of the multiplicative
        martingale component of the dividend process.

    p: ndarray(float, ndim=1 or 2)
        Array containing the sample path(s) of the dividend process.

    X : ndarray(int, ndim=1 or 2)
        Array containing the sample path(s) of the `X` process.

    """
    pass
