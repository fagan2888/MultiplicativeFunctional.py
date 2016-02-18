"""
Author: Daisuke Oyama

Mudule for simulating asset prices under exponentially growing
consumption endowments where the growth rate is determined by a finite
state Markov chain.

"""
import warnings
import numpy as np
from mult_functional import MultFunctionalFiniteMarkov
from utils import _Result


class AssetPricingMultFiniteMarkov(object):
    """
    Class representing the asset pricing model with stochastic discount
    factor and dividend processes governed by a multiplicative
    functional.

    Parameters
    ----------
    mc : MarkovChain
        MarkovChain instance with n states representing the underlying
        `X` process.

    G_S : array_like(float, ndim=2)
        Discount rate matrix. Must be of shape n x n.

    G_d : array_like(float, ndim=2)
        Growth rate matrix for the dividend. Must be of shape n x n.

    d_inits : array_like(float, ndim=1), optional(default=None)
        Array containing the initial values of the dividend, one for
        each state. Must be of length n. If not specified, default to
        the vector of all ones. If it is a scalar, then it is converted
        to the constant vector of that scalar.

    Attributes
    ----------
    mc : MarkovChain
        See Parameters.

    G_S, G_d : ndarray(float, ndim=2)
        See Parameters.

    d_inits: ndarray(float, ndim=1)
        See Parameters.

    n : scalar(int)
        Number of the state.

    P : ndarray(float, ndim=2)
        Transition probability matrix of `mc`.

    mf_S : MultFunctionalFiniteMarkov
        MultFunctionalFiniteMarkov instance for the stochastic discount
        factor process.

    mf_d : MultFunctionalFiniteMarkov
        MultFunctionalFiniteMarkov instance for the dividend process.

    v : ndarray(float, ndim=1)
        Dividend-price ratios.

    """
    def __init__(self, mc, G_S, G_d, d_inits=None):
        self.mc = mc
        self.n = mc.n
        self.P = mc.P

        self.mf_S = MultFunctionalFiniteMarkov(self.mc, G_S)
        self.mf_d = MultFunctionalFiniteMarkov(self.mc, G_d, M_inits=d_inits)

        self.P_check = self.P * self.mf_S.M_matrix
        self.P_tilde = self.P_check * self.mf_d.M_matrix

        if not self._check_spectral_radius():
            msg = 'P_tilde has an eigenvalue not smaller than one'
            # Ignored by warnings.filterwarnings('ignore')
            # somewhere in qunatecon
            # warnings.warn(msg, UserWarning)
            print('Warning:', msg)

        # Solve the linear equation v = P_tilde v + P_check 1
        A = np.identity(self.n) - self.P_tilde
        b = self.P_check.dot(np.ones(self.n))
        self.v = np.linalg.solve(A, b)

    def _check_spectral_radius(self):
        """
        Check that the eigenvalues of P_tilde are smaller than one.
        Under the premise that P_tilde is nonnegative, this implies that
        I - P_tilde is inverse positive.

        """
        eig_vals, _ = np.linalg.eig(self.P_tilde)
        return (eig_vals < 1).all()

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
        res: APSMFMSimulateResult
            Simulation result represetned as a `APSMFMSimulateResult`.
            See `APSMFMSimulateResult` for details. The array for each
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
        res: APSMFMSimulateResult
            Simulation result represetned as a `APSMFMSimulateResult`.
            See `APSMFMSimulateResult` for details. The array for each
            attribute is of the same shape as `X`.

        """
        res_S = self.mf_S.generate_paths(X)
        res_d = self.mf_d.generate_paths(X)
        S, S_tilde = res_S.M, res_S.M_tilde
        d, d_tilde = res_d.M, res_d.M_tilde
        p = d * self.v[X]
        res = APSMFMSimulateResult(X=X,
                                   S=S,
                                   S_tilde=S_tilde,
                                   d=d,
                                   d_tilde=d_tilde,
                                   p=p)
        return res


class LucasTreeFiniteMarkov(object):
    """
    Class representing the Lucas asset pricing model with finite Markov
    states.

    Parameters
    ----------
    mc : MarkovChain
        MarkovChain instance with n states representing the underlying
        `X` process.

    G_C : array_like(float, ndim=2)
        Growth rate matrix for the consumption endowment. Must be of
        shape n x n.

    C_inits : array_like(float, ndim=1), optional(default=None)
        Array containing the initial values of the endowment, one for
        each state. Must be of length n. If not specified, default to
        the vector of all ones. If it is a scalar, then it is converted
        to the constant vector of that scalar.

    Attributes
    ----------
    mc : MarkovChain
        See Parameters.

    G_C : ndarray(float, ndim=2)
        See Parameters.

    C_inits: ndarray(float, ndim=1)
        See Parameters.

    n : scalar(int)
        Number of the state.

    P : ndarray(float, ndim=2)
        Transition probability matrix of `mc`.

    G_S : ndarray(float, ndim=2)
        Discount rate matrix.

    v : ndarray(float, ndim=1)
        Endowment-price ratios.

    """
    def __init__(self, mc, G_C, gamma, delta, C_inits=None):
        self.mc = mc
        self.n = mc.n
        self.P = mc.P

        self.G_C = np.asarray(G_C)

        # Stochastic discount rate matrix
        self.G_S = -delta - gamma * self.G_C

        self.ap = AssetPricingMultFiniteMarkov(
            mc, self.G_S, self.G_C, d_inits=C_inits
        )
        self.P_check = self.ap.P_check
        self.P_tilde = self.ap.P_tilde
        self.v = self.ap.v

    def simulate(self, ts_length, X_init=None,
                 num_reps=None, random_state=None):
        """
        Simulate the model.

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
        res: APSMFMSimulateResult
            Simulation result represetned as a `APSMFMSimulateResult`.
            See `APSMFMSimulateResult` for details. The array for each
            attribute is of shape `(ts_length,)` if `num_reps=None`, or
            of shape `(num_reps, ts_length)` otherwise.

        """
        return self.ap.simulate(ts_length, X_init, num_reps, random_state)


class APSMFMSimulateResult(_Result):
    """
    Contain the information about the simulation result for
    `AssetPricingMultFiniteMarkov`.

    Attributes
    ----------
    S : ndarray(float, ndim=1 or 2)
        Array containing the sample path(s) of the discount factor
        process.

    S_tilde : ndarray(float, ndim=1 or 2)
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
