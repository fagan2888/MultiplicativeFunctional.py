import numbers
import numpy as np


class MultiplicativeFunctional(object):
    """
    Class representing a multiplicative functional.

    Parameters
    ----------
    mc : MarkovChain
        MarkovChain instance with n states representing the `X` process.

    G : array_like(float, ndim=2)
        Growth rate matrix. Must be of shape n x n.

    M_inits : array_like(float, ndim=1) or scalar(float),
              optional(default=None)
        Array containing the initial values of the `M` process, one for
        each state. Must be of length n. If not specified, default to
        the vector of all ones. If it is a scalar, then it is converted
        to the constant vector of that scalar.

    Attributes
    ----------
    mc : MarkovChain
        See Parameters.

    G : ndarray(float, ndim=2)
        See Parameters.

    M_inits: ndarray(float, ndim=1)
        See Parameters.

    n : scalar(int)
        Number of the state.

    P : ndarray(float, ndim=2)
        Transition probability matrix of `mc`.

    M_matrix : ndarray(float, ndim=2)
        Generating matrix for the `M` process.

    P_tilde : ndarray(float, ndim=2)
        Matrix representation of the M-operator, given by
        `P * M_matrix`.

    exp_eta : scalar(float)
        Dominant eigenvalue of `P_tilde`.

    eta : scalar(float)
        Log of `exp_eta`.

    e : ndarray(float, ndim=1)
        Dominant eigenvector of `P_tilde`.

    M_tilde_matrix : ndarray(float, ndim=2)
        Generating matrix for the `M_tilde` process.

    """
    def __init__(self, mc, G, M_inits=None):
        self.mc = mc
        self.n = mc.n
        self.P = mc.P

        self.G = np.asarray(G)

        if self.G.ndim != 2 or self.G.shape[0] != self.G.shape[1]:
            raise ValueError('G must be a square matrix')

        if self.G.shape[0] != self.n:
            raise ValueError(
                'order of G must be equal to the number of states'
            )

        if M_inits is None:
            self.M_inits = np.ones(self.n)
        elif isinstance(M_inits, numbers.Real):
            self.M_inits = np.empty(self.n)
            self.M_inits.fill(M_inits)
        else:
            if len(M_inits) != self.n:
                raise ValueError(
                    'length of M_inits must be equal to the number of states'
                )
            self.M_inits = np.asarray(M_inits)

        self.M_matrix = np.exp(self.G)
        self.P_tilde = self.P * self.M_matrix

        self.exp_eta, self.e = _solve_principal_eig(self.P_tilde)
        self.eta = np.log(self.exp_eta)

        self.M_tilde_matrix = \
            self.M_matrix * self.e / self.e.reshape((self.n, 1)) / self.exp_eta

    def simulate(self, ts_length, X_init=None,
                 num_reps=None, random_state=None):
        """
        Simulate the `M` and `M_tilde` processes.

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
        res: MFSimulateResult
            Simulation result represetned as a `MFSimulateResult`. See
            `MFSimulateResult` for details. The array for each attribute
            is of shape `(ts_length,)` if `num_reps=None`, or of shape
            `(num_reps, ts_length)` otherwise.

        """
        X = self.mc.simulate(ts_length, init=X_init,
                             num_reps=num_reps, random_state=random_state)
        return self.generate_paths(X)

    def generate_paths(self, X):
        """
        Given a simulation of the `X` process, generate sample paths of
        the `M` and `M_tilde` processes.

        Parameters
        ----------
        X : array_like(int)
            Array containing the sample path(s) of the `X` process.

        Returns
        -------
        res: MFSimulateResult
            Simulation result represetned as a `MFSimulateResult`. See
            `MFSimulateResult` for details. The array for each attribute
            is of the same shape as `X`.

        """
        X = np.asarray(X)
        M = _generate_mult_process(X, self.M_matrix, self.M_inits)
        M_tilde_inits = np.ones(self.n)
        M_tilde = _generate_mult_process(X, self.M_tilde_matrix, M_tilde_inits)
        res = MFSimulateResult(X=X,
                               M=M,
                               M_tilde=M_tilde)
        return res


def _generate_mult_process(X, mat, inits):
    """
    Return the array `M` given by `M[t+1]/M[t] = mat[X[t], X[t+1]]`
    with `M[0] = inits[X[0]]`.

    """
    M = np.empty_like(X, dtype=float)
    M[..., 0] = inits[X[..., 0]]
    M[..., 1:] = mat[X[..., :-1], X[..., 1:]]
    np.cumprod(M, axis=-1, out=M)
    return M


def _solve_principal_eig(a):
    """
    Solve the principal eigenvalue problem for a non-negative matrix
    `a`.

    """
    w, v = np.linalg.eig(a)
    idx = np.argmax(w)
    eig_val = w[idx]
    eig_vec = v[:, idx]

    # Let eig_vec non-negative
    sign = 0
    i = 0
    while sign == 0 and i < len(eig_vec):
        sign = np.sign(eig_vec[i])
        i += 1
    if sign < 0:
        eig_vec *= -1

    return eig_val, eig_vec


class _Result(dict):
    # This is sourced from sicpy.optimize.OptimizeResult.
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return self.keys()


class MFSimulateResult(_Result):
    """
    Contain the information about the simulation result for
    `MultiplicativeFunctional`.

    Attributes
    ----------
    M : ndarray(float, ndim=1 or 2)
        Array containing the sample path(s) of the `M` process.

    M_tilde : ndarray(float, ndim=1 or 2)
        Array containing the sample path(s) of the `M_tilde` process.

    X : ndarray(int, ndim=1 or 2)
        Array containing the sample path(s) of the `X` process.

    """
    pass
