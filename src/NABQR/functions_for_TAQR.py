"""Time-Adaptive Quantile Regression (TAQR) Implementation

This module provides the core implementation of the TAQR algorithm based on
Møller's thesis "Modeling of Uncertainty in Wind Energy Forecast" (2006).
It implements an adaptive simplex algorithm for quantile regression problems.
"""

import numpy as np
import scipy.linalg
import time
# --- Improvement 5: Replace print/naked except with Logging ---
import logging

logger = logging.getLogger(__name__)
# --------------------------------------------------------------
from scipy import linalg as la
import statsmodels.api as sm
from .helper_functions import *
import pandas as pd


def opdatering_final(
    X, Xny, IX, Iy, Iex, Ih, Ihc, beta, Rny, K, n, xB, P, tau, i, bins, n_in_bin
):
    """Updates the design matrix and related parameters for the adaptive quantile regression.

    Parameters
    ----------
    X : numpy.ndarray
        Full data matrix
    Xny : numpy.ndarray
        Current design matrix
    IX : numpy.ndarray
        Index set for design matrix columns
    Iy : int
        Index for response variable
    Iex : int
        Index for grouping variable
    Ih : numpy.ndarray
        Index set for basic variables
    Ihc : numpy.ndarray
        Index set for non-basic variables
    beta : numpy.ndarray
        Current coefficient estimates
    Rny : numpy.ndarray
        Residuals array
    K : int
        Number of explanatory variables
    n : int
        Current number of observations
    xB : numpy.ndarray
        Basic solution
    P : numpy.ndarray
        Sign vector
    tau : float
        Quantile level
    i : int
        Current iteration
    bins : numpy.ndarray
        Bin boundaries
    n_in_bin : int
        Maximum number of observations per bin

    Returns
    -------
    tuple
        Updated parameters (Ih, Ihc, xB, Xny, Rny, P, n, i)
    """

    if np.any(np.isinf(Xny)):
        Xny = np.nan_to_num(Xny, nan=0.0, posinf=1e12, neginf=-1e12)


    Xny = np.vstack((Xny, X[n + i - 2, :]))
    Index = np.arange(Xny[:-1, Iy].shape[0])
    j = 1

    while Xny[-1, Iex] > bins[j]:
        j += 1
    j -= 1

    In = Index[(Xny[:-1, Iex] > bins[j]) & (Xny[:-1, Iex] <= bins[j + 1])]

    if In.size > 0:
        Leav = np.min(In)
    else:
        Leav = 1

    minIL, Inmin = np.min(np.abs(Ih - Leav)), np.argmin(np.abs(Ih - Leav))

    if minIL == 0 and len(In) == n_in_bin:
        cB = (P < 0) + P * tau
        invXh = np.linalg.inv(Xny[Ih, IX])
        g = -(P * (Xny[Ihc, IX] @ invXh[:, Inmin])).T @ cB
        h = np.vstack(
            (invXh[:, Inmin], -P * (Xny[Ihc, IX] @ invXh[:, Inmin]))
        ) @ np.sign(g)
        sigma = np.zeros(n - K)
        hm = h[K:]
        xBm = xB[K:]
        xBm[xBm < 0] = 0
        tolerance = 1e-10
        sigma[hm > tolerance] = xBm[hm > tolerance] / hm[hm > tolerance]
        sigma[hm <= tolerance] = np.inf
        alpha, q = np.min(sigma), np.argmin(sigma)

        z = xB - alpha * h
        Ihm = Ih[Inmin]
        Ih[Inmin] = Ihc[q]
        Ihc[q] = Ihm

        xB = z
        xB[q + K] = alpha

        P[q] = np.sign(g) + (g == 0)
        Ih = np.sort(Ih)
        Ihc, IndexIhc = np.sort(Ihc), np.argsort(Ihc)
        P = P[IndexIhc]
        xBm = xB[K:]
        xBm = xBm[IndexIhc]
        xB[K:] = xBm
        beta = xB[:K]

    rny = X[n + i - 2, Iy] - X[n + i - 2, IX] @ beta
    Rny = np.append(Rny, rny)

    if rny < 0:
        P = np.append(P, -1)
        xB = np.append(xB, -rny)
    else:
        P = np.append(P, 1)
        xB = np.append(xB, rny)
    Ihc = np.append(Ihc, n)

    if len(In) == n_in_bin:
        i += 1
        Stay = np.ones(len(Ihc), dtype=bool)
        Stay[Ihc == Leav] = False
        P = P[Stay]
        Ihc = Ihc[Stay]
        Xny = Xny[np.sort(np.hstack((Ih, Ihc))), :]
        xBm = xB[K:]
        xBm = xBm[Stay]
        xB = xB[:-1]
        xB[K:] = xBm
        Ihc[Ihc > Leav] -= 1
        Ih[Ih > Leav] -= 1
    else:
        n += 1

    return Ih, Ihc, xB, Xny, Rny, P, n, i


def rq_initialiser_final(X, r, beta, n):
    """Initialize parameters for the simplex algorithm based on initial solution.

    If the number of zero elements in r is equal to rank(X), then the work is essentially done.
    Otherwise, the if statement will find the index set Ih s.t. X(Ih)*beta=y(Ih) and X(Ih)^(-1)*y(Ih)=beta,
    the important note being that X(Ih) has an inverse.
    This is done by using the LU transform of a non-quadratic matrix.

    Parameters
    ----------
    X : numpy.ndarray
        Design matrix
    r : numpy.ndarray
        Initial residuals
    beta : numpy.ndarray
        Initial coefficients
    n : int
        Number of observations

    Returns
    -------
    tuple
        (xB, Ih, Ihc, P) Basic solution, basic indices, non-basic indices, and sign vector
    """
    Index = np.arange(n)
    if np.sum(r == 0) > len(beta):
        Lr0 = np.sum(r == 0)
        rs, Irs = np.sort(np.abs(r)), np.argsort(np.abs(r))
        Irs = Irs[:Lr0]
        Xh = X[Irs, :]
        P, L, U = scipy.linalg.lu(Xh, permute_l=False)
        In = np.arange(Lr0)
        rI = np.zeros(Lr0 - len(beta), dtype=int)
        for i in range(len(beta), Lr0):
            rI[i - len(beta)] = In[P[:, i] == 1]
        r[Irs[rI]] = 1e-15

    index_to_index = r.flatten() == 0
    if len(index_to_index) < len(Index):
        index_to_index = np.append(
            index_to_index, np.zeros(len(Index) - len(index_to_index), dtype=bool)
        )

    Ih = Index[index_to_index]
    Ihc = np.setdiff1d(Index[: len(r)], Ih)
    P = np.sign(r[Ihc])
    r[np.abs(r) < 1e-15] = 0
    xB = np.hstack((beta, np.abs(r[Ihc])))

    return xB, Ih, Ihc, P


def rq_simplex_alg_final(Ih, Ihc, n, K, xB, Xny, IH, P, tau):
    """Perform one step of the simplex algorithm for quantile regression.

    Parameters
    ----------
    Ih : numpy.ndarray
        Index set for basic variables
    Ihc : numpy.ndarray
        Index set for non-basic variables
    n : int
        Number of observations
    K : int
        Number of explanatory variables
    xB : numpy.ndarray
        Basic solution
    Xny : numpy.ndarray
        Design matrix
    IH : numpy.ndarray
        History of index sets
    P : numpy.ndarray
        Sign vector
    tau : float
        Quantile level

    Returns
    -------
    tuple
        Algorithm parameters for the next iteration
    """
    # --- Improvement 2: Numerical Stability ---
    # Numerical stability: use pseudo-inverse with a fixed tolerance or 
    # add a small regularization term if the matrix is ill-conditioned.
    Xh = Xny[Ih, :]
    try:
        # Check condition number
        cond = np.linalg.cond(Xh)
        if cond > 1e12:
            # If ill-conditioned, use pseudo-inverse
            invXh = np.linalg.pinv(Xh, rcond=1e-15)
        else:
            invXh = la.inv(Xh)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse for singular matrices
        invXh = np.linalg.pinv(Xh, rcond=1e-15)
    except Exception as e:
        raise ValueError(
            f"Matrix inversion failed even with pseudo-inverse. Matrix shape: {Xh.shape}"
        ) from e
    # ------------------------------------------

    cB = (P < 0) + P * tau
    cC = np.vstack((np.ones(K) * tau, np.ones(K) * (1 - tau))).reshape(-1, 1)
    IB2 = -np.dot(P.reshape(-1, 1) * (np.ones((1, K)) * Xny[Ihc, :]), invXh)
    g = IB2.T @ cB
    d = cC - np.vstack((g, -g)).reshape(-1, 1)
    d[np.abs(d) < 1e-15] = 0
    d = d.flatten()

    md, s = np.sort(d), np.argsort(d)
    s = s[md < 0]
    md = md[md < 0]
    c = np.ones(len(s))
    c[s > (K - 1)] = -1
    C = np.diag(c)
    s[s > (K - 1)] -= K
    h = np.vstack([invXh[:, s], IB2[:, s]]) @ C
    alpha = np.zeros(len(s))
    q = np.zeros(len(s), dtype=int)
    xm = xB[K:]
    xm[xm < 0] = 0
    hm = h[K:, :]
    cq = np.zeros(len(s))
    tol1 = 1e-12

    for k in range(len(s)):
        sigma = xm.copy()
        sigma[hm[:, k] > tol1] = xm[hm[:, k] > tol1] / hm[hm[:, k] > tol1, k]
        sigma[hm[:, k] <= tol1] = np.inf
        alpha[k], q[k] = np.min(sigma), np.argmin(sigma)
        cq[k] = c[k]

    gain = md * alpha
    Mgain, IMgain = np.sort(gain), np.argsort(gain)
    CON = np.inf
    j = 0

    if len(gain) == 0:
        gain = 1
    else:
        while CON > 1e6 and j < len(s):
            j += 1
            IhMid = Ih.copy()
            shifter = 0
            IhMid[s[IMgain[j - 1 + shifter]]] = Ihc[q[IMgain[j - 1 + shifter]]]
            IhMid = np.sort(IhMid)

            if IH.shape[1] <= 1:
                IH = IH.T.reshape(-1, 1)
            IhMid = IhMid.reshape(1, -1)

            if (
                np.min(np.sum(np.abs(IH - IhMid.T * np.ones((1, IH.shape[1]))), axis=0))
                == 0
            ):
                CON = np.inf
            else:
                CON = np.linalg.cond(Xny[(IhMid.flatten()), :])

        s = s[IMgain[j - 1 + shifter]]
        q = q[IMgain[j - 1 + shifter]]
        cq = cq[IMgain[j - 1 + shifter]]
        alpha = alpha[IMgain[j - 1 + shifter]]
        IH = np.hstack((IH, IhMid.T))
        h = h[:, IMgain[j - 1 + shifter]]
        gain = gain[IMgain[j - 1 + shifter]]
        md = md[IMgain[j - 1 + shifter]]

    return CON, s, q, gain, md, alpha, h, IH, cq


def rq_purify_final(xB, Ih, Ihc, P, K, Xny, yny):
    """Handle infeasible points in a simplex formulation of a quantile regression problem.

    The underlying assumption is that there are no restrictions in the problem.
    The updating can therefore be done by recalculating all residuals and coefficients.
    The assumption is further that we are in a position s.t.
    Xny*Xny(Ih)^(-1)*yny(Ih)=yny+residuals
    K=rank(Xny)

    Parameters
    ----------
    xB : numpy.ndarray
        Basic solution
    Ih : numpy.ndarray
        Index set for basic variables
    Ihc : numpy.ndarray
        Index set for non-basic variables
    P : numpy.ndarray
        Sign vector
    K : int
        Number of explanatory variables
    Xny : numpy.ndarray
        Design matrix
    yny : numpy.ndarray
        Response vector

    Returns
    -------
    tuple
        (xB, P) Updated basic solution and sign vector
    """
    invXh = np.linalg.inv(Xny[Ih, :])
    xB = np.hstack((invXh @ yny[Ih], yny[Ihc] - Xny[Ihc, :] @ invXh @ yny[Ih]))
    P = np.sign(xB[K:])
    P[P == 0] = 1
    xB[K:] = np.abs(xB[K:])
    return xB, P


def rq_simplex_final(X, IX, Iy, Iex, r, beta, n, tau, bins, n_in_bin):
    """Calculate solution to an adaptive simplex algorithm for quantile regression.

    The function uses knowledge of the solution at time t to calculate the solution at
    time t+1. The basic idea is that the solution to the quantile regression
    problem can be written as:
    y(t) = X(t)'*beta + r(t)

    where beta = X(h)^(-1)*y(h) for some index set h. Simplex algorithm
    is used to calculate the optimal h at time t+1 based on the solution
    at time t.

    Parameters
    ----------
    X : numpy.ndarray
        Design matrix for the linear quantile regression problem
    IX : numpy.ndarray
        Index set referring to columns of X which is the design matrix
    Iy : int
        Index referring to response column in X
    Iex : int
        Index referring to grouping variable column in X
    r : numpy.ndarray
        Residuals from initial solution
    beta : numpy.ndarray
        Initial solution coefficients
    n : int
        Number of elements in r
    tau : float
        Required probability
    bins : numpy.ndarray
        Vector defining partition intervals
    n_in_bin : int
        Number of elements per bin

    Returns
    -------
    tuple
        (N, BETA, GAIN, Ld, Rny, Mx, Re, CON1, T)
        - N: Number of simplex steps
        - BETA: Solution matrix
        - GAIN: Loss function gain
        - Ld: Number of descent directions
        - Rny: One-step-ahead prediction residuals
        - Mx: Minimum constraint solution
        - Re: Training set reliability
        - CON1: Condition numbers
        - T: Computation times

    References
    ----------
    .. [1] J. K. Møller (2006), "Modeling of Uncertainty in Wind Energy
           Forecast". Master Thesis, Technical University of Denmark.
    .. [2] H. B. Nielsen (1999), "Algorithms for Linear Optimization, an
           Introduction". DTU Course Notes.
    """
    GAIN = np.array([0])
    Rny = np.array([0])
    Ld = np.array([0])
    mx = 0
    T = np.zeros(len(X[:, Iy]))
    Mx = np.zeros(len(X[:, Iy]))
    N_num_of_simplex_steps = []
    N_size_of_the_design_matrix_at_time_k = []
    CON1 = np.array([0])
    BETA = beta.reshape(1, -1)

    K = len(beta)
    Xny = X[:(n), :]
    H = np.zeros((1, K))
    tolmx = 1e-15
    j = 0
    LX = len(X[:, Iy])
    i = 2
    k = 0

    xB, Ih, Ihc, P = rq_initialiser_final(Xny[:, IX], r, beta, n)
    Re = np.zeros(LX - n)

    while i + n < LX:
        k += 1
        t = time.time()

        Re[k] = np.sum(P < 0) / n
        mx = np.min(xB[K:])

        if j > 0 and mx < -tolmx:
            xB, P = rq_purify_final(xB, Ih, Ihc, P, K, Xny[:, IX], Xny[:, Iy])
            mx = np.min(xB[K:])

        Mx[k] = mx
        j = 0
        beta = xB[:K]
        BETA = np.vstack((BETA, beta))

        Ih, Ihc, xB, Xny, Rny, P, n, i = opdatering_final(
            X, Xny, IX, Iy, Iex, Ih, Ihc, beta, Rny, K, n, xB, P, tau, i, bins, n_in_bin
        )

        IH = Ih.reshape(-1, 1)
        CON, s, q, gain, md, alpha, h, IH, cq = rq_simplex_alg_final(
            Ih, Ihc, n, K, xB, Xny[:, IX], IH, P, tau
        )
        CON1 = np.append(CON1, CON)

        while gain <= 0 and md < 0 and j < 24 and CON < 1e6:
            GAIN = np.append(GAIN, gain)
            j += 1

            z = xB - alpha * h

            IhM = Ih[s]
            IhcM = Ihc[q]
            Ih[s] = IhcM
            Ihc[q] = IhM
            P[q] = cq

            xB = z
            xB[q + K] = alpha

            Ih = np.sort(Ih)
            Ihc, IndexIhc = np.sort(Ihc), np.argsort(Ihc)
            P = P[IndexIhc]
            xBm = xB[K:]
            xBm = xBm[IndexIhc]
            xB[K:] = xBm

            CON, s, q, gain, md, alpha, h, IH, cq = rq_simplex_alg_final(
                Ih, Ihc, n, K, xB, Xny[:, IX], IH, P, tau
            )
            CON1 = np.append(CON1, CON)

        N_num_of_simplex_steps.append(j)
        N_size_of_the_design_matrix_at_time_k.append(n)
        T[k] = time.time() - t

    N = np.hstack([N_num_of_simplex_steps, N_size_of_the_design_matrix_at_time_k])
    return N, BETA, GAIN[1:], Ld, Rny, Mx, Re, CON1[1:], T


def one_step_quantile_prediction(
    X_input,
    Y_input,
    n_init,
    n_full,
    quantile=0.5,
    already_correct_size=False,
    n_in_X=5000,
    print_output=True,
):
    """Perform one-step quantile prediction using TAQR.

    Takes the entire training set and, based on the last n_init observations,
    calculates residuals and coefficients for the quantile regression.

    Parameters
    ----------
    X_input : numpy.ndarray
        Input features matrix
    Y_input : numpy.ndarray
        Target values array
    n_init : int
        Number of initial observations for training
    n_full : int
        Total number of observations to use
    quantile : float, optional
        Quantile level to predict, by default 0.5
    already_correct_size : bool, optional
        Whether inputs are already correctly sized, by default False
    n_in_X : int, optional
        Number of observations to use in X, by default 5000
    print_output : bool, optional
        Whether to print diagnostic output, by default True

    Returns
    -------
    tuple
        (y_pred, y_actual, BETA) Predictions, actual values, and coefficients
    """
    assert n_init <= n_full - 2, "n_init must be less than or equal to n_full - 2"

    if type(X_input) == pd.DataFrame:
        X_input = X_input.to_numpy()

    if type(Y_input) == pd.Series or type(Y_input) == pd.DataFrame:
        Y_input = Y_input.to_numpy()

    n, m = X_input.shape
    if print_output:
        # --- Improvement 5: Replace print/naked except with Logging ---
        print(f"X_input shape: {X_input.shape}")
        logger.debug(f"X_input shape: {X_input.shape}")
        # --------------------------------------------------------------

    full_length, p = X_input.shape

    X = X_input[:n_full, :].copy()
    Y = Y_input[:n_full]

    X_for_residuals = X[:n_init, :]
    Y_for_residuals = Y[:n_init]

    # --- Improvement 1: Remove R dependency ---
    beta_init, residuals = run_quantile_regression_python(X_for_residuals, Y_for_residuals, tau=quantile)
    # ------------------------------------------

    if print_output:
        # --- Improvement 5: Replace print/naked except with Logging ---
        print(f"len of beta_init: {len(beta_init)}")
        print(
            f"There is: {sum(residuals == 0)} zeros in residuals and {sum(abs(residuals) < 1e-8)} close to zeroes"
        )
        print(f"p: {p}")
        logger.debug(f"len of beta_init: {len(beta_init)}")
        logger.debug(
            f"There is: {sum(residuals == 0)} zeros in residuals and {sum(abs(residuals) < 1e-8)} close to zeroes"
        )
        logger.debug(f"p: {p}")
        # --------------------------------------------------------------

    if len(beta_init) < p:
        beta_init = np.append(beta_init, np.ones(p - len(beta_init)))
    else:
        beta_init = beta_init[:p]

    r_init = set_n_closest_to_zero(arr=residuals, n=len(beta_init))

    if print_output:
        # --- Improvement 5: Replace print/naked except with Logging ---
        print(f"{sum(r_init == 0)} r_init zeros")
        logger.debug(f"{sum(r_init == 0)} r_init zeros")
        # --------------------------------------------------------------

    X_full = np.column_stack((X, Y, np.random.choice([1, 1], size=n_full)))
    IX = np.arange(p)
    Iy = p
    Iex = p + 1
    bins = np.array([-np.inf, np.inf])
    tau = quantile
    n_in_bin = int(1.0 * full_length)
    if print_output:
        # --- Improvement 5: Replace print/naked except with Logging ---
        print(f"n_in_bin: {n_in_bin}")
        logger.debug(f"n_in_bin: {n_in_bin}")
        # --------------------------------------------------------------

    n_input = n_in_X
    N, BETA, GAIN, Ld, Rny, Mx, Re, CON1, T = rq_simplex_final(
        X_full, IX, Iy, Iex, r_init, beta_init, n_input, tau, bins, n_in_bin
    )

    y_pred = np.sum((X_input[(n_input + 2) : (n_full), :] * BETA[1:, :]), axis=1)
    y_actual = Y_input[(n_input) : (n_full - 2)]
    
    if print_output:
        # --- Improvement 5: Replace print/naked except with Logging ---
        print(f"y_pred shape: {y_pred.shape}")
        print(f"y_actual shape: {y_actual.shape}")
        logger.debug(f"y_pred shape: {y_pred.shape}")
        logger.debug(f"y_actual shape: {y_actual.shape}")
        # --------------------------------------------------------------

    return y_pred, y_actual, BETA


# --- Improvement 1: Remove R dependency ---
def run_quantile_regression_python(X, y, tau):
    """Run quantile regression using statsmodels.
# ------------------------------------------

    Parameters
    ----------
    X : numpy.ndarray
        Features matrix
    y : numpy.ndarray
        Target values
    tau : float
        Quantile level

    Returns
    -------
    tuple
        (coefficients, residuals)
    """
    model = sm.QuantReg(y, X)
    res = model.fit(q=tau)
    return res.params, res.resid
