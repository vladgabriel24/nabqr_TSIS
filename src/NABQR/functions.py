"""Neural Adaptive Basis Quantile Regression (NABQR) Core Functions

This module provides the core functionality for NABQR.

This module includes:
- Scoring metrics (Variogram, CRPS, QSS)
- Dataset creation and preprocessing
- Model definitions and training
- TAQR (Time-Adaptive Quantile Regression) implementation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import properscoring as ps
import tensorflow as tf
import tensorflow_probability as tfp
import datetime as dt
# --- Improvement 5: Replace print/naked except with Logging ---
import logging

# Configure logger
logger = logging.getLogger(__name__)
# --------------------------------------------------------------
# --- Improvement 1: Remove R dependency ---
import statsmodels.api as sm
# ------------------------------------------
from .helper_functions import set_n_closest_to_zero
from .functions_for_TAQR import *


def variogram_score_single_observation(x, y, p=0.5):
    """Calculate the Variogram score for a given observation.

    Translated from the R code in Energy and AI paper: 
    "An introduction to multivariate probabilistic forecast evaluation" by Mathias B.B. et al.

    Parameters
    ----------
    x : numpy.ndarray
        Ensemble forecast (m x k), where m is ensemble size, k is forecast horizon
    y : numpy.ndarray
        Actual observations (k,)
    p : float, optional
        Power parameter for the variogram score, by default 0.5

    Returns
    -------
    float
        Variogram score for the observation
    """
    m, k = x.shape
    score = 0

    for i in range(k - 1):
        for j in range(i + 1, k):
            Ediff = (1 / m) * np.sum(np.abs(x[:, i] - x[:, j]) ** p)
            score += (1 / np.abs(i - j)) * (np.abs(y[i] - y[j]) ** p - Ediff) ** 2

    return score / k


def variogram_score_R_multivariate(x, y, p=0.5, t1=12, t2=36):
    """Calculate the Variogram score for all observations for the time horizon t1 to t2.
    Modified from the R code in Energy and AI paper: 
    "An introduction to multivariate probabilistic forecast evaluation" by Mathias B.B. et al.
    Here we use t1 -> t2 as our forecast horizon.
    
    Parameters
    ----------
    x : numpy.ndarray
        Ensemble forecast (m x k)
    y : numpy.ndarray
        Actual observations (k,)
    p : float, optional
        Power parameter, by default 0.5
    t1 : int, optional
        Start hour (inclusive), by default 12
    t2 : int, optional
        End hour (exclusive), by default 36

    Returns
    -------
    tuple
        (score, score_list) Overall score and list of individual scores
    """
    m, k = x.shape
    score = 0
    if m > k:
        x = x.T
        m, k = k, m

    score_list = []
    for start in range(0, k, 24):
        if start + t2 <= k:
            for i in range(start + t1, start + t2 - 1):
                for j in range(i + 1, start + t2):
                    Ediff = (1 / m) * np.sum(np.abs(x[:, i] - x[:, j]) ** p)
                    score += (1 / np.abs(i - j)) * (
                        np.abs(y[i] - y[j]) ** p - Ediff
                    ) ** 2
                score_list.append(score)

    return score / (100_000), score_list


def variogram_score_R_v2(x, y, p=0.5, t1=12, t2=36):
    """
    Calculate the Variogram score for all observations for the time horizon t1 to t2.
    Modified from the paper in Energy and AI, >> An introduction to multivariate probabilistic forecast evaluation <<.
    Assumes that x and y starts from day 0, 00:00.
    

    Parameters:
    x : array
        Ensemble forecast (m x k), where m is the size of the ensemble, and k is the maximal forecast horizon.
    y : array
        Actual observations (k,)
    p : float
        Power parameter for the variogram score.
    t1 : int
        Start of the hour range for comparison (inclusive).
    t2 : int
        End of the hour range for comparison (exclusive).

    Returns:
    --------
    tuple
        (score, score_list) Overall score/100_000 and list of individual VarS contributions
    """

    m, k = x.shape  # Size of ensemble, Maximal forecast horizon
    score = 0
    if m > k:
        x = x.T
        m, k = k, m
    else:
        print("m,k: ", m, k)

    score_list = []
    # Iterate through every 24-hour block
    for start in range(0, k, 24):
        # Ensure we don't exceed the forecast horizon
        if start + t2 <= k:
            for i in range(start + t1, start + t2 - 1):
                for j in range(i + 1, start + t2):
                    Ediff = (1 / m) * np.sum(np.abs(x[:, i] - x[:, j]) ** p)
                    score += (1 / np.abs(i - j)) * (
                        np.abs(y[i] - y[j]) ** p - Ediff
                    ) ** 2
                score_list.append(score)

    # Variogram score
    return score / (100_000), score_list


def calculate_crps(actuals, corrected_ensembles):
    """Calculate the Continuous Ranked Probability Score (CRPS) using the properscoring package.
    If the ensembles do not have the correct dimensions, we transpose them.

    Parameters
    ----------
    actuals : numpy.ndarray
        Actual observations
    corrected_ensembles : numpy.ndarray
        Ensemble forecasts

    Returns
    -------
    float
        Mean CRPS score
    """
    # --- Improvement 5: Replace print/naked except with Logging ---
    try:
        crps = ps.crps_ensemble(actuals, corrected_ensembles)
        return np.mean(crps)
    except Exception as e:
        logger.debug(f"Failed to calculate CRPS with original orientation: {e}. Trying transpose.")
        crps = np.mean(ps.crps_ensemble(actuals, corrected_ensembles.T))
        return crps
    # --------------------------------------------------------------


def calculate_qss(actuals, taqr_results, quantiles):
    """Calculate the Quantile Skill Score (QSS).

    Parameters
    ----------
    actuals : numpy.ndarray
        Actual observations
    taqr_results : numpy.ndarray
        TAQR ensemble forecasts
    quantiles : array-like
        Quantile levels to evaluate

    Returns
    -------
    float
        Quantile Skill Score
    """
    qss_scores = multi_quantile_skill_score(actuals, taqr_results, quantiles)
    # --- Improvement 5: Replace print/naked except with Logging ---
    table = pd.DataFrame({
        "Quantiles": quantiles,
        "QSS NABQR": qss_scores
    })
    print(table)
    logger.info(f"QSS Results:\n{table}")
    return np.mean(qss_scores)
    # --------------------------------------------------------------


def multi_quantile_skill_score(y_true, y_pred, quantiles):
    """Calculate the Quantile Skill Score (QSS) for multiple quantile forecasts.

    Parameters
    ----------
    y_true : numpy.ndarray
        True observed values
    y_pred : numpy.ndarray
        Predicted quantile values
    quantiles : list
        Quantile levels between 0 and 1

    Returns
    -------
    numpy.ndarray
        QSS for each quantile forecast
    """
    y_pred = np.array(y_pred)

    if y_pred.shape[0] > y_pred.shape[1]:
        y_pred = y_pred.T

    assert all(0 <= q <= 1 for q in quantiles), "All quantiles must be between 0 and 1"
    assert len(quantiles) == len(
        y_pred
    ), "Number of quantiles must match inner dimension of y_pred"

    N = len(y_true)
    scores = np.zeros(len(quantiles))

    for i, q in enumerate(quantiles):
        E = y_true - y_pred[i]
        scores[i] = np.sum(np.where(E > 0, q * E, (1 - q) * -E))

    return scores / N

def calculate_scores(
    actuals,
    taqr_results,
    raw_ensembles,
    corrected_ensembles,
    quantiles_taqr,
    data_source,
    plot_reliability=True,
    visualize = True
):
    """Calculate Variogram, CRPS, QSS and MAE for the predictions and corrected ensembles.

    Parameters
    ----------
    actuals : numpy.ndarray
        The actual values
    predictions : numpy.ndarray
        The predicted values
    raw_ensembles : numpy.ndarray
        The raw ensembles
    corrected_ensembles : numpy.ndarray
        The corrected ensembles
    quantiles : list
        The quantiles to calculate the scores for
    data_source : str
        The data source
    """

    # Find common index
    common_index = corrected_ensembles.index.intersection(actuals.index)

    ensembles_CE_index = raw_ensembles.loc[common_index]
    actuals_comp = actuals.loc[common_index]

    variogram_score_raw_v2, _ = variogram_score_R_v2(
        ensembles_CE_index.loc[actuals_comp.index].values, actuals_comp.values
    )
    variogram_score_raw_corrected_v2, _ = variogram_score_R_v2(
        corrected_ensembles.loc[actuals_comp.index].values, actuals_comp.values
    )
    variogram_score_corrected_taqr_v2, _ = variogram_score_R_v2(
        taqr_results.values, actuals_comp.values
    )

    qs_raw = calculate_qss(
        actuals_comp.values,
        ensembles_CE_index.loc[actuals_comp.index].T,
        np.linspace(0.05, 0.95, ensembles_CE_index.shape[1]),
    )
    qs_corr = calculate_qss(
        actuals_comp.values,
        corrected_ensembles.loc[actuals_comp.index].T,
        np.linspace(0.05, 0.95, corrected_ensembles.shape[1]),
    )

    # TODO: Should be done with max and min from the training set. 
    taqr_values_clipped = np.clip(taqr_results, 0, max(actuals_comp.values))
    qs_corrected_taqr = calculate_qss(
        actuals_comp.values, taqr_values_clipped, quantiles_taqr
    )

    

    crps_orig_ensembles = calculate_crps(
        actuals_comp.values.flatten(), ensembles_CE_index.loc[actuals_comp.index].T
    )
    crps_corr_ensembles = calculate_crps(
        actuals_comp.values.flatten(), corrected_ensembles.loc[actuals_comp.index].T
    )
    crps_corrected_taqr = calculate_crps(
        actuals_comp.values.flatten(), np.array(taqr_results)
    )

    # Instead of calculating mean value of ensembles, we just use the median
    MAE_raw_ensembles = np.abs(
        np.median(ensembles_CE_index.loc[actuals_comp.index].values, axis=1)
        - actuals_comp.values
    )
    MAE_corr_ensembles = np.abs(
        np.median(corrected_ensembles.loc[actuals_comp.index].values, axis=1)
        - actuals_comp.values
    )
    MAE_corrected_taqr = np.abs(
        (np.median(np.array(taqr_results), axis=1) - actuals_comp.values)
    )

    scores_data = {
        "Metric": ["MAE", "CRPS", "Variogram", "QS"],
        "Original Ensembles": [
            np.mean(MAE_raw_ensembles),
            crps_orig_ensembles,
            variogram_score_raw_v2,
            np.mean(qs_raw),
        ],
        "Corrected Ensembles": [
            np.mean(MAE_corr_ensembles),
            crps_corr_ensembles,
            variogram_score_raw_corrected_v2,
            np.mean(qs_corr),
        ],
        "NABQR": [
            np.mean(MAE_corrected_taqr),
            crps_corrected_taqr,
            variogram_score_corrected_taqr_v2,
            np.mean(qs_corrected_taqr),
        ],
    }

    scores_df = pd.DataFrame(scores_data).T

    # Calculate relative scores
    scores_data["Corrected Ensembles"] = [
        1
        + (x - scores_data["Original Ensembles"][i])
        / scores_data["Original Ensembles"][i]
        for i, x in enumerate(scores_data["Corrected Ensembles"])
    ]
    scores_data["NABQR"] = [
        1
        + (x - scores_data["Original Ensembles"][i])
        / scores_data["Original Ensembles"][i]
        for i, x in enumerate(scores_data["NABQR"])
    ]

    # Create DataFrame
    scores_df = pd.DataFrame(scores_data).T

    # --- Improvement 5: Replace print/naked except with Logging ---
    print("Scores: ")
    print(scores_df)
    logger.info(f"Scores for {data_source}:\n{scores_df}")
    # --------------------------------------------------------------

    # Print LaTeX table
    latex_output = scores_df.to_latex(
        column_format="lcccc",
        header=True,
        float_format="%.3f",
        caption=f"Performance Metrics for Different Ensemble Methods on {data_source}",
        label="tab:performance_metrics",
        escape=False,
    ).replace("& 0 & 1 & 2 & 3 \\\\\n\\midrule\n", "")

    with open(f"latex_output_{data_source}_scores.tex", "w") as f:
        f.write(latex_output)

    # Reliability plot 
    reliability_points_taqr, reliability_points_ensembles, reliability_points_corrected_ensembles = reliability_func(
        taqr_results,
        corrected_ensembles,
        raw_ensembles,
        actuals,
        quantiles_taqr,
        data_source,
        plot_reliability = visualize,
    )

    return reliability_points_taqr, reliability_points_ensembles, reliability_points_corrected_ensembles, scores_df

def remove_zero_columns(df):
    """Wrapper function to remove columns that contain only zeros from a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame

    Returns
    -------
    pandas.DataFrame
        DataFrame with zero columns removed
    """
    return df.loc[:, (df != 0).any(axis=0)]


def remove_zero_columns_numpy(arr):
    """Remove columns that contain only zeros or constant values from a numpy array.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array

    Returns
    -------
    numpy.ndarray
        Array with zero/constant columns removed
    """
    return arr[:, (arr != 0).any(axis=0) & (arr != arr[0]).any(axis=0)]


def create_dataset_for_lstm(X, Y, time_steps):
    """Create a dataset suitable for LSTM training with multiple time steps (i.e. lags).

    Parameters
    ----------
    X : numpy.ndarray
        Input features
    Y : numpy.ndarray
        Target values
    time_steps : list
        List of time steps to include

    Returns
    -------
    tuple
        (X_lstm, Y_lstm) LSTM-ready datasets
    """
    X = np.array(X)
    Y = np.array(Y)

    Xs, Ys = [], []
    for i in range(len(X)):
        X_entry = []
        for ts in time_steps:
            if i - ts >= 0:
                X_entry.append(X[i - ts, :])
            else:
                X_entry.append(np.zeros_like(X[0, :]))
        Xs.append(np.array(X_entry))
        Ys.append(Y[i])
    return np.array(Xs), np.array(Ys)


class QuantileRegressionLSTM(tf.keras.Model):
    """LSTM-based model for quantile regression.
    Input: x -> LSTM -> Dense -> Dense -> output

    Parameters
    ----------
    n_quantiles : int
        Number of quantiles to predict
    units : int
        Number of LSTM units
    n_timesteps : int
        Number of time steps in input
    """

    def __init__(self, n_quantiles, units, n_timesteps, **kwargs):
        super().__init__(**kwargs)
        self.lstm = tf.keras.layers.LSTM(
            units, input_shape=(None, n_quantiles, n_timesteps), return_sequences=False
        )
        self.dense = tf.keras.layers.Dense(n_quantiles, activation="sigmoid")
        self.dense2 = tf.keras.layers.Dense(n_quantiles, activation="relu")
        self.n_quantiles = n_quantiles
        self.n_timesteps = n_timesteps

    def call(self, inputs, training=None):
        """Forward pass of the model.

        Parameters
        ----------
        inputs : tensorflow.Tensor
            Input tensor
        training : bool, optional
            Whether in training mode, by default None

        Returns
        -------
        tensorflow.Tensor
            Model output
        """
        x = self.lstm(inputs, training=training)
        x = self.dense(x)
        x = self.dense2(x)
        return x

    def get_config(self):
        """Get model configuration.

        Returns
        -------
        dict
            Model configuration
        """
        config = super(QuantileRegressionLSTM, self).get_config()
        config.update(
            {
                "n_quantiles": self.n_quantiles,
                "units": self.lstm.units,
                "n_timesteps": self.n_timesteps,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Create model from configuration.

        Parameters
        ----------
        config : dict
            Model configuration

        Returns
        -------
        QuantileRegressionLSTM
            Model instance
        """
        return cls(**config)


def quantile_loss_3(q, y_true, y_pred):
    """Calculate quantile loss for a single quantile.

    Parameters
    ----------
    q : float
        Quantile level
    y_true : tensorflow.Tensor
        True values
    y_pred : tensorflow.Tensor
        Predicted values

    Returns
    -------
    tensorflow.Tensor
        Quantile loss value
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tfp.stats.percentile(y_true, 100 * q, axis=1)
    error = y_true - y_pred
    return tf.maximum(q * error, (q - 1) * error)


def quantile_loss_func(quantiles):
    """Create a loss function for multiple quantiles.

    Parameters
    ----------
    quantiles : list
        List of quantile levels

    Returns
    -------
    function
        Loss function for multiple quantiles
    """

    def loss(y_true, y_pred):
        """Calculate the loss for given true and predicted values.

        Parameters
        ----------
        y_true : tensorflow.Tensor
            True values
        y_pred : tensorflow.Tensor
            Predicted values

        Returns
        -------
        tensorflow.Tensor
            Combined loss value for all quantiles
        """
        losses = []
        for i, q in enumerate(quantiles):
            loss = quantile_loss_3(q, y_true, y_pred[:, i])
            losses.append(loss)
        return tf.reduce_mean(tf.stack(losses))

    return loss


def map_range(values, input_start, input_end, output_start, output_end):
    """Map values from one range to another.

    Parameters
    ----------
    values : list
        Values to map
    input_start : float
        Start of input range
    input_end : float
        End of input range
    output_start : float
        Start of output range
    output_end : float
        End of output range

    Returns
    -------
    numpy.ndarray
        Mapped values
    """
    mapped_values = []
    for value in values:
        proportion = (value - input_start) / (input_end - input_start)
        mapped_value = output_start + (proportion * (output_end - output_start))
        mapped_values.append(int(mapped_value))

    return np.array(mapped_values)


def legend_without_duplicate_labels(ax):
    """Create a legend without duplicate labels.
    Primarily used for ensemble plots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to create legend for
    """
    handles, labels = ax.get_legend_handles_labels()
    unique = [
        (h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]
    ]
    ax.legend(*zip(*unique))


def remove_straight_line_outliers(ensembles):
    """Remove ensemble members that are perfectly straight lines (constant slope).
    Explanation: Sometimes the output from the LSTM is a straight line, which is not useful for the ensemble.

    Parameters
    ----------
    ensembles : numpy.ndarray
        2D array where rows are time steps and columns are ensemble members

    Returns
    -------
    numpy.ndarray
        Filtered ensemble data without straight-line outliers
    """
    # Calculate differences along the time axis
    differences = np.diff(ensembles, axis=0)

    # Identify columns where all differences are the same (perfectly straight lines)
    straight_line_mask = np.all(differences == differences[0, :], axis=0)

    # Remove the columns with perfectly straight lines
    return ensembles[:, ~straight_line_mask]


def train_model_lstm(
    quantiles,
    epochs: int,
    lr: float,
    batch_size: int,
    x,
    y,
    x_val,
    y_val,
    n_timesteps,
    data_name,
):
    """Train LSTM model for quantile regression.
    The @tf.function decorator is used to speed up the training process.


    Parameters
    ----------
    quantiles : list
        List of quantile levels to predict
    epochs : int
        Number of training epochs
    lr : float
        Learning rate for optimizer
    batch_size : int
        Batch size for training
    x : tensor
        Training input data
    y : tensor
        Training target data
    x_val : tensor
        Validation input data
    y_val : tensor
        Validation target data
    n_timesteps : int
        Number of time steps in input sequence
    data_name : str
        Name identifier for saving model artifacts

    Returns
    -------
    tf.keras.Model
        Trained LSTM model
    """
    model = QuantileRegressionLSTM(
        n_quantiles=len(quantiles), units=256, n_timesteps=n_timesteps
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            losses = quantile_loss_func(quantiles)(y_batch, y_pred)
            total_loss = tf.reduce_mean(losses)

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss

    @tf.function
    def val_step(x_batch, y_batch):
        y_pred = model(x_batch, training=False)
        losses = quantile_loss_func(quantiles)(y_batch, y_pred)
        total_loss = tf.reduce_mean(losses)
        return total_loss

    train_loss_history = []
    val_loss_history = []
    y_preds = []
    y_true = []

    for epoch in range(epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        num_batches = 0

        # Training loop
        for i in range(0, len(x), batch_size):
            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]

            batch_train_loss = train_step(x_batch, y_batch)
            epoch_train_loss += batch_train_loss
            num_batches += 1

            y_preds.append(model(x_batch, training=False))
            y_true.append(y_batch)

        epoch_train_loss /= num_batches
        train_loss_history.append(epoch_train_loss)

        # Validation loop
        num_val_batches = 0
        for i in range(0, len(x_val), batch_size):
            x_val_batch = x_val[i : i + batch_size]
            y_val_batch = y_val[i : i + batch_size]

            batch_val_loss = val_step(x_val_batch, y_val_batch)
            epoch_val_loss += batch_val_loss
            num_val_batches += 1

        epoch_val_loss /= num_val_batches
        val_loss_history.append(epoch_val_loss)

        # --- Improvement 5: Replace print/naked except with Logging ---
        print(
            f"Epoch {epoch+1} Train Loss: {epoch_train_loss:.4f} Validation Loss: {epoch_val_loss:.4f}"
        )
        logger.info(
            f"Epoch {epoch+1} Train Loss: {epoch_train_loss:.4f} Validation Loss: {epoch_val_loss:.4f}"
        )
        # --------------------------------------------------------------

    y_preds_concat = tf.concat(y_preds, axis=0).numpy()
    y_true_concat = tf.concat(y_true, axis=0).numpy()

    return model



def run_taqr(corrected_ensembles, actuals, quantiles, n_init, n_full, n_in_X):
    """Wrapper function to run TAQR on corrected ensembles.

    Parameters
    ----------
    corrected_ensembles : numpy.ndarray
        Shape (n_timesteps, n_ensembles)
    actuals : numpy.ndarray
        Shape (n_timesteps,)
    quantiles : list
        Quantiles to predict
    n_init : int
        Number of initial timesteps for warm start
    n_full : int
        Total number of timesteps
    n_in_X : int
        Number of timesteps in design matrix

    Returns
    -------
    list
        TAQR results for each quantile
    """
    if type(actuals) == pd.Series or type(actuals) == pd.DataFrame:
        actuals = actuals.to_numpy()
        actuals[np.isnan(actuals)] = 0
    else:
        actuals[np.isnan(actuals)] = 0

    taqr_results = []
    actuals_output = []
    BETA_output = []
    for q in quantiles:
        # --- Improvement 5: Replace print/naked except with Logging ---
        print(f"Running TAQR for quantile: {q}")
        logger.info(f"Running TAQR for quantile: {q}")
        # --------------------------------------------------------------
        y_pred, y_actuals, BETA_q = one_step_quantile_prediction(
            corrected_ensembles,
            actuals,
            n_init=n_init,
            n_full=n_full,
            quantile=q,
            already_correct_size=True,
            n_in_X=n_in_X,
        )
        taqr_results.append(y_pred)
        actuals_output.append(y_actuals)
        BETA_output.append(BETA_q)

    return taqr_results, actuals_output[1], BETA_output


def reliability_func(
    quantile_forecasts,
    corrected_ensembles,
    ensembles,
    actuals,
    corrected_taqr_quantiles,
    data_source,
    plot_reliability=True,
):
    n = len(actuals)

    # Ensuring that we are working with numpy arrays
    quantile_forecasts = (
        np.array(quantile_forecasts)
        if type(quantile_forecasts) != np.ndarray
        else quantile_forecasts
    )
    actuals = np.array(actuals) if type(actuals) != np.ndarray else actuals
    actuals_ensembles = actuals.copy()
    actuals_taqr = actuals.copy()
    corrected_taqr_quantiles = (
        np.array(corrected_taqr_quantiles)
        if type(corrected_taqr_quantiles) != np.ndarray
        else corrected_taqr_quantiles
    )
    corrected_ensembles = (
        np.array(corrected_ensembles)
        if type(corrected_ensembles) != np.ndarray
        else corrected_ensembles
    )
    ensembles = np.array(ensembles) if type(ensembles) != np.ndarray else ensembles

    # Handling hpe (high probability ensemble)
    hpe = ensembles[:, 0]
    hpe_quantile = 0.5
    ensembles = ensembles[:, 1:]

    quantiles_ensembles = np.linspace(0.05, 0.95, ensembles.shape[1]).round(3)
    quantiles_corrected_ensembles = np.linspace(
        0.05, 0.95, corrected_ensembles.shape[1]
    ).round(3)

    m, n1 = quantile_forecasts.shape
    if m != len(actuals):
        quantile_forecasts = quantile_forecasts.T
        m, n1 = quantile_forecasts.shape

    # Ensure that the length match up
    if len(actuals) != len(quantile_forecasts):
        if len(actuals) < len(quantile_forecasts):
            quantile_forecasts = quantile_forecasts[: len(actuals)]
        else:
            actuals_taqr = actuals[-len(quantile_forecasts) :]

    if len(actuals) != len(corrected_ensembles):
        if len(actuals) < len(corrected_ensembles):
            corrected_ensembles = corrected_ensembles[: len(actuals)]
        else:
            actuals_taqr = actuals[-len(corrected_ensembles) :]

    if len(actuals) != len(ensembles):
        if len(actuals) < len(ensembles):
            ensembles = ensembles[: len(actuals)]
            hpe = hpe[: len(actuals)]
        else:
            actuals_taqr = actuals[-len(ensembles) :]
            hpe = hpe[-len(ensembles) :]

    # Reliability: how often actuals are below the given quantiles compared to the quantile levels
    reliability_points_taqr = []
    for i, q in enumerate(corrected_taqr_quantiles):
        forecast = quantile_forecasts[:, i]
        observed_below = np.sum(actuals_taqr <= forecast) / n
        reliability_points_taqr.append(observed_below)

    reliability_points_taqr = np.array(reliability_points_taqr)

    reliability_points_ensembles = []
    n_ensembles = len(actuals_ensembles)
    for i, q in enumerate(quantiles_ensembles):
        forecast = ensembles[:, i]
        observed_below = np.sum(actuals_ensembles <= forecast) / n_ensembles
        reliability_points_ensembles.append(observed_below)

    reliability_points_corrected_ensembles = []
    for i, q in enumerate(quantiles_corrected_ensembles):
        forecast = corrected_ensembles[:, i]
        observed_below = np.sum(actuals_ensembles <= forecast) / n_ensembles
        reliability_points_corrected_ensembles.append(observed_below)

    # Handle hpe separately
    observed_below_hpe = np.sum(actuals_ensembles <= hpe) / n_ensembles

    reliability_points_ensembles = np.array(reliability_points_ensembles)

    # Find the index of the 0.5 quantile
    idx_05 = np.where(corrected_taqr_quantiles == 0.5)[0][0]

    if plot_reliability:
        import scienceplots

        with plt.style.context("no-latex"):
            # Plot reliability: nominal quantiles vs calculated quantiles
            plt.figure(figsize=(6, 6))
            plt.plot(
                [0, 1], [0, 1], "k--", label="Perfect Reliability"
            )  # Diagonal line
            plt.scatter(
                corrected_taqr_quantiles,
                reliability_points_taqr,
                color="blue",
                label="NABQR",
            )
            plt.scatter(
                quantiles_ensembles,
                reliability_points_ensembles,
                color="grey",
                label="Original Ensembles",
                marker="p",
                alpha=0.5,
            )
            plt.scatter(
                quantiles_corrected_ensembles,
                reliability_points_corrected_ensembles,
                color="green",
                label="Corrected Ensembles",
                marker="p",
                alpha=0.5,
            )
            plt.scatter(
                hpe_quantile,
                observed_below_hpe,
                color="grey",
                label="High Prob. Ensemble",
                alpha=0.5,
                marker="D",
                s=25,
            )
            plt.xlabel("Nominal Quantiles")
            plt.ylabel("Observed Frequencies")
            plt.title(
                f'Reliability Plot for {data_source.replace("_", " ").replace("lstm", "")}'
            )
            plt.legend()
            plt.grid(True)
            plt.savefig(f"reliability_plot_{data_source}.pdf")
            plt.show()

    return (
        reliability_points_taqr,
        reliability_points_ensembles,
        reliability_points_corrected_ensembles,
    )


# --- Improvement 3: Refactor 'Pipeline' into Class-based structure ---
class NABQRPipeline:
    """Class-based pipeline for NABQR model training and evaluation.
    
    This refactor improves maintainability and follows the Single Responsibility Principle.
    """
    def __init__(
        self,
        name="TEST",
        training_size=0.8,
        epochs=100,
        timesteps_for_lstm=[0, 1, 2, 6, 12, 24, 48],
        **kwargs
    ):
        self.name = name
        self.training_size = training_size
        self.epochs = epochs
        self.timesteps = timesteps_for_lstm
        self.kwargs = kwargs
        self.model = None
        self.scaler_params = {}
        self.test_idx = None

    def prepare_data(self, X, y):
        """Preprocess data: scaling, lagging, and splitting."""
        actuals = y
        ensembles = X
        
        if isinstance(y, pd.Series):
            idx = y.index
        elif isinstance(X, pd.DataFrame):
            idx = X.index
        else:
            idx = pd.RangeIndex(start=0, stop=len(y), step=1)

        if isinstance(y, np.ndarray):
            X_y = np.concatenate((X, y.reshape(-1, 1)), axis=1)
            y = pd.Series(y, index=idx)
        else:
            X_y = np.concatenate((X, y.values.reshape(-1, 1)), axis=1)
            y = pd.Series(y.values.flatten(), index=idx)

        train_size = int(self.training_size * len(actuals))
        ensembles = pd.DataFrame(ensembles, index=idx)
        actuals = pd.DataFrame(actuals, index=idx)
        common_index = ensembles.index.intersection(actuals.index)
        X_y = pd.DataFrame(X_y, index=idx)
        
        ensembles = ensembles.loc[common_index]
        actuals = actuals.loc[common_index]
        X_y = X_y.loc[common_index]

        Xs, X_Ys = create_dataset_for_lstm(ensembles, X_y, self.timesteps)

        if np.isnan(Xs).any():
            Xs[np.isnan(Xs).any(axis=(1, 2))] = 0
        if np.isnan(X_Ys).any():
            X_Ys[np.isnan(X_Ys).any(axis=1)] = 0

        # Scaling based on training set
        XY_s_max_train = np.max(X_Ys[:train_size])
        XY_s_min_train = np.min(X_Ys[:train_size])
        self.scaler_params = {'max': XY_s_max_train, 'min': XY_s_min_train}

        Xs_scaled = (Xs - XY_s_min_train) / (XY_s_max_train - XY_s_min_train)
        X_Ys_scaled = (X_Ys - XY_s_min_train) / (XY_s_max_train - XY_s_min_train)

        validation_size = 100
        
        data = {
            'x_train': tf.convert_to_tensor(Xs_scaled[:train_size]),
            'y_train': tf.convert_to_tensor(X_Ys_scaled[:train_size]),
            'x_val': tf.convert_to_tensor(Xs_scaled[train_size : (train_size + validation_size)]),
            'y_val': tf.convert_to_tensor(X_Ys_scaled[train_size : (train_size + validation_size)]),
            'x_test': Xs_scaled[train_size:],
            'actuals_full': actuals,
            'train_size': train_size,
            'idx': idx,
            'ensembles_full': ensembles
        }
        return data

    def train(self, data):
        """Train the LSTM model."""
        quantiles_lstm = np.linspace(0.05, 0.95, 20)
        self.model = train_model_lstm(
            quantiles=quantiles_lstm,
            epochs=self.epochs,
            lr=1e-3,
            batch_size=50,
            x=data['x_train'],
            y=data['y_train'],
            x_val=data['x_val'],
            y_val=data['y_val'],
            n_timesteps=self.timesteps,
            data_name=f"{self.name}_LSTM_epochs_{self.epochs}",
        )

        save_files = self.kwargs.get("save_files", True)
        if save_files:
            # --- Improvement 5: Replace print/naked except with Logging ---
            try:
                today = dt.datetime.today().strftime("%Y-%m-%d")
                self.model.save(f"Model_{self.name}_{self.epochs}_{today}.keras")
            except Exception as e:
                logger.warning(f"Failed to save model with date suffix: {e}. Saving with default name.")
                self.model.save(f"Models_{self.name}_{self.epochs}.keras")
            # --------------------------------------------------------------

    def run(self, X, y):
        """Execute the full pipeline."""
        data = self.prepare_data(X, y)
        self.train(data)
        
        # Generate corrected ensembles
        corrected_ensembles = self.model(data['x_test'])
        corrected_ensembles = (
            corrected_ensembles * (self.scaler_params['max'] - self.scaler_params['min']) + self.scaler_params['min']
        )
        
        train_size = data['train_size']
        actuals_out_of_sample = data['actuals_full'][train_size:]
        test_idx = data['idx'][train_size:]

        # Run TAQR
        quantiles_taqr = self.kwargs.get("quantiles_taqr", [0.1, 0.3, 0.5, 0.7, 0.9])
        n_full = len(actuals_out_of_sample)
        n_init = int(0.25 * n_full)
        limit = self.kwargs.get("limit", 5000)
        if n_init < limit:
            n_init = min(int(0.5*n_full), limit)

        corrected_ensembles = corrected_ensembles.numpy()
        corrected_ensembles = remove_zero_columns_numpy(corrected_ensembles)
        corrected_ensembles = remove_straight_line_outliers(corrected_ensembles)
        corrected_ensembles = pd.DataFrame(corrected_ensembles, index=test_idx)
        
        taqr_results, actuals_output, BETA_output = run_taqr(
            corrected_ensembles,
            actuals_out_of_sample,
            quantiles_taqr,
            n_init,
            n_full,
            n_init, # n_in_X
        )
        
        # Slicing results
        actuals_out_of_sample = actuals_out_of_sample[(n_init + 1) : (n_full - 1)]
        actuals_output_series = pd.Series(
            actuals_output.flatten(), index=test_idx[(n_init + 1) : (n_full - 1)]
        )
        corrected_ensembles = corrected_ensembles.iloc[(n_init + 1) : (n_full - 1)]
        idx_to_save = test_idx[(n_init + 1) : (n_full - 1)]

        if self.kwargs.get("save_files", True):
            today = dt.datetime.today().strftime("%Y-%m-%d")
            np.save(f"results_{today}_{self.name}_actuals_out_of_sample.npy", actuals_out_of_sample)
            pd.DataFrame(corrected_ensembles, index=idx_to_save).to_csv(f"results_{today}_{self.name}_corrected_ensembles.csv")
            np.save(f"results_{today}_{self.name}_taqr_results.npy", taqr_results)
            np.save(f"results_{today}_{self.name}_actuals_output.npy", actuals_output_series)
            np.save(f"results_{today}_{self.name}_BETA_output.npy", BETA_output)

        return (
            corrected_ensembles,
            pd.DataFrame(np.array(taqr_results).T, index=idx_to_save),
            actuals_output_series,
            BETA_output,
            data['ensembles_full'],
        )

def pipeline(
    X,
    y,
    name="TEST",
    training_size=0.8,
    epochs=100,
    timesteps_for_lstm=[0, 1, 2, 6, 12, 24, 48],
    **kwargs,
):

    """Main pipeline for NABQR model training and evaluation. (Legacy Wrapper)"""
    nabqr_pipe = NABQRPipeline(name, training_size, epochs, timesteps_for_lstm, **kwargs)
    return nabqr_pipe.run(X, y)

def generateAIReport_Gemini(
        quantiles, 
        taqr_results, 
        actuals_output, 
        reliability_points_taqr, 
        reliability_points_ensembles, 
        reliability_points_corrected_ensembles,
        GoogleAPI_token,
        GoogleModelURL
):

    from requests import post
    from .helper_functions import generate_pdf_report
    
    def query(model_API, payload):

        header = {
            "x-goog-api-key": f"{GoogleAPI_token}",
            "Content-Type": "application/json",
        }

        response = post(model_API, headers=header, json=payload)
        return response.json()

    prompt = f"""
        You are an expert statistician and ML practitioner.

        NABQR quantiles forecasting procedure uses raw ensemble data that is provided to an LSTM that outputs corrected ensemble data. Based on the corrected data,
        the TAQR algorithm is applied and then the quantile levels for the provided quantiles are estimated.

        Interpret the calibration / reliability of the NABQR quantiles forecasting procedure based on the following data provided below:

        Quantiles to be estimated: {quantiles}
        TAQR(Time Adaptive Quantile Regression) estimated quantile levels for the quantiles provided above using LSTM corrected ensemble data: {taqr_results}
        Actual data: {actuals_output}

        Points from the acutal data that are below the estimated quantiles compared to the quantile levels: {reliability_points_taqr}
        Points from uncorrected ensemble train data that are below the estimated quantiles compared to the quantile levels: {reliability_points_ensembles}
        Points from LSTM corrected ensemble train data that are below the estimated quantiles compared to the quantile levels: {reliability_points_corrected_ensembles}

        Tasks: 
        1) Compare the actual data with TAQR estimated results. How accurate they are?
        2) Based on the number of points that are below the predicted quantile level from different data sources, how well did NABQR pipeline improved the quantile regression method?
        3) Format the output as a short and concise research report in UTF-8 format that will be written in a PDF file. Do not use markdown format symbols.
    """

    AI_response = query(GoogleModelURL, {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"{prompt}"
                    }
                ]
            }
        ]
    })
    
    try:
        AI_respone_text = AI_response['candidates'][0]['content']['parts'][0]['text']
        generate_pdf_report(AI_respone_text, filename='nabqr_report.pdf', title='NABQR calibration report') 
    except:
        print("\nAI response failed!\n")
        print(f"\nOutput:{AI_response}\n")
        return AI_response

    return AI_respone_text   

     
