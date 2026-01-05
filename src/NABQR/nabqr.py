from .functions import *
from .helper_functions import simulate_correlated_ar1_process, set_n_closest_to_zero, generate_pdf_report
import matplotlib.pyplot as plt
import scienceplots

import requests

plt.style.use(["no-latex"])
from .visualization import visualize_results
import datetime as dt

def run_nabqr_pipeline(
    n_samples=2000,
    phi=0.995,
    sigma=8,
    offset_start=10,
    offset_end=500,
    offset_step=15,
    correlation=0.8,
    data_source="NABQR-TEST",
    training_size=0.7,
    epochs=20,
    timesteps=[0, 1, 2, 6, 12, 24],
    quantiles=[0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99],
    X=None,
    actuals=None,
    simulation_type="sde",
    visualize = True,
    taqr_limit=5000,
    save_files = True,
    GoogleAPI_token = None,
    GoogleModelURL = None,
):
    """
    Run the complete NABQR pipeline, which may include data simulation, model training,
    and visualization. The user can either provide pre-computed inputs (X, actuals)
    or opt to simulate data if both are not provided.

    Parameters
    ----------
    n_samples : int, optional
        Number of time steps to simulate if no data provided, by default 5000.
    phi : float, optional
        AR(1) coefficient for simulation, by default 0.995.
    sigma : float, optional
        Standard deviation of noise for simulation, by default 8.
    offset_start : int, optional
        Start value for offset range, by default 10.
    offset_end : int, optional
        End value for offset range, by default 500.
    offset_step : int, optional
        Step size for offset range, by default 15.
    correlation : float, optional
        Base correlation between dimensions, by default 0.8.
    data_source : str, optional
        Identifier for the data source, by default "NABQR-TEST".
    training_size : float, optional
        Proportion of data to use for training, by default 0.7.
    epochs : int, optional
        Number of epochs for model training, by default 100.
    timesteps : list, optional
        List of timesteps to use for LSTM, by default [0, 1, 2, 6, 12, 24].
    quantiles : list, optional
        List of quantiles to predict, by default [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99].
    X : array-like, optional
        Pre-computed input features. If not provided along with `actuals`, the function
        will prompt to simulate data.
    actuals : array-like, optional
        Pre-computed actual target values. If not provided along with `X`, the function
        will prompt to simulate data.
    simulation_type : str, optional
        Type of simulation to use, by default "ar1". "sde" is more advanced and uses a SDE model and realistic.
    visualize : bool, optional
        Determines if any visual elements will be plotted to the screen or saved as figures.
    taqr_limit : int, optional
        The lookback limit for the TAQR model, by default 5000.
    save_files : bool, optional
        Determines if any files will be saved, by default True. Note: the R-file needs to save some .csv files to run properly.
    Returns
    -------
    tuple
        A tuple containing:

        - corrected_ensembles: pd.DataFrame
            The corrected ensemble predictions.
        - taqr_results: list of numpy.ndarray
            The TAQR results.
        - actuals_output: list of numpy.ndarray
            The actual output values.
        - BETA_output: list of numpy.ndarray
            The BETA parameters.
        - scores: pd.DataFrame
            The scores for the predictions and original/corrected ensembles.

    Raises
    ------
    ValueError
        If user opts not to simulate data when both X and actuals are missing.
    """

    # If both X and actuals are not provided, ask user if they want to simulate
    if X is None or actuals is None:
        if X is not None or actuals is not None:
            raise ValueError("Either provide both X and actuals, or none at all.")
        choice = (
            input(
                "X and actuals are not provided. Do you want to simulate data? (y/n): "
            )
            .strip()
            .lower()
        )
        if choice != "y":
            raise ValueError(
                "Data was not provided and simulation not approved. Terminating function."
            )

        # Generate offset and correlation matrix for simulation
        offset = np.arange(offset_start, offset_end, offset_step)
        m = len(offset)
        corr_matrix = correlation * np.ones((m, m)) + (1 - correlation) * np.eye(m)

        # Generate simulated data
        # Check if simulation_type is valid
        if simulation_type not in ["ar1", "sde"]:
            raise ValueError("Invalid simulation type. Please choose 'ar1' or 'sde'.")
        if simulation_type == "ar1":    
            X, actuals = simulate_correlated_ar1_process(
                n_samples, phi, sigma, m, corr_matrix, offset, smooth=5
            )
        elif simulation_type == "sde":
            initial_params = {
                    'X0': 0.6,
                    'theta': 0.77,
                    'kappa': 0.12,        # Slower mean reversion
                    'sigma_base': 1.05,  # Lower base volatility
                    'alpha': 0.57,       # Lower ARCH effect
                    'beta': 1.2,        # High persistence
                    'lambda_jump': 0.045, # Fewer jumps
                    'jump_mu': 0.0,     # Negative jumps
                    'jump_sigma': 0.1    # Moderate jump size variation
                }
            # Check that initial parameters are within bounds
            bounds = get_parameter_bounds()
            for param, value in initial_params.items():
                lower_bound, upper_bound = bounds[param]
                if not (lower_bound <= value <= upper_bound):
                    print(f"Initial parameter {param}={value} is out of bounds ({lower_bound}, {upper_bound})")
                    if value < lower_bound:
                        initial_params[param] = lower_bound
                    else:
                        initial_params[param] = upper_bound
            
            t, actuals, X = simulate_wind_power_sde(
                initial_params, T=n_samples, dt=1.0
            )



        # Plot the simulated data with X in shades of blue and actuals in bold black
        plt.figure(figsize=(10, 6))
        cmap = plt.cm.Blues
        num_series = X.shape[1] if X.ndim > 1 else 1
        colors = [cmap(i) for i in np.linspace(0.3, 1, num_series)]  # Shades of blue
        if num_series > 1:
            for i in range(num_series):
                plt.plot(X[:, i], color=colors[i], alpha=0.7)
        else:
            plt.plot(X, color=colors[0], alpha=0.7)
        plt.plot(actuals, color="black", linewidth=2, label="Actuals")
        plt.title("Simulated Data")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.show()

    # Run the pipeline
    corrected_ensembles, taqr_results, actuals_output, BETA_output, X_ensembles = pipeline(
        X,
        actuals,
        data_source,
        training_size=training_size,
        epochs=epochs,
        timesteps_for_lstm=timesteps,
        quantiles_taqr=quantiles,
        limit=taqr_limit,
        save_files = save_files
    )

    # Get today's date for file naming
    today = dt.datetime.today().strftime("%Y-%m-%d")

    # Visualize results
    if visualize:
        visualize_results(actuals_output, taqr_results, f"{data_source} example")

    # Calculate scores
    reliability_points_taqr, reliability_points_ensembles, reliability_points_corrected_ensembles, scores = calculate_scores(
        actuals_output,
        taqr_results,
        X_ensembles,
        corrected_ensembles,
        quantiles,
        data_source,
        plot_reliability=True,
        visualize = visualize
    )

    return corrected_ensembles, taqr_results, actuals_output, BETA_output, scores, reliability_points_taqr, reliability_points_ensembles, reliability_points_corrected_ensembles


if __name__ == "__main__":
    run_nabqr_pipeline()