# Install NABQR (if not already installed in your environment)
# pip install nabqr

# Import the key functions from the NABQR package
from NABQR import simulate_wind_power_sde, run_nabqr_pipeline, generateAIReport_Gemini

import configparser



# 1. Simulate synthetic wind power data:
# We'll use the package's built-in wind power SDE simulator to generate a dataset.
initial_params = {
    'X0': 0.6,           # initial value (normalized power)
    'theta': 0.77,       # long-term mean level (mean reversion target)
    'kappa': 0.12,       # mean reversion speed (lower means slower reversion)
    'sigma_base': 1.05,  # base volatility of the process
    'alpha': 0.57,       # GARCH term (volatility autoregression)
    'beta': 1.2,         # GARCH term (volatility persistence)
    'lambda_jump': 0.045,# intensity of random jumps (Poisson rate)
    'jump_mu': 0.0,      # average jump size (here, 0 means symmetric jumps)
    'jump_sigma': 0.1    # volatility of jump sizes
}
# The above parameters are chosen as in the NABQR paper’s example:contentReference[oaicite:43]{index=43} for realism.

# Generate an ensemble of wind power scenarios and the actual time series.
# T=500 specifies 500 time steps, dt=1.0 for hourly steps.
t, actuals, ensembles = simulate_wind_power_sde(initial_params, T=500, dt=1.0)
print("Ensemble shape:", ensembles.shape, "| Actuals length:", len(actuals))
# 'ensembles' is a NumPy array of shape (T, M), where M is the number of ensemble members (scenarios).
# 'actuals' is a length-T array of the true wind power production.

# 2. Run the NABQR pipeline on the simulated data:
# We provide the ensemble matrix X and actuals y, and specify some pipeline settings.
inference_config = configparser.ConfigParser()
inference_config.read('inference.ini')

google_token = inference_config['GoogleAPIKey']['token']
inference_url = inference_config['GoogleModelURL']['URL']

quantiles = [0.05, 0.5, 0.95]

corrected_ensembles, q_hat, actuals_out, beta_params, scores, reliability_points_taqr, reliability_points_ensembles, reliability_points_corrected_ensembles = run_nabqr_pipeline(
    X = ensembles,
    actuals = actuals,
    training_size = 0.7,            # use 70% of data for LSTM training, 30% for testing
    epochs = 20,                    # train the LSTM for 50 epochs (for demonstration purposes)
    timesteps = [0, 1, 2, 6, 12, 24],  # lag indices for LSTM input (hours back in time)
    quantiles = quantiles,  # target quantiles to predict (5th, 50th, 95th percentiles)
    visualize = True,                # whether to generate plots of the results
    GoogleAPI_token = google_token,
    GoogleModelURL = inference_url,
)

AI_response = generateAIReport_Gemini(quantiles, q_hat, actuals_out, reliability_points_taqr, reliability_points_ensembles, reliability_points_corrected_ensembles, google_token, inference_url)