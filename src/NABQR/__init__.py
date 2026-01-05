"""NABQR: Neural Adaptive Basis Quantile Regression

A method for sequential error-corrections tailored for wind power forecast in Denmark.
"""

from .nabqr import run_nabqr_pipeline
from .visualization import visualize_results
from .functions import (
    variogram_score_single_observation,
    variogram_score_R_multivariate,
    calculate_crps,
    calculate_qss,
    pipeline,
    generateAIReport_Gemini,
)
from .helper_functions import (
    set_n_smallest_to_zero,
    set_n_closest_to_zero,
    quantile_score,
    simulate_correlated_ar1_process,
    simulate_wind_power_sde,
)
from .functions_for_TAQR import (
    rq_simplex_final,
    one_step_quantile_prediction,
    opdatering_final,
    rq_initialiser_final,
    rq_simplex_alg_final,
    rq_purify_final,
)

__version__ = "0.0.17"

__all__ = [
    # Main pipeline
    "run_nabqr_pipeline",
    # Visualization
    "visualize_results",
    # Core functions
    "variogram_score_single_observation",
    "variogram_score_R_multivariate",
    "calculate_crps",
    "calculate_qss",
    "pipeline",
    "generateAIReport_Gemini",
    # Helper functions
    "set_n_smallest_to_zero",
    "set_n_closest_to_zero",
    "quantile_score",
    "simulate_correlated_ar1_process",
    "simulate_wind_power_sde",
    # TAQR functions
    "rq_simplex_final",
    "one_step_quantile_prediction",
    "run_taqr",
]
