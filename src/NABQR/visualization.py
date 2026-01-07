import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
# --- Improvement 5: Logging ---
import logging

logger = logging.getLogger(__name__)


# --- Improvement 4: Quantile Visualization ---
def visualize_results(y_hat, q_hat, ylabel, quantiles=None):
    """Create a visualization of prediction intervals with actual values.

    Parameters
    ----------
    y_hat : numpy.ndarray
        Actual observed values
    q_hat : numpy.ndarray
        Predicted quantiles for different probability levels
    ylabel : str
        Label for the y-axis
    quantiles : list, optional
        The specific quantile levels (e.g., [0.1, 0.5, 0.9])

    Returns
    -------
    None
        Saves the plot as 'TEST_NABQR_taqr_pi_plot.pdf' and displays it

    Notes
    -----
    - Creates a filled plot showing prediction intervals using a blue gradient
    - Overlays actual values as a black line
    - Automatically adjusts x-axis date formatting
    """
    y_hat = pd.Series(np.array(y_hat).flatten())
    # --- Improvement 5: Logging ---
    try:
        taqr_results_corrected_plot = pd.DataFrame(np.array(q_hat).T, index=y_hat.index)
    except Exception as e:
        logger.debug(f"Failed to create DataFrame with transpose: {e}. Trying without transpose.")
        taqr_results_corrected_plot = pd.DataFrame(np.array(q_hat), index=y_hat.index)

    m = taqr_results_corrected_plot.shape[1]  # Ensemble size
    # Define the color gradient from dark blue to light cyan
    colors = [
        (173 / 255, 217 / 255, 229 / 255),
        (19 / 255, 25 / 255, 148 / 255),
        (173 / 255, 217 / 255, 229 / 255),
    ]
    cmap = LinearSegmentedColormap.from_list("blue_to_cyan", colors, N=100)
    norm = plt.Normalize(vmin=0, vmax=m - 2)  # Normalize for the ensemble size
    sm = ScalarMappable(cmap=cmap, norm=norm)

    plt.figure(figsize=(10, 6))
    for i in range(m - 1):
        color = sm.to_rgba(i)
        plt.fill_between(
            taqr_results_corrected_plot.index,
            taqr_results_corrected_plot.iloc[:, i],
            taqr_results_corrected_plot.iloc[:, i + 1],
            color=color,
            alpha=0.6,
        )
    
    # Plot the median line (0.5 quantile) if available
    if quantiles is not None and 0.5 in quantiles:
        # Convert to list to find index
        q_list = list(quantiles)
        if 0.5 in q_list:
            idx_05 = q_list.index(0.5)
            plt.plot(taqr_results_corrected_plot.index, taqr_results_corrected_plot.iloc[:, idx_05], 
                     color="darkblue", linestyle="--", linewidth=1.5, label="Median (0.50)")

    plt.plot(y_hat.index, y_hat, color="white", linewidth=3)  # White outline
    plt.plot(y_hat.index, y_hat, color="black", label="Actual Observations", linewidth=2)
    plt.xlim(y_hat.index[0], y_hat.index[-1])

    # Create legend elements
    handles, labels = plt.gca().get_legend_handles_labels()
    
    # Add a proxy for the PI (Prediction Interval)
    q_min = min(quantiles) if quantiles else "min"
    q_max = max(quantiles) if quantiles else "max"
    pi_proxy = Line2D([0], [0], color=sm.to_rgba(m // 2), lw=8, alpha=0.6, 
                      label=f"PI ({q_min} - {q_max})")
    handles.append(pi_proxy)
    
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1))

    plt.xlabel("Time")
    plt.ylabel(ylabel)

    # Configure date formatting on x-axis
    locator = mdates.AutoDateLocator(minticks=6, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax = plt.gca()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.savefig(f"{ylabel}_taqr_pi_plot.pdf")
    plt.show()
