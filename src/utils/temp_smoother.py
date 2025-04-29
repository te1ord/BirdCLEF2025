import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, medfilt

def moving_average_filter(data: np.ndarray, window_size: int) -> np.ndarray:
    """Apply a simple moving average filter along the time axis."""
    kernel = np.ones(window_size, dtype=np.float32) / window_size
    return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), axis=0, arr=data)


def triangular_filter(data: np.ndarray, window_size: int) -> np.ndarray:
    """Apply a triangular (Bartlett) filter along the time axis."""
    tri = np.bartlett(window_size).astype(np.float32)
    tri /= tri.sum()
    return np.apply_along_axis(lambda x: np.convolve(x, tri, mode='same'), axis=0, arr=data)


def savitzky_golay_filter_fn(data: np.ndarray, window_size: int, polyorder: int) -> np.ndarray:
    """Apply Savitzky–Golay filter along the time axis."""
    if window_size % 2 == 0:
        raise ValueError("Savitzky–Golay filter 'window_size' must be odd.")
    return savgol_filter(data, window_length=window_size, polyorder=polyorder, axis=0, mode='interp')


def median_filter_fn(data: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply median filtering along the time axis."""
    if kernel_size % 2 == 0:
        raise ValueError("Median filter 'kernel_size' must be odd.")
    return np.apply_along_axis(lambda x: medfilt(x, kernel_size), axis=0, arr=data)


def exponential_filter(data: np.ndarray, alpha: float) -> np.ndarray:
    """Apply exponential moving average along the time axis."""
    if not (0 < alpha <= 1):
        raise ValueError("Exponential filter 'alpha' must be in (0, 1].")
    ema = np.zeros_like(data, dtype=np.float32)
    ema[0] = data[0]
    for t in range(1, data.shape[0]):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t - 1]
    return ema


def neighbor_average_filter(
    data: np.ndarray,
    middle_weights: dict,
    edge_weights: dict
) -> np.ndarray:
    """Apply simple weighted average with immediate neighbors."""
    new = data.copy()
    n = data.shape[0]
    if n == 0:
        return new
    # middle positions
    for i in range(1, n - 1):
        new[i] = (
            data[i - 1] * middle_weights['prev'] +
            data[i]     * middle_weights['curr'] +
            data[i + 1] * middle_weights['next']
        )
    # edges
    if n > 1:
        new[0] = (
            data[0] * edge_weights['main'] +
            data[1] * edge_weights['neighbor']
        )
        new[-1] = (
            data[-1] * edge_weights['main'] +
            data[-2] * edge_weights['neighbor']
        )
    return new

class Smoothing:
    def __init__(self, method: str, params: dict, class_labels: list):
        """
        method: one of 'moving_average', 'triangular', 'savitzky', 'median', 'exponential', 'neighbor_average'
        params: must exactly contain required keys for the chosen method:
          - moving_average / triangular: {'window_size': int}
          - savitzky: {'window_size': int, 'polyorder': int}
          - median: {'kernel_size': int}
          - exponential: {'alpha': float}
          - neighbor_average: {'middle_weights': dict, 'edge_weights': dict}
        class_labels: list of column names to smooth
        """
        self.method = method
        self.params = params
        self.class_labels = class_labels

        required = {
            'moving_average': ['window_size'],
            'triangular': ['window_size'],
            'savitzky': ['window_size', 'polyorder'],
            'median': ['kernel_size'],
            'exponential': ['alpha'],
            'neighbor_average': ['middle_weights', 'edge_weights'],
        }
        
        if method not in required:
            raise ValueError(f"Unknown smoothing method '{method}'")
        missing = [k for k in required[method] if k not in params]
        if missing:
            raise ValueError(f"Missing params for {method}: {missing}")

    def apply(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Apply the selected smoothing over each contiguous group of row_id."""
        cols = self.class_labels
        groups = predictions['row_id'].str.rsplit('_', n=1).str[0].values
        output = predictions.copy()

        for grp in np.unique(groups):
            mask = (groups == grp)
            chunk_vals = predictions.loc[mask, cols].values

            if self.method == 'moving_average':
                sm = moving_average_filter(chunk_vals, self.params['window_size'])
            elif self.method == 'triangular':
                sm = triangular_filter(chunk_vals, self.params['window_size'])
            elif self.method == 'savitzky':
                sm = savitzky_golay_filter_fn(
                    chunk_vals,
                    self.params['window_size'],
                    self.params['polyorder']
                )
            elif self.method == 'median':
                sm = median_filter_fn(chunk_vals, self.params['kernel_size'])
            elif self.method == 'exponential':
                sm = exponential_filter(chunk_vals, self.params['alpha'])
            elif self.method == 'neighbor_average':
                sm = neighbor_average_filter(
                    chunk_vals,
                    self.params['middle_weights'],
                    self.params['edge_weights']
                )
            else:
                sm = chunk_vals.copy()

            output.loc[mask, cols] = sm

        return output

