import numpy as np
import pandas as pd
from typing import List, Tuple
import warnings
import matplotlib.pyplot as plt
from darts import TimeSeries
from adtk.detector import QuantileDetector, IQRDetector, AutoregressionAD
from adtk.data import validate_series
# from hampel import hampel  # Uncomment when enabling Hampel filter in future
# AD_Format must be defined in the same module or imported here
from scipy.interpolate import UnivariateSpline
from pathlib import Path

def AD_Format(series: pd.Series) -> List[Tuple[float, float]]:
    """
    Identify and merge multiplie continuous ranges if within range in a numeric pandas Series based on a fixed step size.

    This function:
      1. Scans through `series` and groups consecutive values into sections
         where each adjacent difference is within `tol` of `step`.
      2. Records each section as a (start, end) tuple.
      3. Merges any adjacent tuples whose gap is less than 10 units.

    Parameters
    ----------
    series : pd.Series
        A one-dimensional pandas Series of numeric values, assumed sorted ascending.

    Returns
    -------
    List[Tuple[float, float]]
        A list of (start, end) ranges covering contiguous segments of the series,
        with nearby segments merged when their gap < 10.
    """
    step = 0.0005
    tol = 0.005
    section = [series.iloc[0]]
    ranges: List[Tuple[float, float]] = []
    
    # Build initial ranges
    for i in range(1, len(series)):
        diff = series.iloc[i] - series.iloc[i-1]
        if abs(diff - step) < tol:
            section.append(series.iloc[i])
        else:
            ranges.append((section[0], section[-1]))
            section = [series.iloc[i]]
    if section:
        ranges.append((section[0], section[-1]))

    # Merge ranges separated by a small gap (< 10)
    merged: List[Tuple[float, float]] = []
    current_start, current_end = ranges[0]
    for next_start, next_end in ranges[1:]:
        gap = next_start - current_end
        if gap < 10:
            current_end = next_end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged.append((current_start, current_end))

    return merged

def anomaly_det(df,Q_high_quantile=0.99,IQR_scale=3,AD_c=5,
                AD_side="both",
                AD_n_steps=3,
                AD_step_size=100,
                H_window_size=10,
                H_summary=False,
                verbose=False,
                ignore_warn=True,
                name="Temp",
                Return_vals=False):
    """
    Detect anomalies in a respiratory time series using multiple detectors:
      - QuantileDetector
      - IQRDetector
      - AutoregressionAD

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with columns 'Time' and 'Respiration'.
    Q_high_quantile : float, default=0.99
        High quantile threshold for QuantileDetector.
    IQR_scale : float, default=3
        Scale factor for IQRDetector.
    AD_c : float, default=5
        Threshold multiplier for AutoregressionAD.
    AD_side : str, default="both"
        Direction of residuals to flag ('both', 'positive', 'negative').
    AD_n_steps : int, default=3
        Number of lag steps for AutoregressionAD.
    AD_step_size : int, default=100
        Step size between lags for AutoregressionAD.
    H_window_size : int, default=10
        Window size for Hampel filter (unused; kept for future).
    H_summary : bool, default=False
        If True, prints Hampel filter summary (when enabled).
    verbose : bool, default=False
        If True, prints progress messages.
    ignore_warn : bool, default=True
        If True, suppresses FutureWarnings.
    name : str, default="Temp"
        Base filename for saving plots.
    Return_vals : bool, default=False
        If True, returns (anomaly_mask, df_adtk).

    Returns
    -------
    If Return_vals:
        y : numpy.ndarray
            Binary mask (1 for anomaly, 0 otherwise).
        df_adtk : pandas.DataFrame
            ADTK DataFrame used for AutoregressionAD.
    Otherwise:
        Displays and saves anomaly detection plots.
    """
    if ignore_warn:
        warnings.filterwarnings("ignore", category=FutureWarning)

    # Convert to TimeSeries for ADTK detectors
    trimmed_series = TimeSeries.from_dataframe(df, value_cols=['Respiration'])
    df_adtk = df.copy()
    df_adtk.index = pd.to_datetime(df_adtk.index)

    if verbose:
        print("Fitting QuantileDetector...")
    detectorQ = QuantileDetector(high_quantile=Q_high_quantile)
    detectorQ.fit(trimmed_series)
    anomaliesQ = detectorQ.detect(trimmed_series)

    if verbose:
        print("Fitting IQRDetector...")
    detectorIQ = IQRDetector(scale=IQR_scale)
    detectorIQ.fit(trimmed_series)
    anomaliesIQ = detectorIQ.detect(trimmed_series)

    if verbose:
        print("Fitting AutoregressionAD...")
    df_adtk = validate_series(df_adtk).astype(float)
    autoregression_ad = AutoregressionAD(
        c=AD_c,
        side=AD_side,
        n_steps=AD_n_steps,
        step_size=AD_step_size
    )
    anomaliesAR = autoregression_ad.fit_detect(df_adtk).dropna(how='all')
    AD_vals = (anomaliesAR == True)
    mask = AD_vals.any(axis=1).values
    indices = np.where(mask)[0]

    intervals = AD_Format(pd.Series(df['Time'].values[indices]))
    if verbose:
        print("Anomaly intervals:", intervals)

    time_vals = df['Time'].values
    y = np.zeros_like(time_vals, dtype=int)
    for start, stop in intervals:
        y[(time_vals >= start) & (time_vals <= stop)] = 1

    # # Hampel filter (commented for future use)
    # hampel_eval = hampel(df['Respiration'], window_size=H_window_size)
    # detectorH = np.zeros(len(df))
    # detectorH[hampel_eval.outlier_indices] = 1
    # if H_summary:
    #     print("Hampel filtered data:", hampel_eval.filtered_data)

    # Plot detectors
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(df['Time'], df['Respiration'], 'g', label='Respiration')
    plt.title('Respiration Signal')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(df['Time'], anomaliesQ.values(), 'r', label='QuantileDetector')
    plt.title('Quantile Detection')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(df['Time'], anomaliesIQ.values(), 'r', label='IQRDetector')
    plt.title('IQR Detection')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(df['Time'], y, 'r', label='AutoregressionAD')
    plt.title('AutoregressionAD Detection')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{name}_AnomalyDetection.png', bbox_inches='tight')
    plt.show()

    if Return_vals:
        return y, df_adtk

def DataCleaning(
    file_path: str,
    AD_c: float = 4,
    AD_side: str = "both",
    AD_n_steps: int = 3,
    AD_step_size: int = 50,
    spline_s: float = 1.0,
    spline_k: int = 3,
    head_trim_secs: float = 5.0,
    spline_gap_max: float = 1.0,
    verbose: bool = False,
    show_plots: bool = True
):
    """
    Clean respiratory data by:
      1. Detecting anomalies and setting them to NaN.
      2. Trimming any NaNs within the first `head_trim_secs` seconds.
      3. Interpolating small gaps (<= `spline_gap_max` seconds) via spline.
      4. Trimming trailing NaNs (remove data after the first remaining NaN).
    Optionally saves diagnostic plots at each step.

    Parameters
    ----------
    file_path : str
        Path to input CSV file with 'Time' and 'Respiration'.
    AD_c, AD_side, AD_n_steps, AD_step_size : anomaly_det params
    spline_s : float
        Smoothing factor for UnivariateSpline.
    spline_k : int
        Spline degree.
    head_trim_secs : float
        Seconds threshold for trimming initial NaNs.
    spline_gap_max : float
        Seconds threshold for spline gap filling.
    verbose : bool
        If True, print progress messages.
    show_plots : bool
        If True, save and display plots at each stage.

    Returns
    -------
    pd.DataFrame
        The cleaned DataFrame, trimmed and interpolated.
    """
    path = Path(file_path)
    df = pd.read_csv(path)
    
    # 1) Anomaly detection
    if verbose: print("1) Running anomaly_det...")
    y_mask, _ = anomaly_det(
        df, verbose=verbose,
        AD_c=AD_c, AD_side=AD_side,
        AD_n_steps=AD_n_steps,
        AD_step_size=AD_step_size,
        Return_vals=True
    )
    df.loc[y_mask.astype(bool), 'Respiration'] = np.nan

    if show_plots:
        if verbose: print("   Plotting anomalies removed")
        plt.figure(figsize=(10,5))
        plt.plot(df['Time'], df['Respiration'], 'b-', label="Resp with NaNs")
        plt.legend(); plt.tight_layout()
        plt.savefig(path.parent/f"{path.stem}_AnomaliesRemoved.png", bbox_inches='tight')
        plt.close()
    
    # 2) Trim head NaNs
    if verbose: print(f"2) Trimming initial NaN run within {head_trim_secs}s...")
    df = df.sort_values('Time').reset_index(drop=True)
    nan_idxs = np.where(df['Respiration'].isna())[0]
    if nan_idxs.size:
        runs_all = np.split(nan_idxs, np.where(np.diff(nan_idxs)!=1)[0]+1)
        first_window_idx = df.index[df['Time']<=head_trim_secs].max()
        for run in runs_all:
            if run[0] <= first_window_idx:
                end_idx = run[-1]
                df = df.iloc[end_idx+1:].reset_index(drop=True)
                if verbose: print(f"   Dropped rows up to index {end_idx}")
                break

    # 3) Spline interpolate small gaps
    if verbose: print(f"3) Spline interpolation for gaps ≤ {spline_gap_max}s")
    mask_valid = ~df['Respiration'].isna()
    spline = UnivariateSpline(
        df.loc[mask_valid,'Time'],
        df.loc[mask_valid,'Respiration'],
        s=spline_s, k=spline_k
    )
    nan_idxs = np.where(df['Respiration'].isna())[0]
    runs = np.split(nan_idxs, np.where(np.diff(nan_idxs)!=1)[0]+1)
    for run in runs:
        if run.size:
            start, end = run[0], run[-1]
            duration = df.loc[end,'Time'] - df.loc[start,'Time']
            if duration <= spline_gap_max:
                if verbose: print(f"   Filling run {start}-{end}, {duration:.3f}s")
                df.loc[run,'Respiration'] = spline(df.loc[run,'Time'])

    if show_plots:
        if verbose: print("   Plotting spline-interpolated data")
        plt.figure(figsize=(10,5))
        plt.plot(df['Time'], df['Respiration'], 'b-', label="Post-spline")
        plt.legend(); plt.tight_layout()
        plt.savefig(path.parent/f"{path.stem}_SplineInterpolation.png", bbox_inches='tight')
        plt.close()
    
    # 4) Trim trailing NaNs
    if verbose: print("4) Trimming trailing NaNs after first remaining NaN")
    trailing = df.index[df['Respiration'].isna()]
    if trailing.any():
        first_nan = trailing[0]
        df = df.iloc[:first_nan].reset_index(drop=True)
        if verbose: print(f"   Data trimmed to index {first_nan-1}")

    if show_plots:
        if verbose: print("   Plotting final cleaned data")
        plt.figure(figsize=(10,5))
        plt.plot(df['Time'], df['Respiration'], 'b-', label="Final Clean")
        plt.legend(); plt.tight_layout()
        plt.savefig(path.parent/f"{path.stem}_FinalCleaned.png", bbox_inches='tight')
        plt.close()

    return df