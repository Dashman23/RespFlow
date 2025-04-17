import numpy as np
import pandas as pd
from scipy import signal
from pathlib import Path
import os
import shutil

#This will be the preprocessing and filtrering

#Make all Parameters needed for Butterworth Bandpass filter

def ApplyBandpass(df, sm_rate, lw_cut=0.05, hg_cut=2.0, order=1, output='None'):
    """
    Apply a zero‑phase Butterworth bandpass filter to a respiratory time series.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with at least these columns:
          - 'Time': time stamps
          - 'Respiration': raw respiration signal
          - 'Events': (optional) event markers
    sm_rate : int
        Sampling rate in Hz.
    lw_cut : float, default=0.05
        Low‑cut frequency of the passband (Hz).
    hg_cut : float, default=2.0
        High‑cut frequency of the passband (Hz).
    order : int, default=1
        Order of the Butterworth filter.
    output : str, default='None'
        File path to write the filtered DataFrame as CSV. If 'None', no file is saved.

    Returns
    -------
    numpy.ndarray
        The filtered respiration signal as a NumPy array.
    """
    # Create Butterworth bandpass filter coefficients
    b, a = signal.butter(order, [lw_cut, hg_cut], fs=sm_rate, btype='bandpass')
    
    # Apply zero‑phase filtering (filtfilt) to the raw respiration data
    resp_data = df['Respiration'].values
    filt_data = signal.filtfilt(b, a, resp_data)
    
    # If an output path is provided, save the filtered results
    if output != 'None':
        df_filt = pd.DataFrame({
            'Time': df['Time'].values,
            'Respiration': filt_data,
            'Events': df['Events'].values
        })
        df_filt.to_csv(output, index=False)
    
    return filt_data

def Preprocessing(input_raw_dir,output_filtered_dir,sm_rate=2000,lw_cut=0.05,hg_cut=2.0,pad_len=0,order=5):
    """
    Iterate through all participant subfolders in `input_raw_dir`, apply a
    Butterworth bandpass filter (via ApplyBandpass) to each CSV's
    respiration signal, and save the filtered output (Time, Respiration, Events)
    into `output_filtered_dir`.

    Parameters
    ----------
    input_raw_dir : str or Path
        Folder of raw CSV subdirectories.
    output_filtered_dir : str or Path
        Where filtered CSVs will be written, in subfolders '01/', '02/', …
    sm_rate : int, default=2000
        Sampling rate (Hz).
    lw_cut : float, default=0.05
        Low‑cut frequency (Hz).
    hg_cut : float, default=2.0
        High‑cut frequency (Hz).
    pad_len : int, default=0
        Padding length for the filter (currently unused).
    order : int, default=5
        Butterworth filter order (must be ≥ 1).
    """
    raw_path = Path(input_raw_dir)
    out_base = Path(output_filtered_dir)
    count = 1

    for participant_dir in sorted(raw_path.iterdir()):
        if not participant_dir.is_dir():
            continue

        out_dir = out_base / f"{count:02d}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for csv_file in participant_dir.glob("*.csv"):
            df = pd.read_csv(csv_file)
            data_id = csv_file.stem[-8:]
            output_csv = out_dir / f"1_{data_id}.csv"

            ApplyBandpass(
                df,
                sm_rate=sm_rate,
                lw_cut=lw_cut,
                hg_cut=hg_cut,
                order=order,
                output=str(output_csv)
            )
        count += 1