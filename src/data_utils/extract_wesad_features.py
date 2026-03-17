# import numpy as np
# import neurokit2 as nk
# import scipy.stats as stats
# import scipy.signal as signal
# import pandas as pd
# from joblib import Parallel, delayed
# from tqdm import tqdm

# def get_peak_frequency(sig, fs):
#     if len(sig) == 0: return 0
#     freqs, psd = signal.welch(sig, fs=fs, nperseg=min(len(sig), int(fs*2)))
#     return freqs[np.argmax(psd)]

# def butter_lowpass_filter(data, cutoff, fs, order=4):
#     nyq = 0.5 * fs
#     if cutoff >= nyq:
#         return data  # Bypass: Cannot filter frequencies above Nyquist limit
#     b, a = signal.butter(order, cutoff / nyq, btype='low', analog=False)
#     return signal.filtfilt(b, a, data)

# def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
#     nyq = 0.5 * fs
#     if highcut >= nyq:
#         return data  # Bypass
#     b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
#     return signal.filtfilt(b, a, data)

# def butter_highpass_filter(data, cutoff, fs, order=4):
#     nyq = 0.5 * fs
#     if cutoff >= nyq:
#         return data  # Bypass
#     b, a = signal.butter(order, cutoff / nyq, btype='high', analog=False)
#     return signal.filtfilt(b, a, data)

# def extract_single_modality(mod_name, sig, fs):
#     features = {}
#     prefix = f"{mod_name}_"
    
#     # 1. ACCELEROMETER
#     if 'ACC' in mod_name:
#         sub_window_len = int(5 * fs) 
#         num_subs = max(1, sig.shape[1] // sub_window_len)
#         sub_stats = {axis: {'mean':[], 'std':[], 'int':[], 'peak':[]} for axis in ['x','y','z','3d']}
        
#         for i in range(num_subs):
#             sub_sig = sig[:, i*sub_window_len : (i+1)*sub_window_len]
#             sub_3d = np.sqrt(np.sum(sub_sig**2, axis=0))
#             for ax_idx, axis in enumerate(['x', 'y', 'z']):
#                 sub_stats[axis]['mean'].append(np.mean(sub_sig[ax_idx]))
#                 sub_stats[axis]['std'].append(np.std(sub_sig[ax_idx]))
#                 sub_stats[axis]['int'].append(np.sum(np.abs(sub_sig[ax_idx])))
#                 sub_stats[axis]['peak'].append(get_peak_frequency(sub_sig[ax_idx], fs))
#             sub_stats['3d']['mean'].append(np.mean(sub_3d))
#             sub_stats['3d']['std'].append(np.std(sub_3d))
#             sub_stats['3d']['int'].append(np.sum(np.abs(sub_3d)))
            
#         for axis in ['x', 'y', 'z', '3d']:
#             features[f'{prefix}{axis}_mean'] = np.mean(sub_stats[axis]['mean'])
#             features[f'{prefix}{axis}_std'] = np.mean(sub_stats[axis]['std'])
#             features[f'{prefix}{axis}_integral'] = np.mean(sub_stats[axis]['int'])
#             if axis != '3d':
#                 features[f'{prefix}{axis}_peak_freq'] = np.mean(sub_stats[axis]['peak'])

#     # 2. ECG & BVP
#     elif 'ECG' in mod_name or 'BVP' in mod_name:
#         try:
#             if 'ECG' in mod_name:
#                 cleaned = nk.ecg_clean(sig, sampling_rate=fs)
#                 peaks, _ = nk.ecg_peaks(cleaned, sampling_rate=fs)
#             else:
#                 cleaned = nk.ppg_clean(sig, sampling_rate=fs)
#                 peaks, _ = nk.ppg_peaks(cleaned, sampling_rate=fs)
                
#             hrv = nk.hrv(peaks, sampling_rate=fs, show=False)
            
#             features[f'{prefix}HR_mean'] = hrv['HRV_MeanNN'].values[0]
#             features[f'{prefix}HR_std'] = hrv['HRV_SDNN'].values[0]
#             features[f'{prefix}HRV_mean'] = hrv['HRV_MeanNN'].values[0]
#             features[f'{prefix}HRV_std'] = hrv['HRV_SDNN'].values[0]
#             features[f'{prefix}rmsHRV'] = hrv['HRV_RMSSD'].values[0]
#             features[f'{prefix}NN50'] = hrv['HRV_NN50'].values[0]
#             features[f'{prefix}pNN50'] = hrv['HRV_pNN50'].values[0]
#             features[f'{prefix}TINN'] = hrv['HRV_TINN'].values[0]
            
#             features[f'{prefix}ULF'] = hrv['HRV_ULF'].values[0] if 'HRV_ULF' in hrv.columns else 0.0
#             features[f'{prefix}LF'] = hrv['HRV_LF'].values[0]
#             features[f'{prefix}HF'] = hrv['HRV_HF'].values[0]
#             features[f'{prefix}UHF'] = hrv['HRV_UHF'].values[0] if 'HRV_UHF' in hrv.columns else 0.0
            
#             total_power = features[f'{prefix}ULF'] + features[f'{prefix}LF'] + features[f'{prefix}HF'] + features[f'{prefix}UHF']
#             features[f'{prefix}sum_freq'] = total_power
#             features[f'{prefix}LF_HF_ratio'] = hrv['HRV_LFHF'].values[0]
#             features[f'{prefix}LF_norm'] = hrv['HRV_LFn'].values[0]
#             features[f'{prefix}HF_norm'] = hrv['HRV_HFn'].values[0]
#             features[f'{prefix}rel_ULF'] = features[f'{prefix}ULF'] / (total_power + 1e-5)
#             features[f'{prefix}rel_LF'] = features[f'{prefix}LF'] / (total_power + 1e-5)
#             features[f'{prefix}rel_HF'] = features[f'{prefix}HF'] / (total_power + 1e-5)
#             features[f'{prefix}rel_UHF'] = features[f'{prefix}UHF'] / (total_power + 1e-5)
            
#         except Exception as e:
#             keys = ['HR_mean', 'HR_std', 'HRV_mean', 'HRV_std', 'rmsHRV', 'NN50', 'pNN50', 'TINN', 
#                     'ULF', 'LF', 'HF', 'UHF', 'sum_freq', 'LF_HF_ratio', 'LF_norm', 'HF_norm', 'rel_ULF', 'rel_LF', 'rel_HF', 'rel_UHF']
#             features.update({f'{prefix}{k}': 0.0 for k in keys})

#     # 3. EDA
#     elif 'EDA' in mod_name:
#         try:
#             eda_clean = butter_lowpass_filter(sig, cutoff=5.0, fs=fs)
#             features[f'{prefix}mean'] = np.mean(eda_clean)
#             features[f'{prefix}std'] = np.std(eda_clean)
#             features[f'{prefix}min'] = np.min(eda_clean)
#             features[f'{prefix}max'] = np.max(eda_clean)
#             features[f'{prefix}range'] = np.max(eda_clean) - np.min(eda_clean)
#             features[f'{prefix}slope'] = np.polyfit(np.arange(len(eda_clean)), eda_clean, 1)[0]
            
#             eda_signals, info = nk.eda_process(eda_clean, sampling_rate=fs)
#             scl = eda_signals['EDA_Tonic']
#             scr = eda_signals['EDA_Phasic']
            
#             features[f'{prefix}SCL_mean'] = np.mean(scl)
#             features[f'{prefix}SCL_std'] = np.std(scl)
#             features[f'{prefix}SCR_std'] = np.std(scr)
#             features[f'{prefix}SCL_time_corr'] = stats.pearsonr(scl, np.arange(len(scl)))[0]
            
#             peaks = info['SCR_Peaks']
#             features[f'{prefix}num_SCR'] = len(peaks)
#             if len(peaks) > 0:
#                 features[f'{prefix}sum_Amp_SCR'] = np.sum(info['SCR_Amplitude'])
#                 features[f'{prefix}sum_t_SCR'] = np.sum(info['SCR_RecoveryTime']) 
#                 features[f'{prefix}integral_SCR'] = np.sum(np.abs(scr)) 
#             else:
#                 features.update({f'{prefix}sum_Amp_SCR': 0, f'{prefix}sum_t_SCR': 0, f'{prefix}integral_SCR': 0})
#         except Exception as e:
#             keys = ['mean', 'std', 'min', 'max', 'range', 'slope', 'SCL_mean', 'SCL_std', 'SCR_std', 
#                     'SCL_time_corr', 'num_SCR', 'sum_Amp_SCR', 'sum_t_SCR', 'integral_SCR']
#             features.update({f'{prefix}{k}': 0.0 for k in keys})

#     # 4. EMG
#     elif 'EMG' in mod_name:
#         emg_chain1 = butter_highpass_filter(sig, cutoff=1.0, fs=fs)
#         features[f'{prefix}mean'] = np.mean(emg_chain1)
#         features[f'{prefix}std'] = np.std(emg_chain1)
#         features[f'{prefix}range'] = np.max(emg_chain1) - np.min(emg_chain1)
#         features[f'{prefix}integral'] = np.sum(np.abs(emg_chain1))
#         features[f'{prefix}median'] = np.median(emg_chain1)
#         features[f'{prefix}p10'] = np.percentile(emg_chain1, 10)
#         features[f'{prefix}p90'] = np.percentile(emg_chain1, 90)
        
#         sub_window_len = int(5 * fs)
#         num_subs = max(1, len(emg_chain1) // sub_window_len)
#         freqs_list, median_freqs_list, peak_freqs_list = [], [], [] 
#         psd_bands = {f'psd_{b*50}_{(b+1)*50}': [] for b in range(7)} 
#         for i in range(num_subs):
#             sub_sig = emg_chain1[i*sub_window_len : (i+1)*sub_window_len]
#             if len(sub_sig) > 0:
#                 peak_freqs_list.append(get_peak_frequency(sub_sig, fs)) 
#                 f, psd = signal.welch(sub_sig, fs=fs, nperseg=min(len(sub_sig), int(fs*2)))
#                 if len(psd) > 0:
#                     freqs_list.append(f[np.argmax(psd)])
#                     cumulative = np.cumsum(psd)
#                     median_idx = np.where(cumulative >= cumulative[-1]/2)[0]
#                     if len(median_idx) > 0: median_freqs_list.append(f[median_idx[0]])
#                     for b in range(7):
#                         low, high = b*50, (b+1)*50
#                         idx = np.logical_and(f >= low, f < high)
#                         psd_bands[f'psd_{low}_{high}'].append(np.sum(psd[idx]))

#         features[f'{prefix}mean_freq'] = np.mean(freqs_list) if freqs_list else 0
#         features[f'{prefix}median_freq'] = np.mean(median_freqs_list) if median_freqs_list else 0
#         features[f'{prefix}peak_freq'] = np.mean(peak_freqs_list) if peak_freqs_list else 0
#         for b_name in psd_bands.keys():
#             features[f'{prefix}{b_name}'] = np.mean(psd_bands[b_name]) if psd_bands[b_name] else 0

#         emg_chain2 = butter_lowpass_filter(sig, cutoff=50.0, fs=fs)
#         peaks, _ = signal.find_peaks(emg_chain2)
#         features[f'{prefix}num_peaks'] = len(peaks)
#         if len(peaks) > 0:
#             amps = emg_chain2[peaks]
#             features[f'{prefix}mean_amp'] = np.mean(amps)
#             features[f'{prefix}std_amp'] = np.std(amps)
#             features[f'{prefix}sum_amp'] = np.sum(amps)
#             features[f'{prefix}norm_sum_amp'] = np.sum(amps) / len(peaks)
#         else:
#             features.update({f'{prefix}mean_amp': 0, f'{prefix}std_amp': 0, f'{prefix}sum_amp': 0, f'{prefix}norm_sum_amp': 0})

#     # 5. RESPIRATION
#     elif 'RESP' in mod_name:
#         try:
#             resp_clean = butter_bandpass_filter(sig, lowcut=0.1, highcut=0.35, fs=fs)
#             resp_signals, info = nk.rsp_process(resp_clean, sampling_rate=fs)
            
#             features[f'{prefix}rate'] = np.mean(resp_signals['RSP_Rate'])
#             features[f'{prefix}stretch'] = np.max(resp_clean) - np.min(resp_clean)
#             features[f'{prefix}duration'] = len(sig) / fs
            
#             phases = resp_signals['RSP_Phase'].values
#             diffs = np.diff(phases)
#             phase_changes = np.where(diffs != 0)[0]
            
#             inhale_durs, exhale_durs = [], []
#             for i in range(1, len(phase_changes)):
#                 dur = (phase_changes[i] - phase_changes[i-1]) / fs
#                 if phases[phase_changes[i-1] + 1] == 1: inhale_durs.append(dur)
#                 else: exhale_durs.append(dur)
                    
#             features[f'{prefix}inhale_mean'] = np.mean(inhale_durs) if inhale_durs else 0
#             features[f'{prefix}inhale_std'] = np.std(inhale_durs) if inhale_durs else 0
#             features[f'{prefix}exhale_mean'] = np.mean(exhale_durs) if exhale_durs else 0
#             features[f'{prefix}exhale_std'] = np.std(exhale_durs) if exhale_durs else 0
            
#             features[f'{prefix}insp_volume'] = np.sum(resp_clean[phases == 1])
#             features[f'{prefix}I_E_ratio'] = np.sum(phases == 1) / (np.sum(phases == 0) + 1e-5)
#         except Exception as e:
#             keys = ['rate', 'stretch', 'duration', 'inhale_mean', 'inhale_std', 'exhale_mean', 'exhale_std', 'I_E_ratio', 'insp_volume']
#             features.update({f'{prefix}{k}': 0.0 for k in keys})

#     # 6. TEMPERATURE
#     elif 'TEMP' in mod_name:
#         features[f'{prefix}mean'] = np.mean(sig)
#         features[f'{prefix}std'] = np.std(sig)
#         features[f'{prefix}min'] = np.min(sig)
#         features[f'{prefix}max'] = np.max(sig)
#         features[f'{prefix}range'] = np.max(sig) - np.min(sig)
#         features[f'{prefix}slope'] = np.polyfit(np.arange(len(sig)), sig, 1)[0]

#     return features

# def extract_all_windows(data_dict, fs_dict, window_size_sec=60, shift_sec=0.25, n_jobs=-1):
#     first_mod = list(data_dict.keys())[0]
#     max_time_sec = data_dict[first_mod].shape[-1] / fs_dict[first_mod]
#     start_times = np.arange(0, max_time_sec - window_size_sec, shift_sec)
    
#     def process_single_window(current_time):
#         window_feats = {}
#         for mod, sig in data_dict.items():
#             fs = fs_dict[mod]
#             start_idx = int(current_time * fs)
#             end_idx = int((current_time + window_size_sec) * fs)
            
#             window_sig = sig[start_idx:end_idx] if sig.ndim == 1 else sig[:, start_idx:end_idx]
#             mod_features = extract_single_modality(mod, window_sig, fs)
#             window_feats.update(mod_features)
            
#         return window_feats

#     feature_matrix = Parallel(n_jobs=n_jobs)(
#         delayed(process_single_window)(t) for t in tqdm(start_times, desc="  Extracting Windows", leave=False)
#     )
        
#     return feature_matrix
import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import scipy.interpolate as interpolate
from joblib import Parallel, delayed
from tqdm import tqdm


# ---------------------------------------------------------------------------
# FILTER HELPERS
# ---------------------------------------------------------------------------

def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    if cutoff >= nyq:
        return data
    b, a = signal.butter(order, cutoff / nyq, btype='low', analog=False)
    return signal.filtfilt(b, a, data)


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    if highcut >= nyq or lowcut <= 0:
        return data
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return signal.filtfilt(b, a, data)


def butter_highpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    if cutoff >= nyq:
        return data
    b, a = signal.butter(order, cutoff / nyq, btype='high', analog=False)
    return signal.filtfilt(b, a, data)


def get_peak_frequency(sig, fs):
    if len(sig) < 4:
        return 0.0
    nperseg = min(len(sig), int(fs * 2))
    freqs, psd = signal.welch(sig, fs=fs, nperseg=nperseg)
    return float(freqs[np.argmax(psd)])


# ---------------------------------------------------------------------------
# ECG / BVP  —  pure scipy R-peak / PPG-peak detector + HRV
# ---------------------------------------------------------------------------

def _find_ecg_peaks(sig, fs):
    """
    Simple Pan-Tompkins-inspired R-peak detector using scipy only.
    Works on both ECG and BVP (PPG) signals.
    """
    # 1. Bandpass to isolate QRS / pulse band
    if 'bvp' in _find_ecg_peaks._mod_hint:
        filtered = butter_bandpass_filter(sig, lowcut=0.5, highcut=4.0, fs=fs)
    else:
        filtered = butter_bandpass_filter(sig, lowcut=5.0, highcut=15.0, fs=fs)

    # 2. Differentiate + square to emphasise slopes
    diff_sig  = np.diff(filtered, prepend=filtered[0])
    squared   = diff_sig ** 2

    # 3. Moving-window integration (~150 ms)
    win = max(1, int(0.15 * fs))
    kernel    = np.ones(win) / win
    integrated = np.convolve(squared, kernel, mode='same')

    # 4. Adaptive threshold: 50 % of signal max, refractory 200 ms
    threshold   = 0.5 * np.max(integrated)
    min_dist    = int(0.2 * fs)
    peaks, _    = signal.find_peaks(integrated, height=threshold, distance=min_dist)
    return peaks

_find_ecg_peaks._mod_hint = 'ecg'   # will be overwritten per call below


def _hrv_features(peaks, fs, prefix):
    """
    Compute time-domain and frequency-domain HRV from an array of peak indices.
    Returns a dict of features; fills zeros if fewer than 2 peaks found.
    """
    keys_zero = [
        'HR_mean', 'HR_std', 'HRV_mean', 'HRV_std', 'rmsHRV',
        'NN50', 'pNN50', 'TINN',
        'ULF', 'LF', 'HF', 'UHF',
        'sum_freq', 'LF_HF_ratio', 'LF_norm', 'HF_norm',
        'rel_ULF', 'rel_LF', 'rel_HF', 'rel_UHF',
    ]
    features = {f'{prefix}{k}': 0.0 for k in keys_zero}

    if len(peaks) < 2:
        return features

    # --- RR intervals in milliseconds ---
    rr_ms = np.diff(peaks) / fs * 1000.0

    # Remove physiologically implausible RR intervals (< 300 ms or > 2000 ms)
    rr_ms = rr_ms[(rr_ms > 300) & (rr_ms < 2000)]
    if len(rr_ms) < 2:
        return features

    # --- Time domain ---
    mean_rr  = float(np.mean(rr_ms))
    sdnn     = float(np.std(rr_ms, ddof=1))
    rmssd    = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
    nn50     = int(np.sum(np.abs(np.diff(rr_ms)) > 50))
    pnn50    = float(nn50 / len(rr_ms) * 100)

    # TINN approximation: width of the triangular interpolation of the RR histogram
    hist, bin_edges = np.histogram(rr_ms, bins=max(10, len(rr_ms) // 5))
    tinn = float(bin_edges[-1] - bin_edges[0])

    hr_mean = 60000.0 / mean_rr if mean_rr > 0 else 0.0
    hr_std  = float(np.std(60000.0 / rr_ms, ddof=1)) if len(rr_ms) > 1 else 0.0

    features[f'{prefix}HR_mean']  = hr_mean
    features[f'{prefix}HR_std']   = hr_std
    features[f'{prefix}HRV_mean'] = mean_rr
    features[f'{prefix}HRV_std']  = sdnn
    features[f'{prefix}rmsHRV']   = rmssd
    features[f'{prefix}NN50']     = float(nn50)
    features[f'{prefix}pNN50']    = pnn50
    features[f'{prefix}TINN']     = tinn

    # --- Frequency domain via Welch on uniformly resampled RR series ---
    # Resample RR series to 4 Hz (standard in HRV literature)
    try:
        peak_times = peaks[1:] / fs          # time of each RR interval end
        rr_times   = peak_times              # same length as rr_ms after filtering
        # after physiological filtering peak_times may be longer than rr_ms;
        # trim to same length
        min_len    = min(len(rr_times), len(rr_ms))
        rr_times   = rr_times[:min_len]
        rr_ms_trim = rr_ms[:min_len]

        resample_fs  = 4.0
        t_uniform    = np.arange(rr_times[0], rr_times[-1], 1.0 / resample_fs)
        if len(t_uniform) < 8:
            raise ValueError("RR series too short for frequency analysis")

        interp_fn    = interpolate.interp1d(rr_times, rr_ms_trim, kind='linear',
                                            bounds_error=False,
                                            fill_value=(rr_ms_trim[0], rr_ms_trim[-1]))
        rr_uniform   = interp_fn(t_uniform)

        nperseg      = min(len(rr_uniform), int(resample_fs * 60))  # max 60-s segment
        freqs, psd   = signal.welch(rr_uniform, fs=resample_fs, nperseg=nperseg)

        def _band_power(lo, hi):
            idx = (freqs >= lo) & (freqs < hi)
            return float(np.trapz(psd[idx], freqs[idx])) if idx.any() else 0.0

        ulf = _band_power(0.0,  0.003)
        lf  = _band_power(0.04, 0.15)
        hf  = _band_power(0.15, 0.4)
        uhf = _band_power(0.4,  1.0)
        total = ulf + lf + hf + uhf + 1e-10

        features[f'{prefix}ULF']        = ulf
        features[f'{prefix}LF']         = lf
        features[f'{prefix}HF']         = hf
        features[f'{prefix}UHF']        = uhf
        features[f'{prefix}sum_freq']   = total
        features[f'{prefix}LF_HF_ratio']= lf / (hf + 1e-10)
        features[f'{prefix}LF_norm']    = lf / (lf + hf + 1e-10)
        features[f'{prefix}HF_norm']    = hf / (lf + hf + 1e-10)
        features[f'{prefix}rel_ULF']    = ulf / total
        features[f'{prefix}rel_LF']     = lf  / total
        features[f'{prefix}rel_HF']     = hf  / total
        features[f'{prefix}rel_UHF']    = uhf / total

    except Exception:
        pass   # frequency features stay at 0.0 if RR series is too short

    return features


# ---------------------------------------------------------------------------
# EDA  —  tonic/phasic split via moving-average, peaks via scipy
# ---------------------------------------------------------------------------

def _eda_features(sig, fs, prefix):
    features = {}
    try:
        # 1. Lowpass to remove HF noise
        eda_clean = butter_lowpass_filter(sig, cutoff=5.0, fs=fs)

        # Basic stats
        features[f'{prefix}mean']  = float(np.mean(eda_clean))
        features[f'{prefix}std']   = float(np.std(eda_clean))
        features[f'{prefix}min']   = float(np.min(eda_clean))
        features[f'{prefix}max']   = float(np.max(eda_clean))
        features[f'{prefix}range'] = float(np.max(eda_clean) - np.min(eda_clean))
        features[f'{prefix}slope'] = float(np.polyfit(np.arange(len(eda_clean)), eda_clean, 1)[0])

        # 2. Tonic (SCL): very slow moving average (~4 s window)
        scl_win = max(1, int(4.0 * fs))
        scl_kernel = np.ones(scl_win) / scl_win
        scl = np.convolve(eda_clean, scl_kernel, mode='same')

        # 3. Phasic (SCR): residual after removing tonic
        scr = eda_clean - scl

        features[f'{prefix}SCL_mean']      = float(np.mean(scl))
        features[f'{prefix}SCL_std']       = float(np.std(scl))
        features[f'{prefix}SCR_std']       = float(np.std(scr))
        features[f'{prefix}SCL_time_corr'] = float(
            np.corrcoef(scl, np.arange(len(scl)))[0, 1]
            if np.std(scl) > 1e-10 else 0.0
        )

        # 4. SCR peaks: minimum prominence = 0.02 µS, minimum distance = 1 s
        min_dist  = max(1, int(1.0 * fs))
        scr_pos   = np.clip(scr, 0, None)          # only positive excursions
        peaks, props = signal.find_peaks(
            scr_pos,
            prominence=0.02,
            distance=min_dist,
        )

        features[f'{prefix}num_SCR'] = float(len(peaks))

        if len(peaks) > 0:
            amplitudes = scr_pos[peaks]
            features[f'{prefix}sum_Amp_SCR']   = float(np.sum(amplitudes))
            features[f'{prefix}integral_SCR']  = float(np.sum(np.abs(scr)))

            # Recovery time: samples from peak back to half-amplitude (approx)
            recovery_times = []
            for pk in peaks:
                half_amp = scr_pos[pk] / 2.0
                # search forward from peak for signal to drop below half amplitude
                tail = scr_pos[pk:]
                below = np.where(tail <= half_amp)[0]
                recovery_times.append(float(below[0]) / fs if len(below) > 0 else 0.0)
            features[f'{prefix}sum_t_SCR'] = float(np.sum(recovery_times))
        else:
            features[f'{prefix}sum_Amp_SCR']  = 0.0
            features[f'{prefix}sum_t_SCR']    = 0.0
            features[f'{prefix}integral_SCR'] = 0.0

    except Exception as e:
        print(f"[WARN] EDA feature extraction failed ({prefix}): {e}")
        keys = [
            'mean', 'std', 'min', 'max', 'range', 'slope',
            'SCL_mean', 'SCL_std', 'SCR_std', 'SCL_time_corr',
            'num_SCR', 'sum_Amp_SCR', 'sum_t_SCR', 'integral_SCR',
        ]
        features.update({f'{prefix}{k}': 0.0 for k in keys})

    return features


# ---------------------------------------------------------------------------
# RESPIRATION  —  pure scipy peak-finding on bandpass-filtered signal
# ---------------------------------------------------------------------------

def _resp_features(sig, fs, prefix):
    features = {}
    try:
        # Bandpass to typical breathing band 0.1 – 0.5 Hz
        resp_clean = butter_bandpass_filter(sig, lowcut=0.1, highcut=0.5, fs=fs)

        # Peak = end of inhalation, trough = end of exhalation
        min_dist = max(1, int(1.5 * fs))   # minimum 1.5 s between breaths (~40 bpm max)
        peaks,  _ = signal.find_peaks( resp_clean, distance=min_dist)
        troughs,_ = signal.find_peaks(-resp_clean, distance=min_dist)

        window_dur = len(sig) / fs
        breath_rate = float(len(peaks)) / (window_dur / 60.0) if window_dur > 0 else 0.0

        features[f'{prefix}rate']    = breath_rate
        features[f'{prefix}stretch'] = float(np.max(resp_clean) - np.min(resp_clean))
        features[f'{prefix}duration']= float(window_dur)

        # Inhale duration: trough → next peak
        inhale_durs, exhale_durs = [], []
        for pk in peaks:
            prev_troughs = troughs[troughs < pk]
            if len(prev_troughs) > 0:
                inhale_durs.append((pk - prev_troughs[-1]) / fs)
            next_troughs = troughs[troughs > pk]
            if len(next_troughs) > 0:
                exhale_durs.append((next_troughs[0] - pk) / fs)

        features[f'{prefix}inhale_mean'] = float(np.mean(inhale_durs))  if inhale_durs  else 0.0
        features[f'{prefix}inhale_std']  = float(np.std(inhale_durs))   if inhale_durs  else 0.0
        features[f'{prefix}exhale_mean'] = float(np.mean(exhale_durs))  if exhale_durs  else 0.0
        features[f'{prefix}exhale_std']  = float(np.std(exhale_durs))   if exhale_durs  else 0.0

        # Inspiratory volume proxy: integral of positive part of signal
        features[f'{prefix}insp_volume'] = float(np.sum(np.clip(resp_clean, 0, None)))

        # I:E ratio
        total_inhale = sum(inhale_durs)
        total_exhale = sum(exhale_durs)
        features[f'{prefix}I_E_ratio'] = float(total_inhale / (total_exhale + 1e-10))

    except Exception as e:
        print(f"[WARN] RESP feature extraction failed ({prefix}): {e}")
        keys = [
            'rate', 'stretch', 'duration',
            'inhale_mean', 'inhale_std', 'exhale_mean', 'exhale_std',
            'I_E_ratio', 'insp_volume',
        ]
        features.update({f'{prefix}{k}': 0.0 for k in keys})

    return features


# ---------------------------------------------------------------------------
# MAIN PER-MODALITY DISPATCHER
# ---------------------------------------------------------------------------

def extract_single_modality(mod_name, sig, fs):
    features = {}
    prefix   = f"{mod_name}_"

    # ------------------------------------------------------------------ ACC
    if 'ACC' in mod_name:
        sub_window_len = int(5 * fs)
        num_subs = max(1, sig.shape[1] // sub_window_len)
        sub_stats = {axis: {'mean': [], 'std': [], 'int': [], 'peak': []}
                     for axis in ['x', 'y', 'z', '3d']}

        for i in range(num_subs):
            sub_sig = sig[:, i * sub_window_len: (i + 1) * sub_window_len]
            sub_3d  = np.sqrt(np.sum(sub_sig ** 2, axis=0))
            for ax_idx, axis in enumerate(['x', 'y', 'z']):
                sub_stats[axis]['mean'].append(float(np.mean(sub_sig[ax_idx])))
                sub_stats[axis]['std'].append(float(np.std(sub_sig[ax_idx])))
                sub_stats[axis]['int'].append(float(np.sum(np.abs(sub_sig[ax_idx]))))
                sub_stats[axis]['peak'].append(get_peak_frequency(sub_sig[ax_idx], fs))
            sub_stats['3d']['mean'].append(float(np.mean(sub_3d)))
            sub_stats['3d']['std'].append(float(np.std(sub_3d)))
            sub_stats['3d']['int'].append(float(np.sum(np.abs(sub_3d))))

        for axis in ['x', 'y', 'z', '3d']:
            features[f'{prefix}{axis}_mean']     = float(np.mean(sub_stats[axis]['mean']))
            features[f'{prefix}{axis}_std']      = float(np.mean(sub_stats[axis]['std']))
            features[f'{prefix}{axis}_integral'] = float(np.mean(sub_stats[axis]['int']))
            if axis != '3d':
                features[f'{prefix}{axis}_peak_freq'] = float(np.mean(sub_stats[axis]['peak']))

    # ------------------------------------------------------------------ ECG
    elif 'ECG' in mod_name:
        try:
            _find_ecg_peaks._mod_hint = 'ecg'
            peaks = _find_ecg_peaks(sig, fs)
            features.update(_hrv_features(peaks, fs, prefix))
        except Exception as e:
            print(f"[WARN] ECG feature extraction failed ({prefix}): {e}")
            features.update(_hrv_features(np.array([]), fs, prefix))

    # ------------------------------------------------------------------ BVP
    elif 'BVP' in mod_name:
        try:
            _find_ecg_peaks._mod_hint = 'bvp'
            peaks = _find_ecg_peaks(sig, fs)
            features.update(_hrv_features(peaks, fs, prefix))
        except Exception as e:
            print(f"[WARN] BVP feature extraction failed ({prefix}): {e}")
            features.update(_hrv_features(np.array([]), fs, prefix))

    # ------------------------------------------------------------------ EDA
    elif 'EDA' in mod_name:
        features.update(_eda_features(sig, fs, prefix))

    # ------------------------------------------------------------------ EMG
    elif 'EMG' in mod_name:
        emg_chain1 = butter_highpass_filter(sig, cutoff=1.0, fs=fs)
        features[f'{prefix}mean']     = float(np.mean(emg_chain1))
        features[f'{prefix}std']      = float(np.std(emg_chain1))
        features[f'{prefix}range']    = float(np.max(emg_chain1) - np.min(emg_chain1))
        features[f'{prefix}integral'] = float(np.sum(np.abs(emg_chain1)))
        features[f'{prefix}median']   = float(np.median(emg_chain1))
        features[f'{prefix}p10']      = float(np.percentile(emg_chain1, 10))
        features[f'{prefix}p90']      = float(np.percentile(emg_chain1, 90))

        sub_window_len = int(5 * fs)
        num_subs = max(1, len(emg_chain1) // sub_window_len)
        freqs_list, median_freqs_list, peak_freqs_list = [], [], []
        psd_bands = {f'psd_{b * 50}_{(b + 1) * 50}': [] for b in range(7)}

        for i in range(num_subs):
            sub_sig = emg_chain1[i * sub_window_len: (i + 1) * sub_window_len]
            if len(sub_sig) > 0:
                peak_freqs_list.append(get_peak_frequency(sub_sig, fs))
                nperseg = min(len(sub_sig), int(fs * 2))
                f, psd  = signal.welch(sub_sig, fs=fs, nperseg=nperseg)
                if len(psd) > 0:
                    freqs_list.append(float(f[np.argmax(psd)]))
                    cumulative = np.cumsum(psd)
                    median_idx = np.where(cumulative >= cumulative[-1] / 2)[0]
                    if len(median_idx) > 0:
                        median_freqs_list.append(float(f[median_idx[0]]))
                    for b in range(7):
                        lo, hi = b * 50, (b + 1) * 50
                        idx = (f >= lo) & (f < hi)
                        psd_bands[f'psd_{lo}_{hi}'].append(float(np.sum(psd[idx])))

        features[f'{prefix}mean_freq']   = float(np.mean(freqs_list))        if freqs_list        else 0.0
        features[f'{prefix}median_freq'] = float(np.mean(median_freqs_list)) if median_freqs_list else 0.0
        features[f'{prefix}peak_freq']   = float(np.mean(peak_freqs_list))   if peak_freqs_list   else 0.0
        for b_name, vals in psd_bands.items():
            features[f'{prefix}{b_name}'] = float(np.mean(vals)) if vals else 0.0

        emg_chain2 = butter_lowpass_filter(sig, cutoff=50.0, fs=fs)
        peaks, _   = signal.find_peaks(emg_chain2)
        features[f'{prefix}num_peaks'] = float(len(peaks))
        if len(peaks) > 0:
            amps = emg_chain2[peaks]
            features[f'{prefix}mean_amp']     = float(np.mean(amps))
            features[f'{prefix}std_amp']      = float(np.std(amps))
            features[f'{prefix}sum_amp']      = float(np.sum(amps))
            features[f'{prefix}norm_sum_amp'] = float(np.sum(amps) / len(peaks))
        else:
            features[f'{prefix}mean_amp']     = 0.0
            features[f'{prefix}std_amp']      = 0.0
            features[f'{prefix}sum_amp']      = 0.0
            features[f'{prefix}norm_sum_amp'] = 0.0

    # ------------------------------------------------------------------ RESP
    elif 'RESP' in mod_name:
        features.update(_resp_features(sig, fs, prefix))

    # ------------------------------------------------------------------ TEMP
    elif 'TEMP' in mod_name:
        features[f'{prefix}mean']  = float(np.mean(sig))
        features[f'{prefix}std']   = float(np.std(sig))
        features[f'{prefix}min']   = float(np.min(sig))
        features[f'{prefix}max']   = float(np.max(sig))
        features[f'{prefix}range'] = float(np.max(sig) - np.min(sig))
        features[f'{prefix}slope'] = float(np.polyfit(np.arange(len(sig)), sig, 1)[0])

    return features


# ---------------------------------------------------------------------------
# WINDOWED EXTRACTION ENTRY POINT
# ---------------------------------------------------------------------------

def extract_all_windows(data_dict, fs_dict, window_size_sec=60, shift_sec=0.25, n_jobs=-1):
    first_mod    = list(data_dict.keys())[0]
    max_time_sec = data_dict[first_mod].shape[-1] / fs_dict[first_mod]
    start_times  = np.arange(0, max_time_sec - window_size_sec, shift_sec)

    def process_single_window(current_time):
        window_feats = {}
        for mod, sig in data_dict.items():
            fs        = fs_dict[mod]
            start_idx = int(current_time * fs)
            end_idx   = int((current_time + window_size_sec) * fs)
            window_sig = sig[start_idx:end_idx] if sig.ndim == 1 else sig[:, start_idx:end_idx]
            window_feats.update(extract_single_modality(mod, window_sig, fs))
        return window_feats

    feature_matrix = Parallel(n_jobs=n_jobs)(
        delayed(process_single_window)(t)
        for t in tqdm(start_times, desc="  Extracting Windows", leave=False)
    )

    return feature_matrix