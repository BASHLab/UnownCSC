# import sys
# import os
# import copy
# import pickle

# # ----------------------------------------------------------------------
# # 0. CRITICAL HPC FIX: PREVENT BLAS/OPENMP THREAD CONTENTION
# # ----------------------------------------------------------------------
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

# from joblib import Parallel, delayed
# import torch
# import numpy as np
# import pandas as pd
# from scipy import stats
# from tqdm import tqdm
# from sklearn.base import clone
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import LeaveOneGroupOut
# from sklearn.metrics import f1_score, accuracy_score

# # ----------------------------------------------------------------------
# # 1. PATH ROUTING & IMPORTS
# # ----------------------------------------------------------------------
# current_dir = os.path.dirname(os.path.abspath(__file__))
# src_dir = os.path.abspath(os.path.join(current_dir, '..', 'src'))
# project_root = os.path.abspath(os.path.join(current_dir, '..'))

# if src_dir not in sys.path:
#     sys.path.append(src_dir)

# from data_utils.extract_wesad_features import extract_all_windows
# from models import MultiscaleShapeletCSC
# from utils.dataset_cfg import WESAD

# # ----------------------------------------------------------------------
# # 2. DATA LOADING & EVALUATION FUNCTIONS
# # ----------------------------------------------------------------------
# def load_wesad_subject_multi(subject_id, modalities, dataset_root):
#     file_path = os.path.join(dataset_root, subject_id, f"{subject_id}.pkl")
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"🚨 WESAD raw file not found at {file_path}")
        
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f, encoding='latin1')
        
#     raw_labels = data['label']
#     signals_dict = {}
    
#     for mod in modalities:
#         location, sensor = mod.split('_')
        
#         if location == 'chest' and sensor == 'RESP':
#             sensor = 'Resp'
#         elif location == 'chest' and sensor == 'TEMP':
#             sensor = 'Temp'
            
#         raw_signal = data['signal'][location][sensor]
        
#         if raw_signal.ndim > 1 and raw_signal.shape[1] == 1:
#             raw_signal = raw_signal.flatten()
#         elif raw_signal.ndim > 1 and raw_signal.shape[1] == 3:
#             raw_signal = raw_signal.T 
#         signals_dict[mod] = raw_signal
        
#     return signals_dict, raw_labels

# def _fit_fold(clf_name, clf, X_train, X_test, y_train, y_test):
#     clf.fit(X_train, y_train)
#     preds = clf.predict(X_test)
#     return clf_name, f1_score(y_test, preds, average='macro') * 100, accuracy_score(y_test, preds) * 100

# def train_and_evaluate_loso(X, y, groups):
#     classifiers = {
#         'DT':  DecisionTreeClassifier(criterion='entropy', min_samples_split=20),
#         'RF':  RandomForestClassifier(n_estimators=100, random_state=42),
#         'AB':  AdaBoostClassifier(n_estimators=100, random_state=42),
#         'LDA': LinearDiscriminantAnalysis(),
#         'kNN': KNeighborsClassifier(n_neighbors=9)
#     }
    
#     logo = LeaveOneGroupOut()
#     splits = list(logo.split(X, y, groups))
    
#     jobs = [
#         (name, clone(clf), X[tr], X[te], y[tr], y[te])
#         for tr, te in splits
#         for name, clf in classifiers.items()
#     ]
    
#     raw = Parallel(n_jobs=-1, prefer="threads", verbose=50)(
#         delayed(_fit_fold)(name, clf, Xtr, Xte, ytr, yte)
#         for name, clf, Xtr, Xte, ytr, yte in jobs
#     )
    
#     results = {clf: {'f1': [], 'acc': []} for clf in classifiers}
#     for name, f1, acc in raw:
#         results[name]['f1'].append(f1)
#         results[name]['acc'].append(acc)
        
#     return {
#         name: {
#             'f1_mean':  np.mean(results[name]['f1']),
#             'f1_std':   np.std(results[name]['f1']),
#             'acc_mean': np.mean(results[name]['acc']),
#             'acc_std':  np.std(results[name]['acc'])
#         }
#         for name in classifiers
#     }

# def print_wesad_row(row_name, results_dict):
#     row = f"{row_name:<22}"
#     for clf in ['DT', 'RF', 'AB', 'LDA', 'kNN']:
#         res = results_dict[clf]
#         row += f"| {res['f1_mean']:>5.2f} ± {res['f1_std']:>4.2f}  {res['acc_mean']:>5.2f} ± {res['acc_std']:>4.2f} "
#     print(row)

# # ----------------------------------------------------------------------
# # 3. CACHE GENERATION ORCHESTRATOR
# # ----------------------------------------------------------------------
# def build_or_load_caches(device, cfg, cache_dir, wesad_raw_root):
#     all_subjects = cfg.train_set + cfg.val_set + cfg.eval_set
#     caches = {'gt': None, 'joint': None, 'indep': {}}
    
#     gt_path = os.path.join(cache_dir, "gt_features.csv")
#     joint_path = os.path.join(cache_dir, "joint_features.csv")
    
#     if os.path.exists(gt_path) and os.path.exists(joint_path):
#         print("⚡ Loading Joint and Ground Truth caches...")
#         caches['gt'] = pd.read_csv(gt_path)
#         caches['joint'] = pd.read_csv(joint_path)
#     else:
#         print("🐌 Building Joint and Ground Truth caches. This will take time...")
#         joint_model = MultiscaleShapeletCSC(cfg=cfg, n_atoms=64, atom_len=32, scales=[4, 2, 1], quantizer="fsq").to(device)
#         chkpt_path = os.path.join(project_root, 'saved_chk_dir_dtw', 'stamp_wesad_final.pth')
#         joint_model.load_state_dict(torch.load(chkpt_path, map_location=device))
#         joint_model.eval()
        
#         X_gt, X_joint, y_list, groups_list = [], [], [], []
        
#         with torch.no_grad():
#             for subject_id in all_subjects:
#                 print(f"  -> Joint processing {subject_id}...")
#                 signals_dict, raw_labels = load_wesad_subject_multi(subject_id, cfg.modalities, wesad_raw_root)
                
#                 x_dict_tensor = {}
#                 for mod, sig in signals_dict.items():
#                     t = torch.tensor(sig, dtype=torch.float32).unsqueeze(0).to(device)
#                     if t.ndim == 2: t = t.unsqueeze(0) 
#                     x_dict_tensor[mod] = t
                    
#                 recon_dict_tensor, _, _ = joint_model(x_dict_tensor)
#                 recon_numpy_dict = {mod: tensor.squeeze().cpu().numpy() for mod, tensor in recon_dict_tensor.items()}
#                 for mod, sig in recon_numpy_dict.items():
#                     print(f"  {mod}: shape={sig.shape}, min={sig.min():.4f}, max={sig.max():.4f}, std={sig.std():.6f}")
                
#                 gt_feats = extract_all_windows(signals_dict, cfg.sampling_rates, n_jobs=-1)
#                 joint_feats = extract_all_windows(recon_numpy_dict, cfg.sampling_rates, n_jobs=-1)
                
#                 label_fs, shift_sec, window_size_sec = 700, 0.25, 60
#                 max_time_sec = signals_dict['chest_ACC'].shape[-1] / cfg.sampling_rates['chest_ACC']
                
#                 current_time = 0.0
#                 window_labels = []
#                 valid_window_indices = []
#                 window_idx = 0
                
#                 while current_time + window_size_sec <= max_time_sec:
#                     lbl_start = int(current_time * label_fs)
#                     lbl_end = int((current_time + window_size_sec) * label_fs)
#                     mode_label = stats.mode(raw_labels[lbl_start:lbl_end], keepdims=False)[0]
                    
#                     if mode_label in [1, 2, 3]:
#                         window_labels.append(mode_label - 1) 
#                         valid_window_indices.append(window_idx)
                        
#                     current_time += shift_sec
#                     window_idx += 1

#                 gt_feats_filtered = [gt_feats[i] for i in valid_window_indices if i < len(gt_feats)]
#                 joint_feats_filtered = [joint_feats[i] for i in valid_window_indices if i < len(joint_feats)]
                
#                 min_len = min(len(gt_feats_filtered), len(window_labels))
#                 X_gt.extend(gt_feats_filtered[:min_len])
#                 X_joint.extend(joint_feats_filtered[:min_len])
#                 y_list.extend(window_labels[:min_len])
#                 groups_list.extend([int(subject_id[1:])] * min_len)

#         df_gt = pd.DataFrame(X_gt)
#         df_gt['target_label'] = y_list
#         df_gt['subject_group'] = groups_list
#         df_gt.to_csv(gt_path, index=False)
#         caches['gt'] = df_gt
        
#         df_joint = pd.DataFrame(X_joint)
#         df_joint.to_csv(joint_path, index=False)
#         caches['joint'] = df_joint

#     # --- 2. INDEPENDENT CACHES ---
#     for mod in cfg.modalities:
#         indep_path = os.path.join(cache_dir, f"indep_{mod}_features.csv")
#         if os.path.exists(indep_path):
#             print(f"⚡ Loading Independent cache for {mod}...")
#             caches['indep'][mod] = pd.read_csv(indep_path)
#             continue
            
#         print(f"🐌 Building Independent cache for {mod}...")
#         single_cfg = copy.deepcopy(cfg)
#         single_cfg.modalities = [mod]
#         single_cfg.variates = {mod: cfg.variates[mod]}
#         single_cfg.sampling_rates = {mod: cfg.sampling_rates[mod]}
        
#         indep_model = MultiscaleShapeletCSC(cfg=single_cfg, n_atoms=64, atom_len=32, scales=[4, 2, 1], quantizer="fsq").to(device)
#         chkpt_path = os.path.join(project_root, 'saved_chk_dir_single', mod, f'stamp_{mod}_final.pth')
#         if not os.path.exists(chkpt_path): chkpt_path = chkpt_path.replace('final', 'ep100')
        
#         indep_model.load_state_dict(torch.load(chkpt_path, map_location=device))
#         indep_model.eval()
        
#         X_indep = []
#         with torch.no_grad():
#             for subject_id in all_subjects:
#                 signals_dict, raw_labels = load_wesad_subject_multi(subject_id, [mod], wesad_raw_root)
#                 t = torch.tensor(signals_dict[mod], dtype=torch.float32).unsqueeze(0).to(device)
#                 if t.ndim == 2: t = t.unsqueeze(0)
                
#                 recon_dict_tensor, _, _ = indep_model({mod: t})
#                 recon_numpy_dict = {mod: recon_dict_tensor[mod].squeeze().cpu().numpy()}
#                 for mod, sig in recon_numpy_dict.items():
#                     arr = sig if sig.ndim == 1 else sig.flatten()
#                     print(f"  {mod}: shape={sig.shape}, std={arr.std():.6f}, min={arr.min():.4f}, max={arr.max():.4f}")
                
#                 indep_feats = extract_all_windows(recon_numpy_dict, single_cfg.sampling_rates, n_jobs=-1)
                
#                 label_fs, shift_sec, window_size_sec = 700, 0.25, 60
#                 max_time_sec = signals_dict[mod].shape[-1] / single_cfg.sampling_rates[mod]
                
#                 current_time = 0.0
#                 valid_window_indices = []
#                 window_idx = 0
                
#                 while current_time + window_size_sec <= max_time_sec:
#                     lbl_start = int(current_time * label_fs)
#                     lbl_end = int((current_time + window_size_sec) * label_fs)
#                     mode_label = stats.mode(raw_labels[lbl_start:lbl_end], keepdims=False)[0]
#                     if mode_label in [1, 2, 3]: valid_window_indices.append(window_idx)
#                     current_time += shift_sec
#                     window_idx += 1
                    
#                 indep_feats_filtered = [indep_feats[i] for i in valid_window_indices if i < len(indep_feats)]
#                 X_indep.extend(indep_feats_filtered)
                
#         df_indep = pd.DataFrame(X_indep)
#         df_indep = df_indep.iloc[:len(caches['gt'])] 
#         df_indep.to_csv(indep_path, index=False)
#         caches['indep'][mod] = df_indep

#     return caches

# # ----------------------------------------------------------------------
# # 4. MAIN EXECUTION & TABLE GENERATION
# # ----------------------------------------------------------------------
# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     cache_dir = os.path.join(current_dir, 'features_cache')
#     os.makedirs(cache_dir, exist_ok=True)
#     wesad_raw_root = os.path.join(project_root, 'data', 'WESAD')
    
#     cfg = WESAD()
    
#     caches = build_or_load_caches(device, cfg, cache_dir, wesad_raw_root)
    
#     df_gt = caches['gt']
#     y_arr = df_gt.pop('target_label').values
#     groups_arr = df_gt.pop('subject_group').values
#     df_joint = caches['joint']
#     df_indep_dict = caches['indep']

#     table_groupings = {
#         "Motion": None, 
#         "ACC wrist": ['wrist_ACC'],
#         "ACC chest": ['chest_ACC'],
#         "Wrist": None, 
#         "BVP": ['wrist_BVP'],
#         "EDA (wrist)": ['wrist_EDA'],
#         "TEMP (wrist)": ['wrist_TEMP'],
#         "Wrist physio": ['wrist_BVP', 'wrist_EDA', 'wrist_TEMP'],
#         "Chest": None, 
#         "ECG": ['chest_ECG'],
#         "EDA (chest)": ['chest_EDA'],
#         "EMG": ['chest_EMG'],
#         "RESP": ['chest_RESP'],
#         "TEMP (chest)": ['chest_TEMP'],
#         "Chest physio": ['chest_ECG', 'chest_EDA', 'chest_EMG', 'chest_RESP', 'chest_TEMP'],
#         "Combined": None, 
#         "All wrist": ['wrist_ACC', 'wrist_BVP', 'wrist_EDA', 'wrist_TEMP'],
#         "All chest": ['chest_ACC', 'chest_ECG', 'chest_EDA', 'chest_EMG', 'chest_RESP', 'chest_TEMP'],
#         "All physio": ['wrist_BVP', 'wrist_EDA', 'wrist_TEMP', 'chest_ECG', 'chest_EDA', 'chest_EMG', 'chest_RESP', 'chest_TEMP'],
#         "All modalities": cfg.modalities
#     }

#     print("\n" + "="*145)
#     print(" 📊 WESAD TABLE 3: GROUND TRUTH vs INDEPENDENT STAMP vs JOINT STAMP")
#     print("="*145)
#     print(f"{'Modality / Source':<22}| {'DT (F1 / Acc)':<25}| {'RF (F1 / Acc)':<25}| {'AB (F1 / Acc)':<25}| {'LDA (F1 / Acc)':<25}| {'kNN (F1 / Acc)':<25}")
#     print("-" * 145)

#     for row_name, mod_list in table_groupings.items():
#         if mod_list is None:
#             print(f"\n[{row_name}:]")
#             continue
            
#         print(f"\n🚀 STARTING EVALUATION: {row_name} | Spawning 225 total jobs (GT, Indep, Joint)...")
        
#         cols_to_keep = [col for col in df_gt.columns if any(col.startswith(m) for m in mod_list)]
#         X_gt = df_gt[cols_to_keep].values
#         X_joint = df_joint[cols_to_keep].values
        
#         X_indep_df = pd.concat([df_indep_dict[m] for m in mod_list], axis=1)
#         X_indep = X_indep_df.values
        
#         if X_gt.shape[1] == 0 or X_joint.shape[1] == 0 or X_indep.shape[1] == 0:
#             print(f"  ⚠️ SKIPPING [{row_name}]: 0 features found. (Check extraction pipeline)")
#             continue

#         # 1. Filter out NaN edge cases from feature extraction
#         valid_idx = ~np.isnan(X_gt).any(axis=1) & ~np.isnan(X_joint).any(axis=1) & ~np.isnan(X_indep).any(axis=1)

#         X_g_v = X_gt[valid_idx]
#         X_j_v = X_joint[valid_idx]
#         X_i_v = X_indep[valid_idx]

#         # # ---------------------------------------------------------
#         # # 2. THE DEGENERATE MATRIX PATCH: Catch flatlines of ANY value
#         # # ---------------------------------------------------------
#         # def is_degenerate(mat):
#         #     if mat.shape[0] < 2: return True
#         #     # If the max value equals the min value for every column, all rows are identical
#         #     return np.all(np.max(mat, axis=0) == np.min(mat, axis=0))

#         # if is_degenerate(X_g_v) or is_degenerate(X_j_v) or is_degenerate(X_i_v):
#         #     print(f"  ⚠️ SKIPPING [{row_name}]: Matrix is degenerate (all rows identical due to model flatlining).")
#         #     continue
#         # # ---------------------------------------------------------
#         # ---------------------------------------------------------
#         # 2. THE VARIANCE THRESHOLD PATCH: Strip flatlined columns
#         # ---------------------------------------------------------
#         def drop_zero_var(mat):
#             # Calculate variance of each column and keep only those > 0
#             variances = np.var(mat, axis=0)
#             return mat[:, variances > 1e-6]

#         X_g_clean = drop_zero_var(X_g_v)
#         X_j_clean = drop_zero_var(X_j_v)
#         X_i_clean = drop_zero_var(X_i_v)

#         if X_g_clean.shape[1] == 0 or X_j_clean.shape[1] == 0 or X_i_clean.shape[1] == 0:
#             print(f"  ⚠️ SKIPPING [{row_name}]: Matrix collapsed (all features have zero variance).")
#             continue
#         # ---------------------------------------------------------
#         # 3. Run Classifiers safely
#         res_gt = train_and_evaluate_loso(X_g_v, y_arr[valid_idx], groups_arr[valid_idx])
#         res_indep = train_and_evaluate_loso(X_i_v, y_arr[valid_idx], groups_arr[valid_idx])
#         res_joint = train_and_evaluate_loso(X_j_v, y_arr[valid_idx], groups_arr[valid_idx])

#         print_wesad_row(f"  GT ({row_name})", res_gt)
#         print_wesad_row(f"  Indep STAMP", res_indep)
#         print_wesad_row(f"  Joint STAMP", res_joint)
#         print("-" * 145)
import sys
import os
import copy
import pickle

# 0. CRITICAL HPC FIX: PREVENT BLAS/OPENMP THREAD CONTENTION
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# 1. PATH ROUTING & IMPORTS
current_dir  = os.path.dirname(os.path.abspath(__file__))
src_dir      = os.path.abspath(os.path.join(current_dir, '..', 'src'))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if src_dir not in sys.path:
    sys.path.append(src_dir)

from data_utils.extract_wesad_features import extract_all_windows
from models import MultiscaleShapeletCSC
from utils.dataset_cfg import WESAD


# 2. DATA LOADING
def load_wesad_subject_multi(subject_id, modalities, dataset_root):
    file_path = os.path.join(dataset_root, subject_id, f"{subject_id}.pkl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"WESAD raw file not found at {file_path}")

    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    raw_labels   = data['label']
    signals_dict = {}

    for mod in modalities:
        location, sensor = mod.split('_', 1)
        if location == 'chest' and sensor == 'RESP': sensor = 'Resp'
        elif location == 'chest' and sensor == 'TEMP': sensor = 'Temp'

        raw_signal = data['signal'][location][sensor]
        if raw_signal.ndim > 1 and raw_signal.shape[1] == 1:
            raw_signal = raw_signal.flatten()
        elif raw_signal.ndim > 1 and raw_signal.shape[1] == 3:
            raw_signal = raw_signal.T
        signals_dict[mod] = raw_signal

    return signals_dict, raw_labels


# 3. CLASSIFIER TRAINING & EVALUATION
# Stochastic classifiers (DT, RF, AB) are run n_runs times with different seeds
# and reported as mean ± std, matching the WESAD paper format.
# Deterministic classifiers (LDA, kNN) are run once with no std.

def train_and_evaluate_fixed(X_train, y_train, X_test, y_test, n_runs=10):
    deterministic = {
        'LDA': make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        'kNN': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=9)),
    }

    stochastic_templates = {
        'DT': lambda seed: make_pipeline(
            StandardScaler(),
            DecisionTreeClassifier(criterion='entropy', min_samples_split=20, random_state=seed)
        ),
        'RF': lambda seed: make_pipeline(
            StandardScaler(),
            RandomForestClassifier(n_estimators=100, random_state=seed)
        ),
        'AB': lambda seed: make_pipeline(
            StandardScaler(),
            AdaBoostClassifier(n_estimators=100, random_state=seed)
        ),
    }

    results = {}

    # Single-run deterministic classifiers
    for name, clf in deterministic.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        results[name] = {
            'f1':      f1_score(y_test, preds, average='macro') * 100,
            'acc':     accuracy_score(y_test, preds) * 100,
            'f1_std':  None,
            'acc_std': None,
        }

    # Multi-run stochastic classifiers
    for name, clf_fn in stochastic_templates.items():
        f1s, accs = [], []
        for seed in range(n_runs):
            clf = clf_fn(seed)
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            f1s.append(f1_score(y_test, preds, average='macro') * 100)
            accs.append(accuracy_score(y_test, preds) * 100)
        results[name] = {
            'f1':      np.mean(f1s),
            'acc':     np.mean(accs),
            'f1_std':  np.std(f1s),
            'acc_std': np.std(accs),
        }

    return results


def print_wesad_row(row_name, results_dict):
    row = f"{row_name:<22}"
    for clf in ['DT', 'RF', 'AB', 'LDA', 'kNN']:
        res = results_dict[clf]
        if res['f1_std'] is not None:
            row += f"| {res['f1']:>5.2f} ± {res['f1_std']:>4.2f}  {res['acc']:>5.2f} ± {res['acc_std']:>4.2f} "
        else:
            row += f"| {res['f1']:>5.2f}         {res['acc']:>5.2f}         "
    print(row)


# 4. CACHE GENERATION ORCHESTRATOR
def build_or_load_caches(device, cfg, cache_dir, wesad_raw_root):
    all_subjects = cfg.train_set + cfg.val_set + cfg.eval_set
    caches = {'gt': None, 'joint': None, 'indep': {}}

    gt_path    = os.path.join(cache_dir, "gt_features.csv")
    joint_path = os.path.join(cache_dir, "joint_features.csv")

    if os.path.exists(gt_path) and os.path.exists(joint_path):
        print("Loading Joint and Ground Truth caches...")
        caches['gt']    = pd.read_csv(gt_path)
        caches['joint'] = pd.read_csv(joint_path)
    else:
        print("Building Joint and Ground Truth caches. This will take time...")
        joint_model = MultiscaleShapeletCSC(
            cfg=cfg, n_atoms=64, atom_len=32, scales=[4, 2, 1], quantizer="fsq"
        ).to(device)
        chkpt_path = os.path.join(project_root, 'saved_chk_dir_dtw', 'stamp_wesad_final.pth')
        joint_model.load_state_dict(torch.load(chkpt_path, map_location=device))
        joint_model.eval()

        X_gt, X_joint, y_list, groups_list = [], [], [], []

        with torch.no_grad():
            for subject_id in all_subjects:
                print(f"  -> Joint processing {subject_id}...")
                signals_dict, raw_labels = load_wesad_subject_multi(
                    subject_id, cfg.modalities, wesad_raw_root
                )

                x_dict_tensor = {}
                for mod, sig in signals_dict.items():
                    t = torch.tensor(sig, dtype=torch.float32).unsqueeze(0).to(device)
                    if t.ndim == 2: t = t.unsqueeze(0)
                    x_dict_tensor[mod] = t

                recon_dict_tensor, _, _ = joint_model(x_dict_tensor)
                recon_numpy_dict = {
                    mod: tensor.squeeze().cpu().numpy()
                    for mod, tensor in recon_dict_tensor.items()
                }

                gt_feats    = extract_all_windows(signals_dict,     cfg.sampling_rates, n_jobs=-1)
                joint_feats = extract_all_windows(recon_numpy_dict, cfg.sampling_rates, n_jobs=-1)

                label_fs, shift_sec, window_size_sec = 700, 0.25, 60
                ref_mod      = cfg.modalities[0]
                max_time_sec = signals_dict[ref_mod].shape[-1] / cfg.sampling_rates[ref_mod]

                current_time         = 0.0
                window_labels        = []
                valid_window_indices = []
                window_idx           = 0

                while current_time + window_size_sec <= max_time_sec:
                    lbl_start  = int(current_time * label_fs)
                    lbl_end    = int((current_time + window_size_sec) * label_fs)
                    mode_label = stats.mode(raw_labels[lbl_start:lbl_end], keepdims=False)[0]

                    if mode_label in [1, 2, 3]:
                        window_labels.append(mode_label - 1)
                        valid_window_indices.append(window_idx)

                    current_time += shift_sec
                    window_idx   += 1

                gt_feats_filtered    = [gt_feats[i]    for i in valid_window_indices if i < len(gt_feats)]
                joint_feats_filtered = [joint_feats[i] for i in valid_window_indices if i < len(joint_feats)]

                min_len = min(len(gt_feats_filtered), len(joint_feats_filtered), len(window_labels))
                X_gt.extend(gt_feats_filtered[:min_len])
                X_joint.extend(joint_feats_filtered[:min_len])
                y_list.extend(window_labels[:min_len])
                groups_list.extend([int(subject_id[1:])] * min_len)

        df_gt = pd.DataFrame(X_gt)
        df_gt['target_label']  = y_list
        df_gt['subject_group'] = groups_list
        df_gt.to_csv(gt_path, index=False)
        caches['gt'] = df_gt

        df_joint = pd.DataFrame(X_joint)
        df_joint['target_label']  = y_list
        df_joint['subject_group'] = groups_list
        df_joint.to_csv(joint_path, index=False)
        caches['joint'] = df_joint

    # --- INDEPENDENT CACHES ---
    for mod in cfg.modalities:
        indep_path = os.path.join(cache_dir, f"indep_{mod}_features.csv")

        if os.path.exists(indep_path):
            print(f"Loading Independent cache for {mod}...")
            caches['indep'][mod] = pd.read_csv(indep_path)
            continue

        print(f"Building Independent cache for {mod}...")
        single_cfg                = copy.deepcopy(cfg)
        single_cfg.modalities     = [mod]
        single_cfg.variates       = {mod: cfg.variates[mod]}
        single_cfg.sampling_rates = {mod: cfg.sampling_rates[mod]}

        indep_model = MultiscaleShapeletCSC(
            cfg=single_cfg, n_atoms=64, atom_len=32, scales=[4, 2, 1], quantizer="fsq"
        ).to(device)
        chkpt_path = os.path.join(
            project_root, 'saved_chk_dir_single', mod, f'stamp_{mod}_final.pth'
        )
        if not os.path.exists(chkpt_path):
            chkpt_path = chkpt_path.replace('final', 'ep100')

        indep_model.load_state_dict(torch.load(chkpt_path, map_location=device))
        indep_model.eval()

        X_indep      = []
        y_indep      = []
        groups_indep = []

        with torch.no_grad():
            for subject_id in all_subjects:
                signals_dict, raw_labels = load_wesad_subject_multi(
                    subject_id, [mod], wesad_raw_root
                )
                t = torch.tensor(signals_dict[mod], dtype=torch.float32).unsqueeze(0).to(device)
                if t.ndim == 2: t = t.unsqueeze(0)

                recon_dict_tensor, _, _ = indep_model({mod: t})
                recon_numpy_dict = {mod: recon_dict_tensor[mod].squeeze().cpu().numpy()}

                indep_feats = extract_all_windows(
                    recon_numpy_dict, single_cfg.sampling_rates, n_jobs=-1
                )

                label_fs, shift_sec, window_size_sec = 700, 0.25, 60
                max_time_sec = signals_dict[mod].shape[-1] / single_cfg.sampling_rates[mod]

                current_time         = 0.0
                valid_window_indices = []
                window_labels        = []
                window_idx           = 0

                while current_time + window_size_sec <= max_time_sec:
                    lbl_start  = int(current_time * label_fs)
                    lbl_end    = int((current_time + window_size_sec) * label_fs)
                    mode_label = stats.mode(raw_labels[lbl_start:lbl_end], keepdims=False)[0]
                    if mode_label in [1, 2, 3]:
                        valid_window_indices.append(window_idx)
                        window_labels.append(mode_label - 1)
                    current_time += shift_sec
                    window_idx   += 1

                indep_feats_filtered = [
                    indep_feats[i] for i in valid_window_indices if i < len(indep_feats)
                ]

                min_len = min(len(indep_feats_filtered), len(window_labels))
                X_indep.extend(indep_feats_filtered[:min_len])
                y_indep.extend(window_labels[:min_len])
                groups_indep.extend([int(subject_id[1:])] * min_len)

        df_indep = pd.DataFrame(X_indep)
        df_indep['target_label']  = y_indep
        df_indep['subject_group'] = groups_indep
        df_indep.to_csv(indep_path, index=False)
        caches['indep'][mod] = df_indep

    return caches


# 5. MAIN EXECUTION & TABLE GENERATION
if __name__ == "__main__":
    device         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_dir      = os.path.join(current_dir, 'features_cache')
    os.makedirs(cache_dir, exist_ok=True)
    wesad_raw_root = os.path.join(project_root, 'data', 'WESAD')

    cfg = WESAD()

    # Integer subject IDs for train/test masking
    train_subjects = set(int(s[1:]) for s in cfg.train_set)
    test_subjects  = set(int(s[1:]) for s in cfg.eval_set)
    # val_subjects = set(int(s[1:]) for s in cfg.val_set)  # available for hyperparam tuning

    caches = build_or_load_caches(device, cfg, cache_dir, wesad_raw_root)

    df_gt      = caches['gt']
    y_arr      = df_gt.pop('target_label').values
    groups_arr = df_gt.pop('subject_group').values

    df_joint      = caches['joint'].drop(columns=['target_label', 'subject_group'], errors='ignore')
    df_indep_dict = caches['indep']

    table_groupings = {
        "Motion":       None,
        "ACC wrist":    ['wrist_ACC'],
        "ACC chest":    ['chest_ACC'],
        "Wrist":        None,
        "BVP":          ['wrist_BVP'],
        "EDA (wrist)":  ['wrist_EDA'],
        "TEMP (wrist)": ['wrist_TEMP'],
        "Wrist physio": ['wrist_BVP', 'wrist_EDA', 'wrist_TEMP'],
        "Chest":        None,
        "ECG":          ['chest_ECG'],
        "EDA (chest)":  ['chest_EDA'],
        "EMG":          ['chest_EMG'],
        "RESP":         ['chest_RESP'],
        "TEMP (chest)": ['chest_TEMP'],
        "Chest physio": ['chest_ECG', 'chest_EDA', 'chest_EMG', 'chest_RESP', 'chest_TEMP'],
        "Combined":     None,
        "All wrist":    ['wrist_ACC', 'wrist_BVP', 'wrist_EDA', 'wrist_TEMP'],
        "All chest":    ['chest_ACC', 'chest_ECG', 'chest_EDA', 'chest_EMG', 'chest_RESP', 'chest_TEMP'],
        "All physio":   ['wrist_BVP', 'wrist_EDA', 'wrist_TEMP', 'chest_ECG', 'chest_EDA',
                         'chest_EMG', 'chest_RESP', 'chest_TEMP'],
        "All modalities": cfg.modalities,
    }

    # Column header — DT/RF/AB show F1 ± std / Acc ± std, LDA/kNN show F1 / Acc only
    col_w = 24
    print("\n" + "=" * 145)
    print(" WESAD TABLE 3: GROUND TRUTH vs INDEPENDENT STAMP vs JOINT STAMP")
    print(f" Train: {cfg.train_set}")
    print(f" Test:  {cfg.eval_set}")
    print("=" * 145)
    print(
        f"{'Modality / Source':<22}"
        f"| {'DT (F1 ± std / Acc ± std)':<{col_w}}"
        f"| {'RF (F1 ± std / Acc ± std)':<{col_w}}"
        f"| {'AB (F1 ± std / Acc ± std)':<{col_w}}"
        f"| {'LDA (F1 / Acc)':<{col_w}}"
        f"| {'kNN (F1 / Acc)':<{col_w}}"
    )
    print("-" * 145)

    for row_name, mod_list in table_groupings.items():
        if mod_list is None:
            print(f"\n[{row_name}:]")
            continue

        print(f"\n  Evaluating: {row_name}...")

        # --- Build feature matrices ---
        cols_to_keep = [col for col in df_gt.columns if any(col.startswith(m) for m in mod_list)]
        X_gt    = df_gt[cols_to_keep].values
        X_joint = df_joint[cols_to_keep].values

        indep_frames = [
            df_indep_dict[m].drop(columns=['target_label', 'subject_group'], errors='ignore')
            for m in mod_list
        ]
        lengths = [len(f) for f in indep_frames]
        if len(set(lengths)) > 1:
            print(
                f"  WARNING [{row_name}]: Indep modality frames have mismatched lengths: "
                f"{dict(zip(mod_list, lengths))}. Truncating to shortest."
            )
            min_rows     = min(lengths)
            indep_frames = [f.iloc[:min_rows] for f in indep_frames]

        X_indep = pd.concat(indep_frames, axis=1).values

        if X_gt.shape[1] == 0 or X_joint.shape[1] == 0 or X_indep.shape[1] == 0:
            print(f"  SKIPPING [{row_name}]: 0 features found.")
            continue

        # --- Neutralise inf, then filter bad rows ---
        X_gt    = np.where(np.isinf(X_gt),    np.nan, X_gt)
        X_joint = np.where(np.isinf(X_joint), np.nan, X_joint)
        X_indep = np.where(np.isinf(X_indep), np.nan, X_indep)

        def is_valid_row(mat):
            return ~np.isnan(mat).any(axis=1) & ~(mat == 0).all(axis=1)

        valid_idx = is_valid_row(X_gt) & is_valid_row(X_joint) & is_valid_row(X_indep)

        if valid_idx.sum() == 0:
            print(
                f"  SKIPPING [{row_name}]: no valid rows after NaN/zero filter.\n"
                f"    GT NaN rows:    {np.isnan(X_gt).any(axis=1).sum()} / {len(X_gt)}\n"
                f"    Joint NaN rows: {np.isnan(X_joint).any(axis=1).sum()} / {len(X_joint)}\n"
                f"    Indep NaN rows: {np.isnan(X_indep).any(axis=1).sum()} / {len(X_indep)}"
            )
            continue

        X_g_v = X_gt[valid_idx]
        X_j_v = X_joint[valid_idx]
        X_i_v = X_indep[valid_idx]
        y_v   = y_arr[valid_idx]
        g_v   = groups_arr[valid_idx]

        # --- Drop zero-variance columns ---
        def drop_zero_var(mat):
            if mat.shape[0] == 0: return mat
            return mat[:, np.var(mat, axis=0) > 1e-6]

        X_g_clean = drop_zero_var(X_g_v)
        X_j_clean = drop_zero_var(X_j_v)
        X_i_clean = drop_zero_var(X_i_v)

        if X_g_clean.shape[1] == 0 or X_j_clean.shape[1] == 0 or X_i_clean.shape[1] == 0:
            print(f"  SKIPPING [{row_name}]: all features have zero variance.")
            continue

        # --- Fixed train / test split ---
        train_mask = np.isin(g_v, list(train_subjects))
        test_mask  = np.isin(g_v, list(test_subjects))

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            print(
                f"  SKIPPING [{row_name}]: empty train ({train_mask.sum()}) "
                f"or test ({test_mask.sum()}) split after filtering."
            )
            continue

        res_gt = train_and_evaluate_fixed(
            X_g_clean[train_mask], y_v[train_mask],
            X_g_clean[test_mask],  y_v[test_mask],
        )
        res_indep = train_and_evaluate_fixed(
            X_i_clean[train_mask], y_v[train_mask],
            X_i_clean[test_mask],  y_v[test_mask],
        )
        res_joint = train_and_evaluate_fixed(
            X_j_clean[train_mask], y_v[train_mask],
            X_j_clean[test_mask],  y_v[test_mask],
        )

        print_wesad_row(f"  GT ({row_name})", res_gt)
        print_wesad_row(f"  Indep STAMP",     res_indep)
        print_wesad_row(f"  Joint STAMP",     res_joint)
        print("-" * 145)