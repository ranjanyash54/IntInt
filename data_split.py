import pandas as pd
import os
import numpy as np
from tqdm import tqdm

TRAIN_TEST_SPLIT = 0.9
SCENE_LENGTH = 3000

def split_to_scenes(data, out_dir, cols):
    train_fid_start = int(data.frame_id.min())
    train_fid_total = int(data.frame_id.max() - train_fid_start + 1)
    train_fid_end = int(train_fid_start + round((train_fid_total) * TRAIN_TEST_SPLIT))

    print(train_fid_start, train_fid_end, data.frame_id.max())

    for start_fid in tqdm(range(train_fid_start, train_fid_end - SCENE_LENGTH, SCENE_LENGTH)):
        fragment_df = data[(data.frame_id >= start_fid) & (data.frame_id < start_fid + SCENE_LENGTH)].copy()
        if fragment_df.frame_id.max() - fragment_df.frame_id.min() < 600: break
        # process_ccs(fragment_df, centroid_dict)
        np.savetxt(os.path.join(out_dir, 'train', f'{str(start_fid).zfill(6)}.txt'), fragment_df[cols].values, fmt='%f', delimiter='\t')
    
    
    for start_fid in tqdm(range(train_fid_end, data.frame_id.max(), SCENE_LENGTH)):
        fragment_df = data[(data.frame_id >= start_fid) & (data.frame_id < start_fid + SCENE_LENGTH)].copy()
        if fragment_df.frame_id.max() - fragment_df.frame_id.min() < 600: break
        # process_ccs(fragment_df, centroid_dict)
        np.savetxt(os.path.join(out_dir, 'val', f'{str(start_fid).zfill(6)}.txt'), fragment_df[cols].values, fmt='%f', delimiter='\t')

def split_signals(signals, out_dir, cols):
    train_fid_start = int(signals.frame_id.min())
    train_fid_total = int(signals.frame_id.max() - train_fid_start + 1)
    train_fid_end = int(train_fid_start + round((train_fid_total) * TRAIN_TEST_SPLIT))

    print(train_fid_start, train_fid_end, signals.frame_id.max())

    for start_fid in tqdm(range(train_fid_start, train_fid_end - SCENE_LENGTH, SCENE_LENGTH)):
        fragment_df = signals[(signals.frame_id >= start_fid) & (signals.frame_id < start_fid + SCENE_LENGTH)].copy()
        if fragment_df.frame_id.max() - fragment_df.frame_id.min() < 600: break
        # process_ccs(fragment_df, centroid_dict)
        np.savetxt(os.path.join(out_dir, 'train', f'{str(start_fid).zfill(6)}.txt'), fragment_df[cols].values, fmt='%f', delimiter='\t')
    
    
    for start_fid in tqdm(range(train_fid_end, signals.frame_id.max(), SCENE_LENGTH)):
        fragment_df = signals[(signals.frame_id >= start_fid) & (signals.frame_id < start_fid + SCENE_LENGTH)].copy()
        if fragment_df.frame_id.max() - fragment_df.frame_id.min() < 600: break
        # process_ccs(fragment_df, centroid_dict)
        np.savetxt(os.path.join(out_dir, 'val', f'{str(start_fid).zfill(6)}.txt'), fragment_df[cols].values, fmt='%f', delimiter='\t')


def clean_directory(out_data_dir: str, out_signal_dir: str):
    # Create output directories if they don't exist
    for d in [out_data_dir, out_signal_dir]:
        os.makedirs(os.path.join(d, 'train'), exist_ok=True)
        os.makedirs(os.path.join(d, 'val'), exist_ok=True)

    # Delete all txt files in data_out_dir and its subfolders
    for root, dirs, files in os.walk(out_data_dir):
        for file in files:
            if file.endswith('.txt'):
                os.remove(os.path.join(root, file))
                
    # Delete all txt files in signal_out_dir and its subfolders
    for root, dirs, files in os.walk(out_signal_dir):
        for file in files:
            if file.endswith('.txt'):
                os.remove(os.path.join(root, file))


if __name__ == "__main__":
    input_data_root = './datasets/'
    input_data_folder = 'sumo_12hrs_heading_signal_cluster_region'
    input_data_path = os.path.join(input_data_root, input_data_folder)

    data_cols = ['frame_id', 'track_id', 'pos_x', 'pos_y', 'head', 'class', 'cluster', 'signal', 'direction_id', 'maneuver_id', 'region']
    signal_cols = ['frame_id', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    out_data_dir = './data/raw/'
    out_data_cols = ['frame_id','track_id', 'pos_x', 'pos_y', 'head', 'class', 'cluster', 'signal', 'direction_id', 'maneuver_id', 'region']

    out_signal_dir = './data/signal/'
    out_signal_cols = ['frame_id', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

    if not os.path.exists(input_data_path):
        raise FileNotFoundError(f"Input data path does not exist: {input_data_path}")

    clean_directory(out_data_dir, out_signal_dir)

    ## Loads the data and converts into scenes of 3000 frames
    for subdir, dirs, files in os.walk(input_data_path):
        for file in files:
            full_data_path = os.path.join(subdir, file)
            print('At', full_data_path)
            data = pd.read_csv(full_data_path, sep='\t', index_col=False, header=None)
            if 'signal' in file:
                data.columns = signal_cols
                split_signals(data, out_signal_dir, out_signal_cols)
            elif file.endswith('.txt'):
                data.columns = data_cols
                split_to_scenes(data, out_data_dir, out_data_cols)