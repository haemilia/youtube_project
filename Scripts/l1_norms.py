from pathlib import Path
import useful as use
import pandas as pd
import numpy as np
import cv2
priv = use.get_priv()
global DATA_PATH
DATA_PATH = Path(priv["DATA_PATH"])
hf_token = priv["HF_API_KEY"]
features_dir = DATA_PATH / "dataset"

unpopular_df_raw = pd.read_pickle(DATA_PATH/"temp/raw/unpopular_raw.pkl")

from demo import download_video_with_id
import feature_extraction as fe
import pickle
import sys


###################################### DEFINE FUNCTIONS ###############################################################

# Function to calculate L1-norms between frames sampled at a given percentage
def feature_from_video_percentage(video_path, sample_percentage=0.05):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    l1_norms_sampled = []
    sampled_frames = []

    if total_frames == 0:
        print(f"Warning: Video {video_path} has no frames.")
        cap.release()
        return l1_norms_sampled

    # Calculate the number of frames to sample based on the percentage
    num_samples = int(total_frames * sample_percentage)
    if num_samples == 0 and total_frames > 0:
        num_samples = 1  # Ensure at least one frame is sampled if the video has frames

    # Generate evenly spaced frame indices
    frame_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

    # Read the sampled frames
    for index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if ret:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sampled_frames.append(frame_gray)
        else:
            print(f"Warning: Could not read frame at index {index} from {video_path}")

    cap.release()

    # Calculate L1-norms between consecutive sampled frames
    if len(sampled_frames) > 1:
        for i in range(len(sampled_frames) - 1):
            l1_norm = np.sum(np.abs(sampled_frames[i+1] - sampled_frames[i]))
            l1_norms_sampled.append(l1_norm)

    return l1_norms_sampled

def convert_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Type {type(obj)} not serializable")

##################################Script#########################################################################

# Your DataFrame of video IDs
unpopular_id = unpopular_df_raw["id"]

destination_filename = features_dir/"unpopular_l1_norms.pkl"
if (destination_filename).exists():
    with open(destination_filename,'rb') as fr:
        video_l1_dict = pickle.load(fr)
else:
    video_l1_dict = {}

# Save progress
progress_save_path = DATA_PATH / "temp/unpop_l1_progress.txt"

if progress_save_path.exists():
    with open(progress_save_path, "r") as fr:
        start_from = int(fr.read())
else:
    start_from = 0

error_message = "Completed without catching error"
ids_to_query = unpopular_id[start_from:]
total = len(ids_to_query)

# Define the full path to the log file on the shared drive
shared_drive_path = DATA_PATH 
log_file_path = shared_drive_path / "l1_output.log"

# Ensure the directory exists
shared_drive_path.mkdir(parents=True, exist_ok=True)

with open(log_file_path, 'a') as log_file:
    original_stdout = sys.stdout
    sys.stdout = log_file

    try:
        for i, video_id in enumerate(ids_to_query):
            print(f"{i+1} / {total}")
            # download video in temp
            temp_dir = Path("../temp")
            use.create_directory_if_not_exists(temp_dir)
            vid_path = download_video_with_id(video_id, temp_dir, cookie_path="../cookies.txt")

            if vid_path.exists():
                print(f"Successfully downloaded {video_id}")

                print(f"Processing video: {video_id}...")
                l1_norms = feature_from_video_percentage(vid_path)

                # store in dictionary
                video_l1_dict[video_id] = l1_norms

                # save dictionary
                with open(destination_filename,'wb') as fw:
                    pickle.dump(video_l1_dict, fw)
                # save progress
                with open(progress_save_path, "w") as fw:
                    fw.write(str(start_from +i))

                # delete video from temp
                if vid_path is None:
                    vid_path = temp_dir / f"{video_id}.mp4"
                vid_path.unlink(missing_ok=True)
                print(f"Deleted {video_id}.mp4")
            else:
                print(f"Failed to download {video_id}")
                continue # Skip to the next video if download failed

    except Exception as e:
        error_message = str(e)

    sys.stdout = original_stdout


from datetime import datetime

def write_termination_file_pathlib(reason, shared_drive_path_str):
    shared_drive_path = Path(shared_drive_path_str)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"URGENT_termination_status_{timestamp}.txt"
    full_path = shared_drive_path / filename

    try:
        full_path.write_text(f"Loop terminated at: {timestamp}\nReason: {reason}\n")
        print(f"Termination status written to: {full_path}")
    except Exception as e:
        print(f"Error writing to shared drive: {e}")

write_termination_file_pathlib(error_message, DATA_PATH)