#####################################################################################
# main_process.py
# - All `main()` processes of each module/process come here
# - The functions in each module get called from here
######################################################################################
#%%
import useful as use
from pathlib import Path
import pandas as pd
import numpy as np
from googleapiclient.discovery import build
from data_collection import (get_video_categories, 
                             make_vid_cat_dict, 
                             get_categorised_popular_vids,
                             get_general_popular_vids,
                             get_channel_infos,)

# tested
def data_collection_daily(all_categories=False):
    """
    Daily data collection process
    # Args
    - all_categories : (bool) = `False`
        - Whether to 
    """
    # Get important information
    priv = use.get_priv()
    # Announce global constants
    global API_KEY, DATA_PATH, TODAY_DATE
    API_KEY = str(priv["API_KEY"])
    DATA_PATH = Path(priv["DATA_PATH"])
    TODAY_DATE = use.get_today()

    # Set up Youtube Data API access
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    global youtube
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

    # Video Categories Information
    # If the information doesn't exist already, request it
    if not (DATA_PATH / "youtube_korea_video_categories.json").exists():
        # Create directory to keep video categories if it doesn't exist already
        use.create_directory_if_not_exists(DATA_PATH / "categories")

        # Get video categories list of Youtube Korea
        get_video_categories_success = get_video_categories(youtube, 
                                        data_path= DATA_PATH / "youtube_korea_video_categories.json")
        # For consistency
        if not get_video_categories_success:
            raise Exception("Error while getting video categories")
    

    
    # Get video categories information as a dictionary of desired structure
    vid_cat_df, vid_cat_id_to_name_dict = make_vid_cat_dict(DATA_PATH / "youtube_korea_video_categories.json")

    ################################################################################################################
    # Get trending/popular video information for the day
    ################################################################################################################
    files_created_today = {
    'popular_vids':{},
    'channels':{},
    }
    popular_vids_path = DATA_PATH / "popular_vids"
    use.create_directory_if_not_exists(popular_vids_path)
    if all_categories:
        # Collect video information about all categories
        get_categorised_vids_success, all_category_vid_dfs, files_created_today = get_categorised_popular_vids(youtube,
                                                                                                               files_created_today=files_created_today,
                                                                                                               vid_cat_df=vid_cat_df,
                                                                                                               popular_vids_path=popular_vids_path,
                                                                                                               collection_date=TODAY_DATE)
        get_general_vids_success, general_category_vid_df, files_created_today = get_general_popular_vids(youtube,
                                                                                                         files_created_today=files_created_today,
                                                                                                         popular_vids_path=popular_vids_path,
                                                                                                         collection_date=TODAY_DATE)
        # if both requests were successful, then join the results
        if get_categorised_vids_success and get_general_vids_success:
            all_category_vid_dfs["0"] = general_category_vid_df
    else:
        # Collect video information only about general category
        get_general_vids_success, general_category_vid_df, files_created_today = get_general_popular_vids(youtube,
                                                                                                         files_created_today=files_created_today,
                                                                                                         popular_vids_path=popular_vids_path,
                                                                                                         collection_date=TODAY_DATE)
        # In this case, this is the only result
        all_category_vid_dfs = {"0": general_category_vid_df}

    ####################################################################################################################
    # Channel Info Collection
    ####################################################################################################################
    channel_dir = DATA_PATH / "channels"
    use.create_directory_if_not_exists(channel_dir)
    get_channel_info_success, files_created_today= get_channel_infos(youtube,
                                                                    files_created_today=files_created_today,
                                                                    channel_path=channel_dir,
                                                                    collection_date=TODAY_DATE)
    if get_channel_info_success:
        use.report_progress("DATA COLLECTION", f"Complete for {TODAY_DATE}")
    return files_created_today


def batch_process_tabular_data():
    #TODO
    # No requests to API
    # Simply concatenates collected data into one big dataset
    # Performs data cleansing
    # Should be called weekly
    # Fix current process to include video id in the resulting dataset
    pass

def collect_thumbnails():
    #TODO
    # Should be called after batch processing.
    # Request Youtube Data API to get thumbnail images
    pass

def prepare_dataset_for_training():
    #TODO
    # Prepare final dataset into optimal state for training
    # Should have options to exclude certain variables upon request, easily.
    pass

def construct_models_for_comparison():
    #TODO
    # Input: model defining dictionary
    # To be called when developing model
    # Constructs many candidate models
    # Not to be used in service deployment
    # Purpose is for comparing
    pass

def construct_final_model():
    #TODO
    # Constructs final architecture of model
    # Constructs one model
    pass

def load_pretrained_model():
    #TODO
    # Load pre-trained models
    # To be used instead of construction
    pass

if __name__ == "__main__":
    pass
