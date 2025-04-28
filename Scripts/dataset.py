###############################################################
# Outputs?
# - Tabular Data with VLC
# - Tabular Data without VLC
# - Thumbnail Images
# - Thumbnail Image Features
# - (?) Thumbnail Text Features?
# - Title Text Features
# - Video Download and Dynamics Processing
#################################################################
#%%
from pathlib import Path
import pandas as pd
from typing import Tuple
import useful as use
from collections import defaultdict

priv = use.get_priv()
global DATA_PATH
DATA_PATH = Path(priv["DATA_PATH"])
#%%
def concatenate_dfs(multi_df_dir:Path|str, back_up_dir:Path|str) -> Tuple[dict, Path]:
    """
    Read all DataFrame pickle files from `multi_df_dir`, concatenate all into one DataFrame.
    Returns concatenated DataFrame and path to backed-up pickle file
    """
    # Read dataframes from multi_df_dir
    to_concat = defaultdict(list)
    for collection_date, inner_dict in use.get_df_paths(multi_df_dir).items():
        for category_id, df_path in inner_dict.items():
            df = pd.read_pickle(df_path)
            to_concat[category_id].append(df)
    # concatenate
    concatenated = {}
    for category_id, df_list in to_concat.items():
        concatenated[category_id] = pd.concat(df_list)
    # Back up concat df
    backed_up_paths = {}
    use.create_directory_if_not_exists(back_up_dir)
    for category_id, concat_df in concatenated.items():
        concat_df.to_pickle(back_up_dir / f"{category_id}.pkl")
        use.report_progress("SAVE", back_up_dir / f"{category_id}.pkl")
        backed_up_paths[category_id] = back_up_dir / f"{category_id}.pkl"
    # Return concat df and backup dir
    return concatenated, backed_up_paths

def concatenate_popular(popular_dir, category_id, temp_dir):
    concatenated, concat_paths = concatenate_dfs(popular_dir, temp_dir/"popular")
    df_interest = concatenated.get(str(category_id))
    if df_interest is None:
        raise Exception("File does not exist")
    df_dup_removed = df_interest.drop_duplicates("id")
    use.create_directory_if_not_exists(temp_dir/"raw")
    df_dup_removed.to_pickle(temp_dir/"raw/popular_raw.pkl")
    use.report_progress("SAVE", temp_dir/"raw/popular_raw.pkl")
    return df_dup_removed

def concatenate_channels(channels_dir, category_id, temp_dir):
    concatenated, concat_paths = concatenate_dfs(channels_dir, temp_dir/"channels")
    df_interest = concatenated.get(str(category_id))
    if df_interest is None:
        raise Exception("File does not exist")
    df_dup_removed = df_interest.drop_duplicates("channel_id")
    use.create_directory_if_not_exists(temp_dir/"raw")
    df_dup_removed.to_pickle(temp_dir/"raw/channels_raw.pkl")
    use.report_progress("SAVE", temp_dir/"raw/channels_raw.pkl")
    return df_dup_removed

#TODO
def calculate_potential_range_by_channel(popular_df_raw):
    # Convert collection date to pandas Timestamp
    popular_df_raw["collection_date"] = popular_df_raw["collection_date"].apply(lambda cell: pd.Timestamp(cell, tz="Asia/Seoul"))
    # Calculate potential range by collection date
    by_collection_date = popular_df_raw[["published_at", "collection_date"]].groupby("collection_date").agg(lambda things: list(things))
    potential_range_by_collection_date = {}
    for i in range((by_collection_date.shape[0])):
        cur_collection_date = by_collection_date.iloc[i].name
        published_range = by_collection_date.iloc[i, 0]
        min_published = min(published_range)
        max_published = max(published_range)
        potential_range_by_collection_date[cur_collection_date] = [min_published, max_published]

    # Calculate potential range by channel, also maintain popular video ids
    by_channel_id = popular_df_raw[["channel_id", "id", "collection_date"]].groupby("channel_id").agg(lambda things: list(things))
    potential_range_by_channel_id = {}
    for i in range((by_channel_id.shape[0])):
        cur_channel_id = by_channel_id.iloc[i].name
        vid_id_list = by_channel_id.iloc[i]["id"]
        col_date_list = by_channel_id.iloc[i]["collection_date"]
        min_list, max_list = [], []
        for col_date in col_date_list:
            min_col, max_col = potential_range_by_collection_date.get(col_date)
            min_list.append(min_col)
            max_list.append(max_col)
        range_by_channel = [min(min_list), max(max_list)]
        potential_range_by_channel_id[cur_channel_id] = {
            "vid_id": vid_id_list,
            "potential_range": range_by_channel
        }
    # {channel_id: {"vid_id": [popular video ids], "potential_range": [minimum, maximum]}}
    return potential_range_by_channel_id
#TODO
def get_uploads_items_by_channel(youtube, channels_df, temp_path):
    progress_dict_path = temp_path / "progress.json"
    vid_items_dict_path = temp_path / "uploads_result.json"
    if not progress_dict_path.exists():
        # if progress file doesn't exist, initialise
        progress_dict = {
            "upload_pl_index": 0,
            "channel_id": None,
            "middle_of_playlist_request": False,
            "params": {
            "part": "contentDetails,status",
            "maxResults": 50,
            "playlistId": "",
            },
        }
        use.save_dict_to_json(progress_dict, progress_dict_path)
        use.report_progress("SAVE", progress_dict_path)
    else:
        # If progress dict path is given, retrieve
        progress_dict = use.open_dict_from_json(progress_dict_path)
        use.report_progress("RETRIEVE", progress_dict_path)

    def request_upload_pl_items(params:dict):
        request = youtube.playlistItems().list(
            part = params.get("part", None),
            maxResults = params.get("maxResults", None),
            pageToken = params.get("pageToken", None),
            playlistId = params.get("playlistId", None)
        )
        response = request.execute()
        return response     
        
    def request_wo_token(progress_dict, vid_items_dict):
        ## Assert that the flag for middle of playlist request is set to False
        assert progress_dict["middle_of_playlist_request"] is False
        ## Assert that pageToken value is empty
        assert progress_dict["params"].get("pageToken") is None
        ## Assert that this is the first request for the channel_id
        ## == Assert that vid_items_dict entry for the channel_id is empty
        assert bool(vid_items_dict.get(progress_dict["channel_id"])) is False
        try:
            response = request_upload_pl_items(progress_dict["params"])
        except Exception as err:
            # The error could mean that I've reached my quota
            # save progress_dict
            use.save_dict_to_json(progress_dict, progress_dict_path)
            # save vid_items_dict
            use.save_dict_to_json(vid_items_dict, vid_items_dict_path)
            # But raise the error so we can see what went wrong
            raise err
        # If we got a valid response
        vid_items_dict[progress_dict["channel_id"]] = response['items']
        page_token = response.get("nextPageToken")

        if page_token:
            progress_dict["middle_of_playlist_request"] = True
            progress_dict["params"]["pageToken"] = page_token

        return progress_dict, vid_items_dict

    def request_with_token(progress_dict, vid_items_dict):
        ## Assert that the flag for middle of playlist request is set to True
        assert progress_dict["middle_of_playlist_request"] is True
        ## Assert that we have a valid pageToken value  
        assert progress_dict["params"].get("pageToken") is not None
        page_token = progress_dict["params"].get("pageToken")
        while_counter = 0
        # this amount of limit seems reasonable
        while page_token and while_counter < 1000:
            while_counter += 1

            try:
                # send API request
                response = request_upload_pl_items(progress_dict["params"])
            except Exception as err:
                # The error could mean that I've reached my quota
                # save progress_dict
                use.save_dict_to_json(progress_dict, progress_dict_path)
                # save vid_items_dict
                use.save_dict_to_json(vid_items_dict, vid_items_dict_path)
                use.report_progress("UPDATE", vid_items_dict_path)
                # But raise the error so we can see what went wrong
                raise err
            
            vid_items_dict[progress_dict["channel_id"]].extend(response["items"])
            page_token = response.get("nextPageToken")
            if page_token:
                progress_dict["params"]["pageToken"] = page_token
                # save progress_dict
                use.save_dict_to_json(progress_dict, progress_dict_path)
                use.report_progress("UPDATE", progress_dict_path)
                # save vid_items_dict
                # save_dict_to_json(vid_items_dict, vid_items_dict_path)
                # fix.report_progress("UPDATE", vid_items_dict_path)

        # No longer in middle of request
        progress_dict["middle_of_playlist_request"] = False
        del progress_dict["params"]["pageToken"]
        return progress_dict, vid_items_dict


    for upload_i in range(progress_dict["upload_pl_index"], len(channels_df)):
        # Just confirming
        progress_dict["params"]["playlistId"] = channels_df.loc[upload_i, "channel_uploads_pl"]
        progress_dict["channel_id"] = channels_df.loc[upload_i, "channel_id"]

        # initialise vid_id_dict entry, if there was no previous entry
        # this is response items list
        if not vid_items_dict.get(progress_dict["channel_id"]):
            vid_items_dict[progress_dict["channel_id"]] = []

        # requests
        if not progress_dict["middle_of_playlist_request"]:
            progress_dict, vid_items_dict = request_wo_token(progress_dict, vid_items_dict)
            if progress_dict["middle_of_playlist_request"]:
                progress_dict, vid_items_dict = request_with_token(progress_dict, vid_items_dict)
        else:
            progress_dict, vid_items_dict = request_with_token(progress_dict, vid_items_dict)

        # If we've completed the requests for the playlist, update progress_dict
        if progress_dict["upload_pl_index"] + 1 < len(channels_df) - 1:
            progress_dict["upload_pl_index"] += 1
        else:
            progress_dict["upload_pl_index"] = 0
        # Update progress dict and vid items dict
        use.save_dict_to_json(progress_dict, progress_dict_path)
        use.report_progress("UPDATE", progress_dict_path)
        use.save_dict_to_json(vid_items_dict, vid_items_dict_path)
        use.report_progress("UPDATE", vid_items_dict_path)

        # Report progress
        use.report_progress(f"{upload_i} FINISHED channel_id: {progress_dict['channel_id']}", "")
    return vid_items_dict


# def filter_for_unpopular_videos(potential_range_info, uploads_items_by_channel):
#     # {channel_id: [only unpopular video ids]}
#     # return unpopular_video_ids_by_channel
#     pass


def get_unpopular_vid_info(youtube, vid_id_dict, temp_path):
    progress_path = temp_path / "progress.json"
    unpopular_path = temp_path / "unpopular.pkl"
    def convert_vid_id_dict_to_params_list(vid_id_dict):
        i = 0
        params_list = {}
        for channel_id, idlists in vid_id_dict.items():
            for idlist in idlists:
                params_list[i] = {}
                params_list[i]["channel_id"] = channel_id
                params_list[i]["params"] = {
                    "id": ",".join(idlist),
                    "part":"snippet,statistics,contentDetails,paidProductPlacementDetails,liveStreamingDetails",
                }
                i += 1
        return params_list
    def send_request(params:dict):
        request = youtube.videos().list(
            part=params.get("part"),
            id=params.get("id")
        )
        response = request.execute()
        return response
    def get_video_info_row(item):
        row = {}
        row["id"] = item.get('id')
        row["published_at"] = item["snippet"].get("publishedAt")
        row["channel_id"] = item["snippet"].get("channelId")
        row["title"] = item["snippet"].get("title")
        row["description"] = item["snippet"].get("description")
        row["channel_title"] = item["snippet"].get("channelTitle")
        row["thumbnail_url"] = item["snippet"].get("thumbnails", {}).get("default", {}).get("url")
        row["tags"] = item["snippet"].get("tags")
        row["default_language"] = item["snippet"].get("defaultLanguage")
        row["default_audio_language"] = item["snippet"].get("defaultAudioLanguage")
        row["duration"] = item["contentDetails"].get("duration")
        row["3d_or_2d"] = item["contentDetails"].get("dimension")
        row["vid_definition"] = item["contentDetails"].get("definition")
        row["has_captions"] = item["contentDetails"].get("caption")
        row["represents_licensed_content"] = item["contentDetails"].get("licensedContent")
        row["blocked_region"] = item["contentDetails"].get("regionRestriction", {}).get("blocked")
        row["content_rating_kr"] = item["contentDetails"]["contentRating"].get("kmrbRating")
        row["content_rating_yt"] = item["contentDetails"]["contentRating"].get("ytRating")
        row["views"] = item["statistics"].get("viewCount")
        row["likes"] = item["statistics"].get("likeCount")
        row["dislikes"] = item["statistics"].get("dislikeCount")
        row["comments"] = item["statistics"].get("commentCount")
        row["has_ppl"] = item.get("paidProductPlacementDetails", {}).get("hasPaidProductPlacement")
        row["stream_actual_start"] = item.get("liveStreamingDetails", {}).get("actualStartTime")
        row["stream_actual_end"] = item.get("liveStreamingDetails", {}).get("actualEndTime")
        row["stream_sch_start"] = item.get("liveStreamingDetails", {}).get("scheduledStartTime")
        row["stream_sch_end"] = item.get("liveStreamingDetails", {}).get("scheduledEndTime")
        return row
    def process_response(response):
        row_list = []
        for item in response["items"]:
            row = get_video_info_row(item)
            row_list.append(row)
        return pd.DataFrame(row_list)
    # convert vid_id_dict to params list
    params_list = convert_vid_id_dict_to_params_list(vid_id_dict)

    df_list = []
    total_params = len(params_list)
    for param_index, param_bundle in params_list.items():
        print(param_index, "/", total_params)
        try:
            response = send_request(param_bundle["params"])
        except Exception as e:
            print("Error! Progress:", param_index)
            with open(progress_path, "w") as f:
                f.write(str(param_index))
            if df_list:
                pd.concat(df_list).reset_index().to_pickle(unpopular_path)
                use.report_progress("EMERGENCY SAVE", unpopular_path)
            return
        else:
            df = process_response(response)
            df_list.append(df)
    unpopular_raw = pd.concat(df_list).reset_index()
    unpopular_raw.to_pickle(unpopular_path)
    use.report_progress("SAVE", unpopular_path)
    return unpopular_raw


def cleanse_tabular_data(popular_df_raw, unpopular_df_raw, dataset_dir):
    from data_processing import cleansing
    def cleanse_vid_df(df:pd.DataFrame, cleansing:dict):
        cleansed_df = pd.DataFrame([])
        cleansed_df["video_id"] = df["id"]
        cleansed_df["length_of_title"] = df["title"].apply(cleansing["length_of_title"])
        cleansed_df["length_of_description"] = df["description"].apply(cleansing["length_of_description"])
        cleansed_df["number_of_tags"] = df["tags"].apply(cleansing["number_of_tags"])
        def_lang_cleansed = df["default_language"].apply(cleansing["default_language"])
        def_audio_cleansed = df["default_audio_language"].apply(cleansing["default_audio_language"])
        cleansed_df["duration"] = df["duration"].apply(cleansing["duration"])
        cleansed_df["has_captions"] = df["has_captions"].apply(cleansing["has_captions"])
        cleansed_df["represents_licensed_content"] = df["represents_licensed_content"].apply(cleansing["represents_licensed_content"])
        cleansed_df["is_blocked_somewhere"] = df["blocked_region"].apply(cleansing["is_blocked_somewhere"])
        cleansed_df["views"] = df["views"].apply(cleansing["views"])
        likes = df["likes"].apply(cleansing["likes"])
        comments = df["comments"].apply(cleansing["comments"])
        cleansed_df["has_ppl"] = df["has_ppl"].apply(cleansing["has_ppl"])
        cleansed_df["was_streamed"] = df["stream_actual_start"].apply(cleansing["was_streamed"])
        cleansed_df = pd.concat([cleansed_df, def_lang_cleansed, def_audio_cleansed, likes, comments], axis=1)
        cleansed_df = cleansed_df.dropna() # Found an error regarding livestream video and duration, decided to drop rather than to make it some other value.
        return cleansed_df

    def label_dfs(df_list:list, labels:list, label_name="target") -> list:
        """
        Returns labeled dataframes with column name of `label_name`.
        ## Input
        - df_list: list of pandas.DataFrames
        - labels: corresponding list of label values
        - label_name: name of the label column, by default `target`
        ## Output
        - list of labeled pandas.DataFrames
        """
        # assert that the lengths of df_list and labels match up
        assert len(df_list) == len(labels)
        result = []
        for df, label in zip(df_list, labels):
            # assert that label_name should not already exist in the dataframes
            assert not df.get(label_name, False)
            df[label_name] = label
            result.append(df)
        return result
    def drop_vlc(df:pd.DataFrame):
        result = df.drop(["views", "likes", "private-likes", "comments", "private-comments"], axis=1)
        return result

    def construct_label_info(df:pd.DataFrame, target_name="target"):
        result = df[["video_id", target_name]].copy()
        return result


    popular_cleansed = cleanse_vid_df(popular_df_raw, cleansing)
    unpopular_cleansed = cleanse_vid_df(unpopular_df_raw, cleansing)

    popular_labeled, unpopular_labeled = label_dfs([popular_cleansed, unpopular_cleansed], [1, 0], "target")
    tabular_labeled = pd.concat([popular_labeled, unpopular_labeled]).reset_index(drop=True)
    tabular_labeled_no_vlc = drop_vlc(tabular_labeled)
    label_info = construct_label_info(tabular_labeled)
    
    use.create_directory_if_not_exists(dataset_dir)
    tabular_labeled.to_pickle(dataset_dir / "tabular_features_vlc.pkl")
    tabular_labeled_no_vlc.to_pickle(dataset_dir / "tabular_features_no_vlc.pkl")
    label_info.to_pickle(dataset_dir/"label_info.pkl")
    return tabular_labeled, tabular_labeled_no_vlc, label_info

def get_thumbnails(popular_df_raw, unpopular_df_raw, thumbnails_storage_dir):
    # Download thumbnails and store in thumbnails_storage_dir, with video id as file name
    popular_id_url = popular_df_raw[["id", "thumbnail_url"]]
    unpopular_id_url = unpopular_df_raw[["id", "thumbnail_url"]]

    def switch_to_hq(url_str):
        split_url = url_str.split("/")
        split_url[-1] = "hqdefault.jpg"
        rejoined_url = "/".join(split_url)
        return rejoined_url

    popular_id_url.loc[:, "thumbnail_url"] = popular_id_url["thumbnail_url"].apply(switch_to_hq)
    unpopular_id_url.loc[:, "thumbnail_url"] = unpopular_id_url["thumbnail_url"].apply(switch_to_hq)

    use.create_directory_if_not_exists(thumbnails_storage_dir)

    def download_image(progress_id:int, video_id:str, image_url:str, save_folder=DATA_PATH / "thumbnails"):
        """
        Downloads an image from a URL and saves it to a specified folder.

        Args:
            image_url (str): The URL of the image to download.
            save_folder (str, optional): The folder to save the image in. Defaults to DATA_PATH / "popular_vids".
        """
        import requests
        try:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes

            use.create_directory_if_not_exists(save_folder) # Create the folder if it doesn't exist
 
            filename = video_id + ".jpg"
            filepath = save_folder / filename

            with open(filepath, 'wb') as file:
                file.write(response.content)

            use.report_progress("DOWNLOAD", f"{filepath} --{progress_id}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {image_url}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {image_url}: {e}")
    for i, (video_id, image_url) in popular_id_url.iterrows():
        download_image(i, video_id, image_url, thumbnails_storage_dir / "1")
    for i, (video_id, image_url) in unpopular_id_url.iterrows():
        download_image(i, video_id, image_url, thumbnails_storage_dir / "0")
    return thumbnails_storage_dir / "1", thumbnails_storage_dir / "0"

#TODO
def process_thumbnails(thumbnails_storage_dir, label_info):
    # Extract image features from thumbnails
    # return thumbnail_features
    pass

#TODO
def process_titles(popular_df_raw, unpopular_df_raw, label_info):
    # Extract text features from titles
    # return title_features
    pass

#TODO
def get_videos(label_info, video_storage_dir):
    # download videos into video storage dir
    pass

#TODO
def process_videos(video_storage_dir, label_info):
    # return video_features
    pass

#TODO
def construct_final_dataset(tabular_features, thumbnail_features, title_features, video_features):
    # return final_dataset
    pass


########################################################################################################################
#TODO
#########################################################################################################################
def merge_datasets(data_dict:dict, keys_input:str|list):
    """
    Merges pandas DataFrames from a dictionary based on the provided keys.

    Args:
        data_dict (dict): A dictionary where keys are integers and values are pandas DataFrames.
        keys_input (str or list): A string of keys (e.g., "012") or a list of keys (e.g., [0, 1, 2])
                                   indicating which DataFrames to merge.

    Returns:
        pandas.DataFrame: A single DataFrame resulting from the inner merge of the selected DataFrames
                          on the columns "video_id" and "target". Returns None if no valid keys are provided
                          or if the dictionary is empty.
    """
    if not data_dict:
        print("Error: The input dictionary is empty.")
        return None

    if isinstance(keys_input, str):
        try:
            keys_to_merge = [int(k) for k in keys_input]
        except ValueError:
            print("Error: Invalid key string. Please use digits only.")
            return None
    elif isinstance(keys_input, list):
        keys_to_merge = [int(k) for k in keys_input]
    else:
        print("Error: Invalid input for keys. Please provide a string or a list of keys.")
        return None

    valid_dataframes = [data_dict.get(key) for key in keys_to_merge if key in data_dict and isinstance(data_dict[key], pd.DataFrame)]

    if not valid_dataframes:
        print("Error: No valid DataFrames found for the provided keys.")
        return None

    if len(valid_dataframes) == 1:
        return valid_dataframes[0].copy()  # Return a copy to avoid modifying the original

    merged_df = valid_dataframes[0]
    for i in range(1, len(valid_dataframes)):
        merged_df = pd.merge(merged_df, valid_dataframes[i], on=["video_id", "target"], how="inner")

    return merged_df

def dataset_construction(instruction:str|list, locations:dict):
    """
    Constructs dataset according to instruction.
    
    Args:
        instruction (str|list): Instructions on constructing dataset 
        locations (dict): Where to find the feature datasets
    
    # For instructions
    - 0: tabular_with_vlc
    - 1: tabular_without_vlc
    - 2: title
    - 3: thumbnail
    - 4: video_l1
    - 5: video_xclip
    """
    features = {}
    ## Load tabular data
    tabular_features = pd.read_pickle(locations["tabular_features_vlc"])
    tabular_features_no_vlc = pd.read_pickle(locations["tabular_features_no_vlc"])    
    features[0] = tabular_features
    features[1] = tabular_features_no_vlc
    # Load title data
    title_features = pd.read_pickle(locations["title_features"])
    features[2] = title_features
    # Load thumbnail data
    thumbnail_features = pd.read_pickle(locations["thumbnail_features"])
    features[3] = thumbnail_features
    # Load video l1 features
    video_l1_features = pd.read_pickle(locations["video_l1_features"])
    features[4] = video_l1_features
    # Load video xclip features
    video_xclip_features = pd.read_pickle(locations["video_xclip_features"])
    features[5] = video_xclip_features
 
    # Merge according to instruction
    resulting_df = merge_datasets(features, instruction)

    # Separate X and y
    X_columns = resulting_df.columns.difference(["target"])
    result_X = resulting_df[X_columns].drop("video_id", axis=1).values
    result_y = resulting_df["target"].values
    return result_X, result_y

#%%
import useful as use
import pandas as pd
import numpy as np
from pathlib import Path

priv = use.get_priv()
DATA_PATH = Path(priv["DATA_PATH"])
new_dataset_dir = Path("../Datasets")

tabular_features = pd.read_pickle(DATA_PATH / "dataset/tabular_features_vlc.pkl")
tabular_features_no_vlc = pd.read_pickle(DATA_PATH/"dataset/tabular_features_no_vlc.pkl")
label_info = pd.read_pickle(DATA_PATH/"dataset/label_info.pkl")

popular_ids = label_info[label_info["target"] == 1]["video_id"]
num_of_popular = len(popular_ids)
unpopular_ids = label_info[label_info["target"] == 0]["video_id"]
sampled_unpopular_ids = unpopular_ids.sample(num_of_popular * 2, random_state=86)

ids_to_keep = pd.concat([popular_ids, sampled_unpopular_ids])
resampled_tabular_features = tabular_features[tabular_features["video_id"].isin(ids_to_keep)]
resampled_tabular_features_no_vlc = tabular_features_no_vlc[tabular_features_no_vlc["video_id"].isin(ids_to_keep)]

resampled_tabular_features.to_pickle(new_dataset_dir/"tabular_features_vlc.pkl")
resampled_tabular_features_no_vlc.to_pickle(new_dataset_dir/"tabular_features_no_vlc.pkl")
features = {}
features[0] = tabular_features
features[1] = tabular_features_no_vlc
with np.load(DATA_PATH/"dataset/popular_title_features.npz") as data:
    keys = data.files
    vector_arrays = [data[key] for key in keys]
    num_elements = vector_arrays[0].shape[0] if vector_arrays else 512 # Default to 512 if no arrays
    vector_column_names = [f'title_{i}' for i in range(num_elements)]

    # Construct the DataFrame
    df = pd.DataFrame(vector_arrays, index=keys, columns=vector_column_names)

    # Reset the index to make the keys a regular column
    df = df.reset_index()
    df = df.rename(columns={'index': 'video_id'})
    # Add label
    df["target"] = 1
    
    popular_title_features = df

with np.load(DATA_PATH/"dataset/unpopular_title_features.npz") as data:
    keys = sampled_unpopular_ids
    vector_arrays = [data[key] for key in keys]
    num_elements = vector_arrays[0].shape[0] if vector_arrays else 512 # Default to 512 if no arrays
    vector_column_names = [f'title_{i}' for i in range(num_elements)]

    # Construct the DataFrame
    df = pd.DataFrame(vector_arrays, index=keys, columns=vector_column_names)

    # Reset the index to make the keys a regular column
    df = df.reset_index()
    df = df.rename(columns={'index': 'video_id'})
    # Add label
    df["target"] = 0
    
    unpopular_title_features = df

with np.load(DATA_PATH/"dataset/popular_thumbnail_features.npz") as data:
    keys = data.files
    vector_arrays = [data[key] for key in keys]
    num_elements = vector_arrays[0].shape[0] if vector_arrays else 512 # Default to 512 if no arrays
    vector_column_names = [f'thumbnail_{i}' for i in range(num_elements)]
    # Construct the DataFrame
    df = pd.DataFrame(vector_arrays, index=keys, columns=vector_column_names)

    # Reset the index to make the keys a regular column
    df = df.reset_index()
    df = df.rename(columns={'index': 'video_id'})
    # Add label
    df["target"] = 1

    popular_thumbnail_features = df

with np.load(DATA_PATH/"dataset/unpopular_thumbnail_features.npz") as data:
    keys = sampled_unpopular_ids
    vector_arrays = [data.get(key) for key in keys]
    num_elements = vector_arrays[0].shape[0] if vector_arrays else 512 # Default to 512 if no arrays
    vector_column_names = [f'thumbnail_{i}' for i in range(num_elements)]
    # Construct the DataFrame
    df = pd.DataFrame(vector_arrays, index=keys, columns=vector_column_names)

    # Reset the index to make the keys a regular column
    df = df.reset_index()
    df = df.rename(columns={'index': 'video_id'})
    # Add label
    df["target"] = 0
    
    unpopular_thumbnail_features = df

title_features = pd.concat([popular_title_features, unpopular_title_features])
title_features.to_pickle(new_dataset_dir / "title_features.pkl")
thumbnail_features = pd.concat([popular_thumbnail_features, unpopular_thumbnail_features])
thumbnail_features.to_pickle(new_dataset_dir / "thumbnail_features.pkl")
features[2] = title_features
features[3] = thumbnail_features

#%%
use.create_directory_if_not_exists("../Temp")
popular_ids.to_pickle(Path("../Temp") / "popular_ids.pkl")
sampled_unpopular_ids.to_pickle(Path("../Temp") / "unpopular_ids.pkl")

import json
import pickle
with open(DATA_PATH / "dataset/popular_l1_norms.json") as rf:
    popular_l1_norms = json.load(rf)

with open(DATA_PATH / "dataset/unpopular_l1_norms.pkl", "rb") as rf:
    unpopular_l1_norms = pickle.load(rf)

with open(DATA_PATH / "dataset/unpopular_video_features.pkl", "rb") as rf:
    unpopular_video_features = pickle.load(rf)

with open(Path("../Temp") / "popular_l1_norms.pkl", "wb") as wf:
    pickle.dump(popular_l1_norms, wf)

with open(Path("../Temp") / "unpopular_l1_norms.pkl", "wb") as wf:
    pickle.dump(unpopular_l1_norms, wf)

with open(Path("../Temp") / "unpopular_video_features.pkl", "wb") as wf:
    pickle.dump(unpopular_video_features, wf)



ids_to_query = []
for vid_id in sampled_unpopular_ids:
    if vid_id not in unpopular_video_features.keys():
        ids_to_query.append(vid_id)
len(ids_to_query)

#%%
#### dataset name
# 0: tabular_with_vlc
# 1: tabular_without_vlc
# 2: title
# 3: thumbnail
# 4: video_l1
# 5: video
# locations = {
#     "tabular_features_vlc": DATA_PATH / "dataset/tabular_features_vlc.pkl",
#     "tabular_features_no_vlc": DATA_PATH / "dataset/tabular_features_no_vlc.pkl",
#     "title_features": DATA_PATH/"dataset/title_features.pkl",
#     "thumbnail_features": DATA_PATH/"dataset/thumbnail_features.pkl",
# }
# result_dataset = dataset_construction("123", locations)
# X_columns = result_dataset.columns.difference(["target"])
# result_X = result_dataset[X_columns]
# result_y = result_dataset["target"]
#%%
# Get xclip and l1 features
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

temp_dir = Path("../Temp")
dataset_dir = Path("../Datasets")

with open(temp_dir / "popular_l1_norms.pkl", "rb") as fr:
    popular_l1 = pickle.load(fr)

with open(temp_dir / "unpopular_l1_norms.pkl", "rb") as fr:
    unpopular_l1 = pickle.load(fr)

with open(temp_dir / "popular_xclip_features.pkl", "rb") as fr:
    popular_xclip = pickle.load(fr)

with open(temp_dir / "unpopular_xclip_features.pkl", "rb") as fr:
    unpopular_xclip = pickle.load(fr)

# Even though we don't have enough unpopular vids, I'm putting the dataset together
# If possible, run again when more unpopular vids have been collected.


# Process l1_norms
def extract_fixed_length_features(l1_norms, num_segments=5, percentile_values=[10, 30, 50, 70, 90], num_bins=10):
    """
    Extracts a fixed-length feature vector from a sequence of L1 norms.

    Args:
        l1_norms (np.ndarray): A 1D NumPy array representing the sequence of L1 norms.
        num_segments (int): The number of segments to divide the L1-norm sequence into.
        percentile_values (list): A list of percentile values to calculate (e.g., [25, 50, 75]).
        num_bins (int): The number of bins for the histogram.

    Returns:
        np.ndarray: A 1D NumPy array containing the fixed-length feature vector.
    """
    features = []
    n = len(l1_norms)

    # 1. Segment-based statistics
    if n > 0 and num_segments > 0:
        segment_size = n // num_segments
        remainder = n % num_segments
        start = 0
        for i in range(num_segments):
            end = start + segment_size + (1 if i < remainder else 0)
            if start < end:
                segment = l1_norms[start:end]
                features.extend([np.mean(segment), np.std(segment)])
            start = end
    elif n > 0:
        # Handle case where num_segments is 0 or sequence is too short
        features.extend([np.mean(l1_norms), np.std(l1_norms)])
    else:
        # Handle empty sequence
        features.extend([0.0] * 2 * num_segments) # Add placeholders

    # 2. Percentiles
    percentile_features = np.percentile(l1_norms, percentile_values)
    features.extend(percentile_features)

    # 3. Histogram
    hist, _ = np.histogram(l1_norms, bins=num_bins)
    features.extend(hist)

    return np.array(features)

popular_l1_features = {}
for video_id, l1_norms in popular_l1.items():
    l1_features = extract_fixed_length_features(np.array(l1_norms))
    popular_l1_features[video_id] = l1_features

video_ids = popular_l1_features.keys()
l1_features = [popular_l1_features[vid_id] for vid_id in video_ids]
num_elements = l1_features[0].shape[0]
column_names = [f"l1_{i}"for i in range(num_elements)]
popular_l1_df = pd.DataFrame(l1_features, index=video_ids,columns=column_names)
popular_l1_df = popular_l1_df.reset_index()
popular_l1_df = popular_l1_df.rename(columns={"index": "video_id"})
popular_l1_df["target"] = 1


unpopular_l1_features = {}
for video_id, l1_norms in unpopular_l1.items():
    if l1_norms is None:
        continue
    l1_features = extract_fixed_length_features(np.array(l1_norms))
    unpopular_l1_features[video_id] = l1_features

video_ids = unpopular_l1_features.keys()
l1_features = [unpopular_l1_features[vid_id] for vid_id in video_ids]
num_elements = l1_features[0].shape[0]
column_names = [f"l1_{i}"for i in range(num_elements)]
unpopular_l1_df = pd.DataFrame(l1_features, index=video_ids,columns=column_names)
unpopular_l1_df = unpopular_l1_df.reset_index()
unpopular_l1_df = unpopular_l1_df.rename(columns={"index": "video_id"})
unpopular_l1_df["target"] = 0

video_l1_features = pd.concat([popular_l1_df, unpopular_l1_df])
video_l1_features.to_pickle(dataset_dir / "video_l1_features.pkl")

# now xclip features
print(len(popular_xclip))
popular_xclip_features = {video_id: xclip_features for video_id, xclip_features in popular_xclip.items() if xclip_features is not None}
print(len(popular_xclip_features))
video_ids = popular_xclip_features.keys()
xclip_features = [popular_xclip_features[vid_id] for vid_id in video_ids]
num_elements = xclip_features[0].shape[0]
column_names = [f"xclip_{i}" for i in range(num_elements)]
df = pd.DataFrame(xclip_features, 
                  index=video_ids,
                  columns=column_names)
df = df.reset_index()
df = df.rename(columns={"index":"video_id"})
df["target"] = 1
popular_xclip_df = df


print(len(unpopular_xclip))
unpopular_xclip_features = {video_id: xclip_features for video_id, xclip_features in unpopular_xclip.items() if xclip_features is not None}
print(len(unpopular_xclip_features))
video_ids = unpopular_xclip_features.keys()
xclip_features = [unpopular_xclip_features[vid_id] for vid_id in video_ids]
num_elements = xclip_features[0].shape[0]
column_names = [f"xclip_{i}" for i in range(num_elements)]
df = pd.DataFrame(xclip_features, 
                  index=video_ids,
                  columns=column_names)
df = df.reset_index()
df = df.rename(columns={"index":"video_id"})
df["target"] = 1
unpopular_xclip_df = df

video_xclip_features = pd.concat([popular_xclip_df, unpopular_xclip_df])
video_xclip_features.to_pickle(dataset_dir / "video_xclip_features.pkl")
display(video_xclip_features)