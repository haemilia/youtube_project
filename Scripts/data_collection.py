from pathlib import Path
import pandas as pd
import useful as use

## Get video categories list of Korea
def get_video_categories(youtube, data_path:str|Path, region_code="KR", hl="ko_KR") -> bool:
    """
    Requests to the Youtube Data API for video categories of Korea. 
    Saves the results as a DataFrame json file at `data_path`.
    Returns `True` if process was performed successfully. 
    # Args
    - youtube : 
        - Youtube Data API access resource object by googleapiclient
    - data_path : str|pathlib.Path
        - The path to where the video categories DataFrame should be saved to
    - region_code : str
        - The regionCode parameter instructs the API to return the list of video categories available in the specified country. 
        - The parameter value is an ISO 3166-1 alpha-2 country code.
        - The default value is "KR"
    - hl : str
        - The hl parameter specifies the language that should be used for text values in the API response. 
        - The default value is "ko-KR".
    """
    try:
        request = youtube.videoCategories().list(
            part="snippet",
            hl=hl, # Retrieves category name in Korean
            regionCode=region_code # Retrieves categories of Korean Youtube by default
        )
        response = request.execute()
        use.report_progress("SEND REQUEST", "Video Categories")
    except Exception as e:
        use.report_progress("FAILED REQUEST", "Video Categories")
        raise Exception("Error happened while retrieving video categories list")

    # Minimal processing for saving    
    video_categories = []
    for item in response['items']:
        video_category_row = {}
        video_category_row['id'] = item['id']
        video_category_row['title'] = item['snippet']['title']
        video_categories.append(video_category_row)
    video_category_df = pd.DataFrame(video_categories)
    video_category_df.to_json(data_path)
    use.report_progress("SAVE", data_path)
    return True

## Retrieve video categories information as dictionary of my wanted structure
def make_vid_cat_dict(path) -> tuple:
    """
    Converts Video Category DataFrame into same Dataframe with just the "id" column converted to string, 
    and a dictionary that allows easy conversion from "id" to the Video Category title/name.
    Returns a tuple of the converted pd.DataFrame and a Python dictionary.
    # Parameters:
    - path: path to `youtube_korea_video_categories.json`; contains vid_cat_df
    - (vid_cat_df: pd.DataFrame; has 2 columns(["id", "title"])
        - "id": contains videoCategoryId of Youtube videoCategories
        - "title": containes each id's title/name)
    # Output:
    - tuple:
        - vid_cat_df: Dataframe with just the "id" column converted to string
        - vid_cat_id_to_name_dict: dictionary that allows easy conversion from "id" to the Video Category title/name
    """
    vid_cat_df = pd.read_json(path)
    vid_cat_df["id"] = vid_cat_df["id"].astype(str)
    vid_cat_id_to_name_dict = vid_cat_df.set_index("id").to_dict()["title"]
    return vid_cat_df, vid_cat_id_to_name_dict

## Get mostPopular video information from each video category
def get_video_info(youtube, category_id:str = "0"):
    """
    Requests information about mostPopular videos to Youtube Data API's `Videos` resource
    # Args
    - youtube:
        - Youtube Data API access resource object by googleapiclient
    - category_id: (str)
        - The videoCategoryId parameter identifies the video category for which the chart should be retrieved.  
        - By default, charts are not restricted to a particular category. 
        - The default value is 0.
    """

    def get_video_info_row(item):
        row = {}
        row["id"] = item.get('id')
        row["published_at"] = item["snippet"].get("publishedAt")
        row["channel_id"] = item["snippet"].get("channelId")
        row["title"] = item["snippet"].get("title")
        row["description"] = item["snippet"].get("description")
        row["channel_title"] = item["snippet"].get("channelTitle")
        row["thumbnail_url"] = item["snippet"].get("thumbnails", {}).get("high", {}).get("url")
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
    
    try:
        if category_id:
            request = youtube.videos().list(
                part="snippet,statistics,contentDetails,paidProductPlacementDetails,liveStreamingDetails",
                chart="mostPopular",
                regionCode="KR",
                maxResults=50,  # max value = 50
                videoCategoryId=category_id
            )

            response = request.execute()
            use.report_progress("SEND REQUEST", "Video Info Popular")
            next_page_token = response.get("nextPageToken")
        else:
            request = youtube.videos().list(
                part="snippet,statistics,contentDetails,paidProductPlacementDetails,liveStreamingDetails",
                chart="mostPopular",
                regionCode="KR",
                maxResults=50,  # max value = 50
                )

            response = request.execute()
            use.report_progress("SEND REQUEST", "Video Info Popular")
            next_page_token = response.get("nextPageToken")
    except Exception as e:
        # if we weren't able to get any results, just return nothing, and say that we failed to get results.
        return False, None
    

    # What happens if we did manage to get results from request:
    df_rows = []

    for item in response["items"]:  
            row = get_video_info_row(item)
            df_rows.append(row)

    
    while next_page_token:
        try:
            if category_id:
                request = youtube.videos().list(
                    part="snippet,statistics,contentDetails,paidProductPlacementDetails,liveStreamingDetails",
                    chart="mostPopular",
                    regionCode="KR",
                    maxResults=50,
                    videoCategoryId=category_id,
                    pageToken=next_page_token # retrieve next page
                )
            else:
                request = youtube.videos().list(
                    part="snippet,statistics,contentDetails,paidProductPlacementDetails,liveStreamingDetails",
                    chart="mostPopular",
                    regionCode="KR",
                    maxResults=50,
                    pageToken=next_page_token # retrieve next page
                )

            response = request.execute()
            use.report_progress("SEND REQUEST", "Video Info Popular")
            next_page_token = response.get("nextPageToken")

            for item in response["items"]:  
                row = get_video_info_row(item)
                df_rows.append(row)
        except Exception as e:
            return False, None
    return True, pd.DataFrame(df_rows)

# Process df
# Add collection date column, add category_id column
def process_df(df:pd.DataFrame, collection_date:str, category_id:str) -> pd.DataFrame:
    """
    Adds collection date column, category_id column to DataFrames
    # Args
    - df: pandas.DataFrame
        - The DataFrame that `collection_date` and `category_id` columns will be added to
    - collection_date:
        - The date to add
    - category_id:
        - The category id information to add
    """
    # Add data collection date to dataframe
    def add_data_collection_date(df:pd.DataFrame) -> pd.DataFrame:
        from datetime import datetime

        now = datetime.now()
        collection_time= now.strftime("%Y-%m-%d %H:%M:%S")
        df['collection_date'] = pd.Timestamp(" ".join([collection_date, collection_time]))
        return df
    # Add category id to dataframe
    def add_category_id(df:pd.DataFrame, cat_id:str) -> pd.DataFrame:
        df['category_id'] = cat_id
        return df
    
    col_date_added = add_data_collection_date(df, collection_date)
    cat_id_added = add_category_id(col_date_added, category_id)
    return cat_id_added

# Save dataframe with designated format in destination path
def save_df_format0(df:pd.DataFrame, destination_dir:str|Path, collection_date:str, category_id:str)-> Path:
    """
    Save dataframe as pickle in designated format and path.
    Returns full path to dataframe pickle file.
    # Args
    - df: pandas.DataFrame
        - The dataframe that is to be written
    - destination_path: pathlib.Path
        - The directory in which the pickle file is to be written
    - collection_date: str
        - Date information necessary for filename format
    - category_id: str
        - Category id information necessary for filename format
    """
    if category_id == "None":
        category_id = "0"
    df_filename = "_".join([collection_date, category_id]) + ".pkl"
    destination_dir = Path(destination_dir)# ensure that this is pathlib path
    df.to_pickle(destination_dir / df_filename)
    return destination_dir / df_filename

# For categorised popular videos
def get_categorised_popular_vids(youtube, files_created_today:dict, vid_cat_df:pd.DataFrame, popular_vids_path:Path|str, collection_date:str):
    """
    Request Youtube Data API for most popular videos in given categories.
    Categories are given through `vid_cat_df` format.
    If successful, returns True, and the resulting dictionary of pandas Dataframe values.
    If unsuccessful, returns False, and None.
    # Args
    - youtube:
        - Youtube Data API access resource object by googleapiclient
    - vid_cat_df: pandas.DataFrame
        - DataFrame containing video category ids
    - popular_vids_path: pathlib.Path|str
        - Path to directory for popular video information
    - collection_date: str
        - Date of data collection
    """
    df_dict = {}
    for i in vid_cat_df["id"].astype(str):
        success, df = get_video_info(youtube, i)
        if success:
            df = process_df(df, collection_date=collection_date, category_id=str(i))
            df_dict[i] = df
            saved_to = save_df_format0(df, popular_vids_path, collection_date, str(i))
            files_created_today["popular_vids"][i] = saved_to
            use.report_progress("SAVE", saved_to)
    if df_dict:
        return True, df_dict, files_created_today
    else:
        return False, None, files_created_today
# For general popular videos
def get_general_popular_vids(youtube, files_created_today:dict, popular_vids_path:Path|str, collection_date:str):
    """
    Request Youtube Data API for most popular videos in general category (0). 
    If successful, returns True, and the resulting pandas Dataframe.
    If unsuccessful, returns False, and None.
    # Args
    - youtube:
        - Youtube Data API access resource object by googleapiclient
    - popular_vids_path: pathlib.Path|str
        - Path to directory for popular video information
    - collection_date: str
        - Date of data collection
    """
    # Changed from 'None' to 0, because it's much easier to work with, and we have no 0 category id anyways.
    i = 0
    success, df = get_video_info(youtube, i)
    if success:
        df = process_df(df, collection_date=collection_date, category_id=str(i))
        saved_to = save_df_format0(df, popular_vids_path, collection_date, str(i))
        files_created_today["popular_vids"][str(i)] = saved_to
        use.report_progress("SAVE", saved_to)
        return True, df, files_created_today
    else:
        return False, None, files_created_today

# Request for channel info
def get_channel_info(youtube, df:pd.DataFrame):
    """
    Requests Youtube Data API for channel information.
    List of channels given through pandas DataFrame.
    # Args
    - youtube:
        - Youtube Data API access resource object by googleapiclient
    - df: pandas.DataFrame
        - DataFrame that contains a column `channel_id` that stores channel ids to query with
    """
    def get_channel_info_row(item, channel_id):
        row = {}
        row["channel_id"] = channel_id
        row["channel_title"] = item["snippet"].get("title")
        row["channel_description"] = item["snippet"].get("description")
        row["custom_url"] = item["snippet"].get("customUrl")
        row["channel_published_at"] = item["snippet"].get("publishedAt")
        row["channel_default_language"] = item["snippet"].get("defaultLanguage")
        row["channel_country"] = item["snippet"].get("country")
        row["channel_uploads_pl"] = item["contentDetails"].get("relatedPlaylists", {}).get("uploads")
        row["channel_views"] = item["statistics"].get("viewCount")
        row["channel_subs"] = item["statistics"].get("subscriberCount")
        row["can_upload_long_vids"] = item["status"].get("longUploadStatus")
        row["made_for_kids"] = item["status"].get("madeForKids")
        return row
    channel_rows = []
    for channel_id in df["channel_id"].unique():
        try:
            request = youtube.channels().list(
                    part="snippet,contentDetails,statistics, status",
                    id=channel_id
                )
            response = request.execute()
            channel_rows.append(get_channel_info_row(response["items"][0], channel_id))
        except Exception as e:
            channel_rows.append({"channel_id": channel_id})
    return pd.DataFrame(channel_rows)


# Get channel info    
def get_channel_infos(youtube, files_created_today:dict, channel_path:Path|str, collection_date:str):
    """
    Requests Youtube Data API for channel information, then saves data in `channel_path`.
    Returns True if successful, returns False if not.
    # Args
    - youtube:
        - Youtube Data API access resource object by googleapiclient
    - channel_path: pathlib.Path | str
        - Path to directory where channel information is to be stored
    - collection_date: str
        - Date of data collection
    """
    pop_video_file_paths = files_created_today.get("popular_vids")
    if pop_video_file_paths is None:
        return False
    use.create_directory_if_not_exists(channel_path)
    for cat_id, pop_vids in pop_video_file_paths.items():
        pop_df = pd.read_pickle(pop_vids)
        ch_df = get_channel_info(youtube, pop_df)
        ch_df = process_df(ch_df, 
                           collection_date=collection_date, 
                           category_id=str(cat_id))
        saved_to = save_df_format0(ch_df, channel_path, collection_date ,str(cat_id))
        files_created_today["channels"][str(cat_id)] = saved_to
        use.report_progress("SAVE", saved_to)
    return True, files_created_today

