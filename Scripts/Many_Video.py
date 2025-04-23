
import pandas as pd
import numpy as np
import pathlib
import fix
import useful as use

## Process of getting API Key and path for storing the resulting data
priv = use.get_priv()
API_KEY = priv["API_KEY"]
DATA_PATH = pathlib.Path(priv["DATA_PATH"])
PROGRESS_PATH = DATA_PATH / "progress"

## Today's date
TODAY_DATE = use.get_today()

## Code adapted from past project and references in Youtube Data API's documentation
from googleapiclient.discovery import build

YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

global youtube
youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

## Process path file names
get_df_paths = use.get_df_paths

## Save dictionary to json file
save_dict_to_json = use.save_dict_to_json

## Open dictionary from json file
open_dict_from_json = use.open_dict_from_json

## Process dfs in their final form, just in case
def process_pop_ch_dfs():
    def process(paths_dict):
        for collection_date, categories in paths_dict.items():
            for category_id, df_path in categories.items():
                df = pd.read_json(df_path)
                df_processed = fix.process_df(df, collection_date, category_id)
                df_processed.to_json(df_path, date_format="iso")
                fix.report_progress("PROCESS", df_path)

    popular_vids_path = DATA_PATH / "popular_vids"
    channel_path = DATA_PATH / "channels"
    popular_vids_paths = get_df_paths(popular_vids_path)
    channel_paths = get_df_paths(channel_path)
    process(popular_vids_paths)
    process(channel_paths)

def add_keys_to_dict_default_vals(dict_:dict, keys:list, default_value)->dict:
    key_is_in_dict = lambda k: dict_.get(k, False)
    for key in keys:
        if not key_is_in_dict(key):
            dict_[key] = default_value
    return dict_

## Get potential range from each popular_vids df
## Get channel appearances from each popular_vids df
def potential_range_channel_appearances(popular_vids_paths):
    # portential range container
    potential_range = {}

    # channel appearance container
    channel_appearances = []

    # what to do for each dataframe
    def each_df(df:pd.DataFrame, potential_range:dict, channel_appearances:list)->tuple:
        df["collection_date"] = df["collection_date"].apply(lambda cell: pd.Timestamp(cell, tz="Asia/Seoul"))
        assert len(df['collection_date'].unique()) == 1
        collection_date = df["collection_date"].unique()[0]

        assert len(df['category_id'].unique()) == 1
        category_id = df['category_id'].unique()[0]

        # calculate potential range
        df_potential_range = (df["published_at"].min().isoformat(), df["published_at"].max().isoformat())
        # store potential range
        add_keys_to_dict_default_vals(potential_range, [collection_date], {})
        add_keys_to_dict_default_vals(potential_range[collection_date], [category_id], ())
        potential_range[collection_date][category_id] = df_potential_range

        # calculate channel appearance
        df_ch = df[["channel_id", "collection_date", "category_id"]].groupby(["category_id", "collection_date"]).value_counts().unstack([0, 1])
        # store channel appearance
        channel_appearances.append(df_ch)

        return potential_range, channel_appearances
    
    # Loop over all popular_vids dfs
    for collection_date, categories in popular_vids_paths.items():
        for category_id, df_path in categories.items():
            df = pd.read_json(df_path)
            potential_range, channel_appearances = each_df(df, potential_range, channel_appearances)

    # concat channel appearances
    channel_appearances_final = pd.concat(channel_appearances, axis=1).sort_index(axis=1)
    # convert potential_range
    potential_range = pd.DataFrame(potential_range)
    # return statement
    return potential_range, channel_appearances_final

def process_imported_channel_appearance(channel_appearance_import):
    imported_columns = list(channel_appearance_import.columns)
    tupled_columns = []
    for col_str in imported_columns:
        col_str = col_str.strip("()")
        cat_id_str, col_date_str = col_str.split(",", 1)
        cat_id, col_date = int(cat_id_str), pd.Timestamp(col_date_str, tz="Asia/Seoul")
        tupled_columns.append((cat_id, col_date))
    channel_appearance_import.columns = pd.MultiIndex.from_tuples(tupled_columns, names=("category_id", "collection_date"))
    channel_appearance_import.index = channel_appearance_import.index.set_names("channel_id")
    channel_appearance_processed = channel_appearance_import.copy()
    del channel_appearance_import
    return channel_appearance_processed



def process_imported_potential_range(potential_range_import):
    imported_columns = pd.Series(potential_range_import.columns)
    if not isinstance(imported_columns[0], pd.Timestamp):
        tz_columns = imported_columns.apply(lambda cell: pd.Timestamp(cell, tz="Asia/Seoul"))
    else:
        tz_columns = imported_columns.apply(lambda cell: pd.Timestamp(cell))
    potential_range_import.columns = tz_columns
    potential_range_processed = potential_range_import.copy()
    del potential_range_import
    return potential_range_processed

# Get condensed potential range by channel id
def potential_range_by_channel(channel_appearance_df: pd.DataFrame, potential_range_df:pd.DataFrame,):
    channel_appearance_id_dates = channel_appearance_df[channel_appearance_df.notna()].stack([0, 1], future_stack=True).dropna().index
    potential_range_for_channel = {}
    for channel_id, category_id, collection_date in channel_appearance_id_dates:
        if isinstance(collection_date, pd.Timestamp):
            collection_date = collection_date.isoformat()
        add_keys_to_dict_default_vals(potential_range_for_channel, [channel_id], [])
        date_range = potential_range_df.loc[category_id, collection_date]
        potential_range_for_channel[channel_id].append(date_range)
    
    def process_date_vals(date_val):
        if isinstance(date_val, str):
            date_val = pd.Timestamp(date_val, tz='UTC')
        return date_val

    condensed_potential_range_for_channel = {}
    for channel_id, pot_ranges in potential_range_for_channel.items():
        min_val = pot_ranges[0][0]
        max_val = pot_ranges[0][1]
        for pot_range in pot_ranges:
            if pot_range[0] < min_val:
                min_val = pot_range[0]
            if pot_range[1] > max_val:
                max_val = pot_range[1]
        condensed_potential_range_for_channel[channel_id] = {
            "min_date" : process_date_vals(min_val).isoformat(),
            "max_date" : process_date_vals(max_val).isoformat()
        }
    return condensed_potential_range_for_channel

def get_all_channel_uploads_pl(channel_paths:pd.DataFrame, limit_id=0, limit_category=False):
    all_upload_pls = []
    if limit_category:
        for collection_date, categories in channel_paths.items():
            for category_id, df_path in categories.items():
                if int(category_id) == int(limit_id):
                    df = pd.read_json(df_path)
                    all_upload_pls.append(df[["channel_id", "channel_uploads_pl"]])
    else:
        for collection_date, categories in channel_paths.items():
            for category_id, df_path in categories.items():
                try:
                    df = pd.read_json(df_path)
                except:
                    print("While trying to read df, error in:", df_path, )
                all_upload_pls.append(df[["channel_id", "channel_uploads_pl"]])
    return pd.concat(all_upload_pls).drop_duplicates().reset_index(drop=True)

def channel_appearance_before_save(channel_appearance_df):
    og_columns = channel_appearance_df.columns
    og_names = og_columns.names
    og_cat_ids = og_columns.get_level_values(0)
    og_dates = og_columns.get_level_values(1)
    new_dates = pd.Series(og_dates).apply(lambda c: c.isoformat())
    new_columns = pd.MultiIndex.from_arrays([og_cat_ids, new_dates], names=og_names)
    channel_appearance_df.columns = new_columns
    return channel_appearance_df


def request_upload_pl_items(params:dict):
    request = youtube.playlistItems().list(
        part = params.get("part", None),
        maxResults = params.get("maxResults", None),
        pageToken = params.get("pageToken", None),
        playlistId = params.get("playlistId", None)
    )
    response = request.execute()
    return response

def process_playlist_response(items_list, channel_id, potential_range):
    min_date = pd.Timestamp(potential_range[channel_id]['min_date'], tz='UTC')
    max_date = pd.Timestamp(potential_range[channel_id]['max_date'], tz='UTC')
    vid_id_list = []
    for item in items_list:
        if item.get('status').get('privacyStatus') != "public":
            continue
        video_published_at = item.get('contentDetails').get('videoPublishedAt')
        if min_date < video_published_at < max_date:
            vid_id_list.append(item.get('contentDetails').get('videoId'))
    return vid_id_list

def get_and_store_upload_pl(uploads_pl_df, progress_dict_path=None, vid_items_dict_path=None):
    if progress_dict_path:
        # If progress dict path is given, retrieve
        progress_dict = open_dict_from_json(progress_dict_path)
        fix.report_progress("RETRIEVE", progress_dict_path)
    else:
        # if progress dict is empty, intialise progress dict
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
        progress_dict_path = PROGRESS_PATH / "progress_dict.json"
        save_dict_to_json(progress_dict, progress_dict_path)
        fix.report_progress("INITIALISE", progress_dict_path)

    if vid_items_dict_path:
        vid_items_dict = open_dict_from_json(vid_items_dict_path)
    else:
        vid_items_dict_path = PROGRESS_PATH / "vid_items_dict.json"
        vid_items_dict = {}
        fix.report_progress("INITIALISE", progress_dict_path)
        
        
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
            save_dict_to_json(progress_dict, progress_dict_path)
            # save vid_items_dict
            save_dict_to_json(vid_items_dict, vid_items_dict_path)
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
                save_dict_to_json(progress_dict, progress_dict_path)
                # save vid_items_dict
                save_dict_to_json(vid_items_dict, vid_items_dict_path)
                fix.report_progress("UPDATE", vid_items_dict_path)
                # But raise the error so we can see what went wrong
                raise err
            
            vid_items_dict[progress_dict["channel_id"]].extend(response["items"])
            page_token = response.get("nextPageToken")
            if page_token:
                progress_dict["params"]["pageToken"] = page_token
                # save progress_dict
                save_dict_to_json(progress_dict, progress_dict_path)
                fix.report_progress("UPDATE", progress_dict_path)
                # save vid_items_dict
                # save_dict_to_json(vid_items_dict, vid_items_dict_path)
                # fix.report_progress("UPDATE", vid_items_dict_path)

        # No longer in middle of request
        progress_dict["middle_of_playlist_request"] = False
        del progress_dict["params"]["pageToken"]
        return progress_dict, vid_items_dict


    for upload_i in range(progress_dict["upload_pl_index"], len(uploads_pl_df)):
        # Just confirming
        progress_dict["params"]["playlistId"] = uploads_pl_df.loc[upload_i, "channel_uploads_pl"]
        progress_dict["channel_id"] = uploads_pl_df.loc[upload_i, "channel_id"]

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
        if progress_dict["upload_pl_index"] + 1 < len(uploads_pl_df) - 1:
            progress_dict["upload_pl_index"] += 1
        else:
            progress_dict["upload_pl_index"] = 0
        # Update progress dict and vid items dict
        save_dict_to_json(progress_dict, progress_dict_path)
        fix.report_progress("UPDATE", progress_dict_path)
        save_dict_to_json(vid_items_dict, vid_items_dict_path)
        fix.report_progress("UPDATE", vid_items_dict_path)

        # Report progress
        fix.report_progress(f"{upload_i} FINISHED channel_id: {progress_dict['channel_id']}", "")
    return vid_items_dict

# Filter out uploads items that don't belong in the  
def filter_vid_items(vid_items_dict:dict, potential_ranges_by_channel_id:dict) -> dict:
    vid_id_dict = {}
    def filter_condition(item:dict, conditions:dict) -> bool:
        c1, c2, c3 = False, False, False
        if item.get("status").get("privacyStatus") == conditions["privacyStatus"]:
            c1 = True
        if pd.Timestamp(item.get("contentDetails").get("videoPublishedAt")) > conditions["min_date"]:
            c2 = True
        if pd.Timestamp(item.get("contentDetails").get("videoPublishedAt")) < conditions["max_date"]:
            c3 = True
        return c1 and c2 and c3
        
    for channel_id, itemlist in vid_items_dict.items():
        # If we don't have potential range information for the channel
        if not bool(potential_ranges_by_channel_id.get(channel_id)):
            fix.report_progress("SKIP", channel_id)
            continue
        conditions = {
            "privacyStatus": "public",
            "min_date": pd.Timestamp(potential_ranges_by_channel_id[channel_id]["min_date"], tz='UTC'),
            "max_date": pd.Timestamp(potential_ranges_by_channel_id[channel_id]["max_date"], tz='UTC'),
        }
        vid_id_dict[channel_id] = []
        ii = -1
        i = 0
        first_time = True
        for item in itemlist:
            if i % 50 == 0 and first_time:
                vid_id_dict[channel_id].append([])
                ii += 1
                first_time = False
            elif i % 50 != 0:
                first_time = True
            if filter_condition(item, conditions):
                vid_id_dict[channel_id][ii].append(item.get("contentDetails").get("videoId"))
                i += 1
    return vid_id_dict
## Tested.
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
## Tested.
def more_vids_request(params_list, progress_path=DATA_PATH / "progress/more_video_progress.txt", more_vids_path=DATA_PATH / "more_vids/2025-03-19_0.json"):
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
                pd.concat(df_list).reset_index().to_json(more_vids_path)
            return
        else:
            df = process_response(response)
            df_list.append(df)
    return pd.concat(df_list).reset_index()

def main():
    popular_vids_paths = use.get_df_paths(DATA_PATH / "popular_vids")
    channel_paths = use.get_df_paths(DATA_PATH / "channels")


    pipeline = {
        "pre_process": False,
        "tables": False,
        "uploads_prep": False,
        "uploads_request":False,
        "vids_prep": False,
        "vids_request":False,
        "post_process": True,
        "category_id": 0,
        "collection_date":True,
    }
    # Custom collection date or use today's date
    if pipeline["collection_date"]:
        COLLECTION_DATE = "2025-03-19"
    else:
        COLLECTION_DATE = TODAY_DATE


    # The pre-processing that happens in `fix`
    if pipeline["pre_process"]:
        process_pop_ch_dfs()
    
    # Constructing Tables
    if pipeline["tables"]:
        # get potential ranges for data collection date and category id
        # get which channels appeared where and when
        potential_range_df, channel_appearance_df = potential_range_channel_appearances(popular_vids_paths)
        # store potential_range
        potential_range_df.columns = pd.Series(potential_range_df.columns).apply(lambda col: col.isoformat())
        potential_range_df.to_json(PROGRESS_PATH / "potential_range_df.json")
        use.report_progress("SAVE", PROGRESS_PATH / "potential_range_df.json")
        # store channel_appearance
        channel_appearance_df = channel_appearance_before_save(channel_appearance_df)
        channel_appearance_df.to_json(PROGRESS_PATH / "channel_appearance_df.json")
        use.report_progress("SAVE", PROGRESS_PATH / "channel_appearance_df.json")
    else:
        potential_range_df = pd.read_json(PROGRESS_PATH / "potential_range_df.json")
        potential_range_df = process_imported_potential_range(potential_range_df)
        use.report_progress("LOAD", PROGRESS_PATH / "potential_range_df.json")
        channel_appearance_df = pd.read_json(PROGRESS_PATH / "channel_appearance_df.json")
        channel_appearance_df = process_imported_channel_appearance(channel_appearance_df)
        use.report_progress("LOAD", PROGRESS_PATH / "channel_appearance_df.json")

    # Prepping for uploads request
    if pipeline["uploads_prep"]:
        # get condensed potential ranges by channel id
        # returns dictionary
        potential_ranges_by_channel_id = potential_range_by_channel(channel_appearance_df, potential_range_df)
        # store potential_ranges_by_channel_id
        save_dict_to_json(potential_ranges_by_channel_id, PROGRESS_PATH / "potential_range_by_channel_id.json")
        use.report_progress("SAVE", PROGRESS_PATH / "potential_range_by_channel_id.json")

        uploads_pl_df = get_all_channel_uploads_pl(channel_paths, limit_id=pipeline["category_id"], limit_category=True)
        print(f"Length of uploads_pl_df: {len(uploads_pl_df)}")
        uploads_pl_df.to_json(PROGRESS_PATH / "uploads_pl_df.json")
        use.report_progress("SAVE", PROGRESS_PATH / "uploads_pl_df.json")
    else:
        potential_ranges_by_channel_id = open_dict_from_json(PROGRESS_PATH / "potential_range_by_channel_id.json")
        use.report_progress("LOAD", PROGRESS_PATH / "potential_range_by_channel_id.json")
        uploads_pl_df = pd.read_json(PROGRESS_PATH / "uploads_pl_df.json")
        use.report_progress("LOAD", PROGRESS_PATH / "uploads_pl_df.json")

    progress_dict_path = PROGRESS_PATH / "progress_dict.json"
    vid_items_dict_path = PROGRESS_PATH / "vid_items_dict.json"

    # Uploads Request
    if pipeline["uploads_request"]:
        get_and_store_upload_pl(uploads_pl_df, progress_dict_path, vid_items_dict_path)

    vid_items_dict = open_dict_from_json(vid_items_dict_path)
    use.report_progress("LOAD", vid_items_dict_path)

    # Video Ids Request Prep
    if pipeline["vids_prep"]:
        vid_id_dict = filter_vid_items(vid_items_dict, potential_ranges_by_channel_id)
        save_dict_to_json(vid_id_dict, PROGRESS_PATH / "vid_id_dict.json")
        use.report_progress("SAVE", PROGRESS_PATH / "vid_id_dict.json")
        params_list = convert_vid_id_dict_to_params_list(vid_id_dict)
        save_dict_to_json(params_list, DATA_PATH / "progress/params_list.json")
        use.report_progress("SAVE", DATA_PATH / "progress/params_list.json")
    else:
        vid_id_dict = open_dict_from_json(PROGRESS_PATH / "vid_id_dict.json")
        use.report_progress("LOAD", PROGRESS_PATH / "vid_id_dict.json")
        params_list = open_dict_from_json(DATA_PATH / "progress/params_list.json")
        use.report_progress("LOAD", DATA_PATH / "progress/params_list.json")
    cat_id = pipeline.get("category_id")
    if pipeline["vids_request"]:
        more_vids_df = more_vids_request(params_list)
        more_vids_df.to_json(DATA_PATH / f"more_vids/{COLLECTION_DATE}_{cat_id}.json")
        use.report_progress("SAVE", DATA_PATH / f"more_vids/{COLLECTION_DATE}_{cat_id}.json")
    else:
        more_vids_df = pd.read_json(DATA_PATH / f"more_vids/{COLLECTION_DATE}_{cat_id}.json")
        use.report_progress("LOAD", DATA_PATH / f"more_vids/{COLLECTION_DATE}_{cat_id}.json")
    
    if pipeline["post_process"]:
        more_vids_df["collection_date"] = COLLECTION_DATE
        more_vids_df["category_id"] = 0
        more_vids_df.drop("index", axis=1)
        more_vids_df.to_json(DATA_PATH / f"more_vids/{COLLECTION_DATE}_{cat_id}.json")
        use.report_progress("UPDATE", DATA_PATH / f"more_vids/{COLLECTION_DATE}_{cat_id}.json")


if __name__ == "__main__":
    main()
