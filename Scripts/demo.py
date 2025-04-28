from yt_dlp import YoutubeDL
from pathlib import Path
import urllib
from urllib.parse import urlparse, parse_qs
import pandas as pd
import useful as use

def download_video_with_id(video_id:str, output_path:Path|str, cookie_path=None) -> Path|None:
    """
    Downloads YouTube video from Youtube video ID.
    Args:
        video_id(str): The YouTube video ID
        output_path(Path|str): Where the resulting video should be saved
    Returns:
        video_filepath(Path|None): If there was a successful download, or if it had already been downloaded, returns the path to the video file.
        Otherwise returns None.
    """
    # If the video already exists, then move on
    output_path = Path(output_path)
    video_filepath = output_path / f"{video_id}.mp4"
    if video_filepath.exists():
        print("Video already exists. Exiting download")
        return video_filepath
    # options
    if cookie_path is not None:
        base_opts = {
            "format": "bestvideo[height<=360][ext=mp4][vcodec^=?avc]",  # Download best quality video up to 360p in MP4 format (no audio)
            "quiet": False,
            "cookiefile": cookie_path,
            }
    else:
        base_opts = {
            "format": "bestvideo[height<=360][ext=mp4][vcodec^=?avc]",  # Download best quality video up to 360p in MP4 format (no audio)
            "quiet": False,
            } 
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = base_opts.copy()
    ydl_opts["outtmpl"] = str(video_filepath)
    with YoutubeDL(ydl_opts) as ydl:
        try:
            print(f"Downloading {video_id}...")
            ydl.download([url])
            if video_filepath.exists():
                print(f"Successfully downloaded {video_id}")
            else:
                print(f"Failed to download {video_id}")
                return None
        except urllib.error.HTTPError as e:
            if "private video" in str(e).lower():
                print("Error: API access failed, due to video being private")
            elif "HTTP Error 403" in str(e) or "HTTP Error 401" in str(e) or "Unable to extract info" in str(e) and "needs cookie" in str(e).lower():
                print("Error: API access failed, likely due to missing or invalid cookies.")
                raise e
                # You can add your logic here to handle the missing cookie situation,
                # such as prompting the user to provide cookies or trying to refresh them.
            else:
                print(f"An unexpected download error occurred: {e}")
                # Handle other potential download errors
        except Exception as e:
            print(f"A general error occurred: {e}")
            if "bot" in str(e):
                raise e
            elif "Video unavailable" in str(e):
                raise e
    return video_filepath

def extract_video_id(url:str)->str|None:
    """
    Extracts the YouTube video ID from a YouTube video URL.

    Args:
        url(str): The YouTube video URL (string).

    Returns:
        video_id(str): The YouTube video ID (string) or None if it cannot be found.
    """
    try:
        parsed_url = urlparse(url)
        if parsed_url.netloc in ['www.youtube.com', 'm.youtube.com', 'youtube.com', 'youtu.be']:
            if parsed_url.path == '/watch':
                query_params = parse_qs(parsed_url.query)
                video_id = query_params.get('v')
                if video_id:
                    return video_id[0]
            elif parsed_url.netloc == 'youtu.be':
                video_id = parsed_url.path[1:]
                return video_id
    except:
        return None
    return None
def prepare_for_youtube_api_request(youtube_api_key):
    from googleapiclient.discovery import build
    # Set up Youtube Data API access
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=youtube_api_key)
    return youtube

def get_tabular_video_info(video_id:str, youtube) -> pd.DataFrame:
    def construct_request(video_id):
        params = {
            "id": video_id,
            "part": "snippet,statistics,contentDetails,paidProductPlacementDetails,liveStreamingDetails",
            }
        return params
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
    
    # use video id to construct request
    request_params = construct_request(video_id)
    # send request
    response = send_request(request_params)
    # process response
    df = process_response(response)

    return df
def cleanse_tabular_video_info(raw_df:pd.DataFrame, drop_vlc=False)-> pd.DataFrame:
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
    def do_drop_vlc(df:pd.DataFrame):
        result = df.drop(["views", "likes", "private-likes", "comments", "private-comments"], axis=1)
        return result
    # cleanse!
    cleansed_df = cleanse_vid_df(raw_df, cleansing=cleansing)
    # If asked to do so, drop Views, Likes, Comments
    if drop_vlc:
        cleansed_df = do_drop_vlc(cleansed_df)

    return cleansed_df
def get_title(raw_df:pd.DataFrame)->str:
    title = raw_df["title"][0]
    return title

def get_thumbnail_image(video_id:str, thumbnail_url:str, save_folder:Path) -> Path:
    def switch_to_hq(url_str):
        split_url = url_str.split("/")
        split_url[-1] = "hqdefault.jpg"
        rejoined_url = "/".join(split_url)
        return rejoined_url
    # make sure to download a relatively high quality image
    if thumbnail_url.split("/")[-1] != "hqdefault.jpg":
        thumbnail_url = switch_to_hq(thumbnail_url)

    def download_image(video_id:str, image_url:str, save_folder:Path):
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

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {image_url}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while processing {image_url}: {e}")

        return filepath
    
    # download
    thumbnail_path = download_image(video_id, thumbnail_url, save_folder)
    return thumbnail_path