from yt_dlp import YoutubeDL
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from IPython.display import HTML
import base64

def download_video_with_id(video_id:str, output_path:Path|str) -> Path|None:
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
        except Exception as e:
            print(f"Failed to process {video_id}: {e}")
            return None
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