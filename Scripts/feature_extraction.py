from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel
from PIL import Image
import cv2
import torch
import numpy as np
from pathlib import Path
import useful as use


def get_image_from_dir(image_dir):
    image_paths = list(image_dir.glob("*.jpg"))
    image_dict = {img_path.stem: img_path for img_path in image_paths}
    return image_dict

def set_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def load_processor(processor_name:str="openai/clip-vit-base-patch32")->CLIPProcessor:
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return processor

def load_video_processor(hf_token:str, processor_name:str = "microsoft/xclip-base-patch32") -> AutoProcessor:
    processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32", token=hf_token)
    return processor

def load_model(device, model_name="openai/clip-vit-base-patch32")->CLIPModel:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()
    return model

def load_video_model(device, hf_token:str, model_name:str = "microsoft/xclip-base-patch32") -> AutoModel:
    model = AutoModel.from_pretrained("microsoft/xclip-base-patch32", token=hf_token)
    model.eval()
    return model

def feature_from_image(image_path:Path, processor:CLIPProcessor, model:CLIPModel, device:torch.device)->np.ndarray:
    """
    Extracts semantic features from thumbnail image using the CLIP model,
    following the Hugging Face documentation example.

    Args:
        image_path (str): The path to the thumbnail image file.
        processor (CLIPProcessor): The pre-trained CLIP processor.
        model (CLIPModel): The pre-trained CLIP model.

    Returns:
        features: A numpy array representing the semantic feature vector of the video.
                       Returns None if there's an issue processing the video.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    features:np.ndarray = image_features.cpu().numpy().squeeze()
    return features
    
def feature_from_text(text: str, processor, model, device)->np.ndarray:
    """
    Extracts semantic features from title text using the CLIP model,
    following the Hugging Face documentation example.

    Args:
        text (str): The title text.
        processor (CLIPProcessor): The pre-trained CLIP processor.
        model (CLIPModel): The pre-trained CLIP model.

    Returns:
        features: A numpy array representing the semantic feature vector of the video.
                       Returns None if there's an issue processing the video.
    """
    # Check token length before encoding
    tokens = processor.tokenizer(text, return_tensors="pt")
    token_count = tokens["input_ids"].shape[-1]
    
    if token_count > 77:
        print(f"[Warning] Truncating text: '{text[:50]}...' ({token_count} tokens)")

    # Tokenize with truncation enabled
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,  # Ensure no crash
        max_length=77     # Enforce CLIP's hard limit
    ).to(device)

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

    features:np.ndarray = text_features.cpu().numpy().squeeze()
    return features

def feature_from_video(video_path:Path, processor: AutoProcessor, model: AutoModel, device:torch.device, num_of_frames:int=16) -> np.ndarray|None:
    """
    Extracts semantic features from a video using the XCLIP model,
    following the Hugging Face documentation example.

    Args:
        video_path (str): The path to the video file.
        processor (AutoProcessor): The pre-trained XCLIP processor.
        model (AutoModel): The pre-trained XCLIP model.

    Returns:
        video_features: A numpy array representing the semantic feature vector of the video.
                       Returns None if there's an issue processing the video.
    """
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file at {video_path}")
            return None

        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_duration_seconds = frame_count / fps if fps > 0 else 0

        # Define the number of frames to sample
        num_frames_to_sample = num_of_frames
        if frame_count < num_frames_to_sample:
            indices = np.linspace(0, frame_count - 1, frame_count, dtype=int)
        else:
            indices = np.linspace(0, frame_count - 1, num_frames_to_sample, dtype=int)

        sampled_frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                sampled_frames.append(frame_rgb)
            else:
                print(f"Warning: Could not read frame at index {i}")

        cap.release()

        if not sampled_frames:
            print("Error: No frames were successfully sampled from the video.")
            return None
        
        # Process the sampled frames
        inputs = processor(videos=sampled_frames, return_tensors="pt")
        print("Shape of inputs['pixel_values']:", inputs['pixel_values'].shape)

    
        # Get the video features
        with torch.no_grad():
            video_features = model.get_video_features(**inputs)


        # The output of get_video_features is likely a tensor of shape (batch_size, feature_dimension)
        # Since we are processing one video at a time, batch_size will be 1.
        # We want to extract the feature vector and convert it to a NumPy array.
        return video_features.squeeze(0).cpu().numpy()

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    

def feature_from_video_with_l1(video_path:Path, processor: AutoProcessor, model: AutoModel, device:torch.device, sample_percentage) -> np.ndarray|None:
    """
    Extracts semantic features from a video using the XCLIP model,
    following the Hugging Face documentation example.

    Args:
        video_path (str): The path to the video file.
        processor (AutoProcessor): The pre-trained XCLIP processor.
        model (AutoModel): The pre-trained XCLIP model.

    Returns:
        video_features: A numpy array representing the semantic feature vector of the video.
                       Returns None if there's an issue processing the video.
    """
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file at {video_path}")
            return None, None  
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            print(f"Warning: Video {video_path} has no frames")
            cap.release()
            return None, None
        # Define number of frames to sample for l1-norms
        l1_num_samples = int(total_frames * sample_percentage)
        if l1_num_samples == 0 and total_frames > 0:
            l1_num_samples = 1 # ensure at least one frame is sampled
        # Generate evenly spaced l1 frame indices
        l1_frame_indices = np.linspace(0, total_frames - 1, l1_num_samples, dtype=int)
        
        # Define the number of frames to sample for XCLIP
        num_frames_to_sample = 8
        if total_frames < num_frames_to_sample:
            X_CLIP_indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
        else:
            X_CLIP_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)
        
        # Sample frames for l1 norms
        l1_sampled_frames = []
        for l1_i in l1_frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, l1_i)
            ret, frame = cap.read()
            if ret:
                frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                l1_sampled_frames.append(frame_grey)
            else:
                print(f"Warning: Could not read frame at index {l1_i} from {video_path}")
        
        # Sample frames for xclip
        xclip_sampled_frames = []
        for x_i in X_CLIP_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, x_i)
            ret, frame = cap.read()
            if ret:
                # Convert frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                xclip_sampled_frames.append(frame_rgb)
            else:
                print(f"Warning: Could not read frame at index {x_i} from {video_path}")

        cap.release()

        if not (l1_sampled_frames and xclip_sampled_frames):
            print("Error: No frames were successfully sampled from the video.")
            return None, None
        
        # Calculate L1-norms between consecutive sampled frames
        l1_norms_sampled = []
        if len(l1_sampled_frames) > 1:
            for i in range(len(l1_sampled_frames) - 1):
                l1_norm = np.sum(np.abs(l1_sampled_frames[i+1] - l1_sampled_frames[i]))
                l1_norms_sampled.append(l1_norm)
            
        # Process the sampled frames
        inputs = processor(videos=xclip_sampled_frames, return_tensors="pt")
        print("Shape of inputs['pixel_values']:", inputs['pixel_values'].shape)

        # Get the video features
        with torch.no_grad():
            xclip_features = model.get_video_features(**inputs)


        # The output of get_video_features is likely a tensor of shape (batch_size, feature_dimension)
        # Since we are processing one video at a time, batch_size will be 1.
        # We want to extract the feature vector and convert it to a NumPy array.
        return xclip_features.squeeze(0).cpu().numpy(), l1_norms_sampled

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

    
def main():
    #### Private config
    priv = use.get_priv()
    DATA_PATH = Path(priv["DATA_PATH"])

    # Where to save the image feature dataset
    output_dir = DATA_PATH / "dataset"
    # Where popular video thumbnails are located
    pop_dir = DATA_PATH / "thumbnails/popular"
    # Get all the image paths
    pop_thumbs = get_image_from_dir(pop_dir)
    ### Prepare device, processor, model
    device = set_device()
    processor = load_processor()
    model = load_model(device)

    # Iterate through all popular video thumbnails and gather output feature vectors
    pop_vectors = {}
    for pop_vid_id, pop_vid_path in pop_thumbs.items():
        pop_vectors[pop_vid_id] = feature_from_image(pop_vid_path, 
                                                     device=device,
                                                     model=model,
                                                     processor=processor)
    # Save popular image features
    pop_output_path = output_dir / "popular_image_features.npz"
    np.savez(pop_output_path, **pop_vectors)

    # Delete these variables, as they're not used any more
    del pop_vectors, pop_thumbs, pop_dir, pop_output_path

    # Where popular video thumbnails are located
    unpop_dir = DATA_PATH / "thumbnails/unpopular"
    # Get all the image paths
    unpop_thumbs = get_image_from_dir(unpop_dir)

    # Iterate through all unpopular video thumbnails and gather output feature vectors
    unpop_vectors = {}
    for unpop_vid_id, unpop_vid_path in unpop_thumbs.items():
        unpop_vectors[unpop_vid_id] = feature_from_image(unpop_vid_path,
                                                         device=device,
                                                         model=model,
                                                         processor=processor)
        
    # Save unpopular image features
    unpop_output_path = output_dir / "unpopular_image_features.npz"
    np.savez(unpop_output_path, **unpop_vectors)

    # Delete these variables, as they're not used any more
    

if __name__ == "__main__":
    main()