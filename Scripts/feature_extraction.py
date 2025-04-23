from transformers import CLIPProcessor, CLIPModel
from PIL import Image
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

def load_model(device, model_name="openai/clip-vit-base-patch32")->CLIPModel:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()
    return model

def feature_from_image(image_path:Path, processor:CLIPProcessor, model:CLIPModel, device:torch.device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    features = image_features.cpu().numpy().squeeze()
    return features
    
def feature_from_text(text: str, processor, model, device):
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

    features = text_features.cpu().numpy().squeeze()
    return features

    
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