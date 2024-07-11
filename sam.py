import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor


# SAM - Segment Anything Model
device = "cuda" if torch.cuda.is_available() else "cpu"
sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

# image - to be replaced
img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

# mask - to be replaced
mask_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
segmentation_map = Image.open(requests.get(mask_url, stream=True).raw).convert("1")

input_points = [[[450, 600]]]  # 2D location of a window in the image

inputs = sam_processor(raw_image, input_points=input_points, segmentation_maps=segmentation_map, return_tensors="pt").to(device)
with torch.no_grad():
    sam_outputs = sam_model(**inputs)

masks = sam_processor.image_processor.post_process_masks(
    sam_outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)

# scores = sam_outputs.iou_scores




