from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import requests

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# url = "/Users/allen/sources/pallas/luz-py/data/cats.jpg"
# url = "/Users/allen/Desktop/barty.jpg"
url = "/Users/allen/Desktop/barty_side.jpg"
image = Image.open(url).convert("RGB")

image_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Large-hf")

# prepare image for the model
inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# interpolate to original size and visualize the prediction
post_processed_output = image_processor.post_process_depth_estimation(
    outputs,
    target_sizes=[(image.height, image.width)],
)

predicted_depth = post_processed_output[0]["predicted_depth"]
depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
depth = depth.detach().cpu().numpy() * 255
depth = Image.fromarray(depth.astype("uint8"))

# Convert depth image to RGB mode for better visualization
depth = depth.convert("L")  # Convert to grayscale
# depth.save("cats_depth.png")
# depth.save("barty_depth.png")
depth.save("barty_side_depth.png")
depth.show(title="Depth Estimation")  # Show the depth map
