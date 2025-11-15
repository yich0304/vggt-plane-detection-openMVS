import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load and preprocess example images (replace with your own image paths)
image_names = ["C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010559.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010602.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010603.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010605.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010608.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010612.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010615.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010617.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010622.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010624.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010626.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010628.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010633.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010637.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010638.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010641.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010643.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010645.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010649.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010653.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010655.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010659.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010702.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010705.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010718.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010727.jpg",
               "C:/Users/zouyicheng/Desktop/desai/vggt/images/20250623_010731.jpg"
               ]  
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)