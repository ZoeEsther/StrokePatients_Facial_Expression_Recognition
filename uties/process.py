import numpy as np
import PIL.Image as image
from torchvision import transforms

def input_porocess(imgRGB,imgSize,device):
    transformer = transforms.Compose([
        # transforms.RandomRotation(60),
        # transforms.Grayscale(num_output_channels=3),
        transforms.Resize((imgSize,imgSize)),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.2,0.2,0.2])
    ])
    try:
        img0 = image.fromarray(np.uint8(imgRGB))
    except:
        return False
    img_tensor = transformer(img0)

    NP_img = np.array(img_tensor)

    train_mean = np.mean(NP_img, axis=(1, 2))
    train_std = np.std(NP_img, axis=(1, 2))
    img = transforms.Normalize(mean=train_mean, std=train_std)(img_tensor)
    img = img.view(-1, 3, imgSize, imgSize).to(device)
    return img, img_tensor