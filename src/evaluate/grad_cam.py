
import torch
import torch.nn as nn
import torchvision
import torchvision.models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pytorch_grad_cam

from torchinfo import summary

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import sys

sys.path.insert(0, './src')
from models.locnet import UNetWithResnet50Encoder

#from ..models.locnet import UNetWithResnet50Encoder
from torchinfo import summary
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget



## Two functions to display the image and the heatmap for both LOCNet and ResNet50

image_path = "./Image/dog/n02109525_37.JPEG"  # set to one image
model_path = "./pre_trained/loc_model_best.pth.tar"  # set to the LocNet model path

model_path2 = "./pre_trained/loc_model_best.pth"  # test 

class WrapperModel(torch.nn.Module):
    def __init__(self, model): 
        super(WrapperModel, self).__init__()
        self.model = model
        self.input_block = model.input_block
        self.down_blocks = model.down_blocks
        
    def forward(self, x):
        # In out case conv_block_1 or conv_block_2
        #main_output, cls_pred , output_feauture = self.model(x, with_output_feature_map=True)
        outputs = self.model(x)
        return outputs[1]
    

# Temp just for loading the dictionary

# def LoacImageToTensor(image_path):
#     image = Image.open(image_path)
#     image = touchvision.transforms.ToTensor()(image)
#     image = touchvision.transforms.Resize((224, 224))(image)
#     image = touchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
#     image = image.unsqueeze(0)
#     return image
def NormalProcess(image_path):
    image = np.array(Image.open(image_path))
    rgb_img = np.float32(image) / 255
    input_tensor = preprocess_image(rgb_img,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    print(input_tensor.shape)
    return input_tensor

def TestOnOneImage(image_path):
    #image = np.array(Image.open(image_path))
    # plt.imshow(image)
    # plt.show()
    # rgb_img = np.float32(image) / 255
    trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.CenterCrop((224, 224))
        ])
    
    # img = trans(image)
    
    img = Image.open(image_path)
    img = trans(img)
    # input_tensor = preprocess_image(img,
    #                             mean=[0.485, 0.456, 0.406],
    #                             std=[0.229, 0.224, 0.225])
    img = torchvision.transforms.ToTensor()(img)
    img= img.reshape(1, 3, 224, 224)
    print(img.shape)
    return img

def TestOnOriginalImage(image_path):
    image = np.array(Image.open(image_path))
    return image/ 255


def LoadLocNetModel():

    checkpoint = torch.load(model_path,map_location='cpu')

    model = UNetWithResnet50Encoder()
    
    #model = torchvision.models.resnet50(pretrained= True)  
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']

        # Sometimes the state_dict keys have a prefix like 'module.' if they were saved from a DataParallel module
        # If so, you need to strip this prefix
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Load the stripped state_dict
        model.load_state_dict(state_dict)
    else:
        # Directly try loading the checkpoint (not recommended if the above keys are present)
        model.load_state_dict(checkpoint)
    #summary(model)
    
    #print(model.down_blocks[0])
    # first_layer_name = next(model.named_children())[0]
    # print("First layer name:", first_layer_name)
    # for name, layer in model.named_children():
    #     print(name, layer)
    
    # layer = model.up_blocks[-1]
    # print(layer)
    return model

# Current set to just display one, change the file path to the image you want to display
def ResNet50Cam():
    model = torchvision.models.resnet50(pretrained= True)  
    target_layer = [model.layer4[-1] ]
    img = TestOnOriginalImage(image_path)
    input_tensor_temp = NormalProcess(image_path)
    target = [ClassifierOutputTarget(2)] ## index of the class, seems 2 for the sample
    #target = None
    with GradCAM(model=model, target_layers=target_layer) as cam:
        grayscale_cams = cam(input_tensor=input_tensor_temp, targets=target)
        cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
    cam = np.uint8(255*grayscale_cams[0, :])
    cam = cv2.merge([cam, cam, cam])
    images = np.hstack((np.uint8(255*img) , cam_image))
    Image.fromarray(images)
    plt.imshow(images)
    plt.show()

def visualize_cam(heatmap):
    plt.imshow(heatmap, cmap='jet', interpolation='bilinear')
    plt.colorbar()
    plt.show()
    
# check the target layer set to before the bridge of U-Net
def LocNetCam():
    #model = LoadLocNetModel()
    #model.eval()
    model = WrapperModel(LoadLocNetModel())
    # target_layer = [model] # alreayd wrapped
    #model = WrapperModel(LoadLocNetModel())
    
    target_layer = [model.down_blocks[0] ]
    img = TestOnOriginalImage(image_path)
    input_tensor_temp = TestOnOneImage(image_path)
    #input_temp2 = NormalProcess(image_path)
    target = [ClassifierOutputTarget(2)] 

    # with GradCAM(model=model, target_layers=target_layer) as cam:
    #     grayscale_cams = cam(input_tensor=input_tensor_temp, targets=target)
    #     cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)

    
    # cam = np.uint8(255*grayscale_cams[0, :])
    # cam = cv2.merge([cam, cam, cam])
    # images = np.hstack((np.uint8(255*img) , cam_image))
    cam = GradCAM(model=model, target_layers=target_layer)
    grayscale_cams = cam(input_tensor=input_tensor_temp , targets=target)
    print("Grayscale CAMs:", grayscale_cams.shape)
    heatmap = grayscale_cams[0]  # Simplify to (224, 224)
    visualize_cam(heatmap)
    # Image.fromarray(images)
    # plt.imshow(images)
    # plt.show()
    
#ResNet50Cam()
# LocNetCam()
# LocNetCam()
# LoadLocNetModel()
LocNetCam()
