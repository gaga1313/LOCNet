
import sys
import os
sys.path.insert(0, './src')
from models import *
from models.autoencoder import AutoEncoder

from torchinfo import summary
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch
import torch.nn as nn
import torchvision
import torchvision.models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pytorch_grad_cam


class TestBaseModel(nn.Module):
        
    def __init__(self, ckpt_path='./pre_trained/baseline.pth.tar'):
        super().__init__()
        self.model = AutoEncoder()
        self.model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu') )['state_dict'])
        self.model.eval()

    def forward(self, x):
        x = self.model.input_block(x)
        x = self.model.input_pool(x)
        for i, block in enumerate(self.model.down_blocks, 2):
            x = block(x)
            if i == (AutoEncoder.DEPTH - 1):
                continue

        cls_pred = self.model.avg_pool(x)
        cls_pred = torch.flatten(cls_pred, start_dim=1)
        cls_pred = self.model.dropout(cls_pred)
        cls_pred = self.model.classifier(cls_pred)
        return cls_pred
    
class TestBestModel(nn.Module):
        
    def __init__(self, ckpt_path='./pre_trained/best.pth.tar'):
        super().__init__()
        self.model = AutoEncoder()
        self.model.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu') )['state_dict'])
        self.model.eval()

    def forward(self, x):
        x = self.model.input_block(x)
        x = self.model.input_pool(x)
        for i, block in enumerate(self.model.down_blocks, 2):
            x = block(x)
            if i == (AutoEncoder.DEPTH - 1):
                continue

        cls_pred = self.model.avg_pool(x)
        cls_pred = torch.flatten(cls_pred, start_dim=1)
        cls_pred = self.model.dropout(cls_pred)
        cls_pred = self.model.classifier(cls_pred)
        return cls_pred
    
    
def GetBaselineModel():
    model = TestBaseModel()
    model.eval()
    return model

def GetBestModel():
    model = TestBestModel()
    model.eval()
    return model


#image_path = "./Image/dog/n02109525_37.JPEG"
#image_path = "./Image/session-1/0068_ske_dnn_0_bear_00_bear-0053-sketch-10.png"  # label8
#image_path = "./Image/session-1/0145_ske_dnn_0_bicycle_00_bicycle-0094-sketch-47.png"  # label3
#image_path = "./Image/session-1/0635_ske_dnn_0_keyboard_00_keyboard-0059-sketch-15.png"  # label1
#image_path = "./Image/session-1/0550_ske_dnn_0_dog_00_dog-5354-sketch-25.png"  # label15



#label = 15
label = None
image_path = None
saved_path = "./saved_Image/"

def TestOnOneImage(image_path):
    trans = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.CenterCrop((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
        ])
    

    img = Image.open(image_path)
    # print("One one:", np.array(img).shape)
    # if len(np.array(img).shape) == 2:
    #     img = np.stack([img] *3, axis=-1)
    # print("One one after:", img.shape)

    #print("One:", img.shape)
    if len(np.array(img).shape) == 2:
        img = np.stack([img] *3, axis=-1)

    img = torchvision.transforms.ToTensor()(img)
    # if len(np.array(img).shape) == 3 and np.array(img).shape[0] == 1:
    #     img = img.reshape(224, 224)
    #     img = np.stack([img] *3, axis=-1)
    # img= img.reshape(1, 3, 224, 224)
    # img = Image.fromarray(img)
    img = trans(img)
 #   img = torchvision.transforms.ToTensor()(img)
    img= img.reshape(1, 3, 224, 224)

    #print("One:", img.shape)
    return img



def TestOnOriginalImage(image_path):
    trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.CenterCrop((224, 224)),
        ])
    image = np.array(Image.open(image_path))
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    image = torchvision.transforms.ToTensor()(image)
    print("Original:", image.shape)
    image = trans(image)
    image = np.array(image)
    image = np.transpose(image, (1, 2, 0))

    return image

def visualize_cam(heatmap):
    plt.imshow(heatmap, cmap='jet', interpolation='bilinear')
    plt.colorbar()
    plt.show()
    
def NormalProcess(image_path):
    trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.CenterCrop((224, 224))
        ])
    
    image = np.array(Image.open(image_path))
    rgb_img = np.float32(image) / 255

    print("RGB", rgb_img.shape)
    # check 
    if len(rgb_img.shape) == 2:
        rgb_img = np.stack([rgb_img] * 3, axis=-1)
    print("RGB after", rgb_img.shape)
    rgb_img = torchvision.transforms.ToTensor()(rgb_img)
    rgb_img = trans(rgb_img)
    rgb_img = np.array(rgb_img)
    rgb_img = np.transpose(rgb_img, (1, 2, 0))
    input_tensor = preprocess_image(rgb_img,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    print("Tensor", input_tensor.shape)
    return input_tensor



def CamDisplayBase(filename= None,outlayer = False):
    model = GetBaselineModel()
    #print(model)
    target_layer = [model.model.down_blocks[0][-1]]
    if outlayer:
        target_layer = [model.model.down_blocks[3][-1]]
    img = TestOnOriginalImage(image_path+filename)
    input_tensor_temp = TestOnOneImage(image_path+filename)
    #input_temp2 = NormalProcess(image_path)
    target = [ClassifierOutputTarget(label)] 
    
    with GradCAM(model=model, target_layers=target_layer) as cam:
        # Generate the Grad-CAM heatmap
        grayscale_cams = cam(input_tensor=input_tensor_temp, targets=target)
        # Resize heatmap to match the original image size
        heatmap_resized = cv2.resize(grayscale_cams[0, :], (img.shape[1], img.shape[0]))
        # Overlay heatmap on original image
        cam_image = show_cam_on_image(img , heatmap_resized, use_rgb=True)

    # Convert the cam_image for visualization
    #cam_image = np.uint8(255 * cam_image)
    
    # cam = np.uint8(255*grayscale_cams[0, :])
    # cam = cv2.merge([cam, cam, cam])
    # images = np.hstack((np.uint8(255*img) , cam_image))
    # images = np.hstack( cam_image)
    # Image.fromarray(images)
    # Display the image
    # plt.imshow(images)
    # plt.axis('off')
    # plt.show()
    result_image = Image.fromarray(cam_image)
    return result_image
    
def CamDisplayBest(filename= None,outlayer = False):
    model = GetBestModel()
    #print(model)
    target_layer = [model.model.down_blocks[0][-1]]
    if outlayer: 
        target_layer = [model.model.down_blocks[3][-1]]
    img = TestOnOriginalImage(image_path+filename)
    input_tensor_temp = TestOnOneImage(image_path+filename)
    #input_temp2 = NormalProcess(image_path)
    target = [ClassifierOutputTarget(label)] 
    
    with GradCAM(model=model, target_layers=target_layer) as cam:
        # Generate the Grad-CAM heatmap
        grayscale_cams = cam(input_tensor=input_tensor_temp, targets=target)
        # Resize heatmap to match the original image size
        heatmap_resized = cv2.resize(grayscale_cams[0, :], (img.shape[1], img.shape[0]))
        # Overlay heatmap on original image
        cam_image = show_cam_on_image(img , heatmap_resized, use_rgb=True)
    
    # cam = np.uint8(255*grayscale_cams[0, :])
    # cam = cv2.merge([cam, cam, cam])
    #images = np.hstack((np.uint8(255*img) , cam_image))
    result_image = Image.fromarray(cam_image)
    
    # Display the image
    # plt.imshow(images)
    # plt.axis('off')
    # plt.show()
    return result_image
    
def ResNet50Cam(filename= None, outlayer = False):
    model = torchvision.models.resnet50(pretrained= True)  
    #print(model)
    print(label)
    target_layer = [model.layer1[-1] ]
    if outlayer:
        target_layer = [model.layer4[-1] ]
    img = TestOnOriginalImage(image_path+filename)
    input_tensor_temp = NormalProcess(image_path+filename)
    target = [ClassifierOutputTarget(label)] ## index of the class, seems 2 for the sample
    #target = None
    with GradCAM(model=model, target_layers=target_layer) as cam:
        grayscale_cams = cam(input_tensor=input_tensor_temp, targets=target)
        cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
    cam = np.uint8(255*grayscale_cams[0, :])
    cam = cv2.merge([cam, cam, cam])
    #images = np.hstack((np.uint8(255*img) , cam_image))
    result_image = Image.fromarray(cam_image)
    #plt.title(" graph")
    # plt.imshow(images)
    # plt.axis('off')
    # plt.show()
    #save_path = os.path.join(saved_path, filename.split('.')[0] + '_ResNet50.png')
    #result_image.save(save_path)
    return result_image
    
#CamDisplayBase()
#ResNet50Cam()
#CamDisplayBest()



label_dict = {
    'cat': 10,
    'airplane': 4,
    'bear': 8,
    'bicycle': 3,
    'bird': 14,
    'boat': 9,
    'bottle': 11,
    'car': 13,
    'chair': 7,
    'clock': 5,
    'dog': 15,
    'elephant': 2,
    'keyboard': 1,
    'knife': 0,
    'oven': 6,  
    'truck': 12
    
}

def GeneratingSavingResult():
    global image_path
    global label
    image_path = "./Image/session-1/"
    
    if not os.path.exists("saved_Image"):
        os.makedirs("saved_Image")
        
    files = os.listdir(image_path)
    for file_name in files:
        if file_name.endswith(('.png')):  # Check if the file is an image
            
            full_path = os.path.join(image_path, file_name)  # Create full path to the image
            print(full_path)
            for key in label_dict:
                if key in file_name:
                    label = label_dict[key]
                    break
 
            res1 = ResNet50Cam(file_name, outlayer = False)
            res2 = ResNet50Cam(file_name ,outlayer = True)
            res3 = CamDisplayBase(file_name,outlayer = False)
            res4 = CamDisplayBase(file_name,outlayer = True)
            res5 = CamDisplayBest(file_name,outlayer = False)
            res6 = CamDisplayBest(file_name,outlayer = True)
            images = np.hstack((res1, res2, res3, res4, res5, res6))
            save_path = os.path.join(saved_path, file_name.split('.')[0] + '_cam.png')
            Image.fromarray(images).save(save_path)
            
            
            
GeneratingSavingResult()