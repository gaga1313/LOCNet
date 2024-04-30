import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms
from harmonization.evaluation import evaluate_clickme
from src.models import *
from src.models.autoencoder import AutoEncoder
from src.utils.harmo import custom_load_clickme_val

# then, you need to specify
# (i) your model
# (ii) the preprocessing function
# (iii) the explanation function


class TestModel(nn.Module):
    def __init__(self, ckpt_path='../../ckpt/model_best.pth.tar'):
        super().__init__()
        self.model = AutoEncoder()
        self.model.load_state_dict(torch.load(ckpt_path)['state_dict'])
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


# (i) the model
model = TestModel()
model.eval()

# (ii) the preprocessing function
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# (iii) the explainer (saliency)
def torch_explainer(xbatch, ybatch):
    # preprocess the input
    xbatch = torch.stack([transform(image) for image in xbatch.numpy().astype(np.uint8)])

    ybatch = torch.Tensor(ybatch.numpy())

    xbatch.requires_grad_()

    out = model(xbatch)

    output = torch.sum(out * ybatch)
    output.backward()

    saliency, _ = torch.max(xbatch.grad.data.abs(), dim=1)
    # explainer need to return numpy array
    saliency = np.array(saliency)

    # for debugging
    # visualize_saliency(xbatch, saliency)
    # visualize_saliency(xbatch[:1], saliency[:1])
    # print(ybatch[0])
    # print(out[0])

    return saliency


# visualization function
def visualize_saliency(images, saliency_maps):
    import matplotlib.pyplot as plt
    # Convert images from PyTorch's NCHW format to NumPy's NHWC format for visualization
    images = images.detach().permute(0, 2, 3, 1).cpu().numpy()  # Assuming images tensor is already on the appropriate device

    # unnormalize the images
    images = images * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])

    num_images = len(images)
    fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(10, 5 * num_images))
    if num_images == 1:
        axes = [axes]

    for i, (image, saliency) in enumerate(zip(images, saliency_maps)):
        ax1 = axes[i][0]
        ax2 = axes[i][1]

        ax1.imshow(image)
        ax1.set_title('Original Image')
        ax1.axis('off')

        # Display the original image and overlay the saliency map with transparency
        ax2.imshow(image, alpha=0.6)
        ax2.imshow(saliency, cmap='jet', alpha=0.4)  # Using jet colormap for saliency visibility
        ax2.set_title('Saliency Map')
        ax2.axis('off')

    plt.tight_layout()
    plt.show()


# now let's load the dataset
clickme_dataset = custom_load_clickme_val(batch_size=128)

# we're ready to get our score, here we test only on the first 5 batches
scores = evaluate_clickme(model,
                          explainer=torch_explainer,
                          clickme_val_dataset=clickme_dataset.take(5))
print(scores['alignment_score'])
