import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import random

iteration = 100

transform = transforms.Compose([

    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


kernel_sharp_y = torch.tensor([
    [-1, -2, -1],
    [0.2, 0, 0.2],
    [1, 2, 1]
], dtype=torch.float32).repeat(3,1,1,1)

kernel_sharp_x = torch.tensor([
    [-1, 0.2, 1],
    [-2, 0, 2],
    [-1, 0.2, 1]
], dtype=torch.float32).repeat(3,1,1,1)

kernel_blur = torch.tensor([
    [1, 2, 1],
    [2, 3, 2],
    [1, 2, 1],
], dtype=torch.float32).repeat(3,1,1,1)

dataset = ImageFolder(root='dataset1/', transform=transform)
dataset_size = len(dataset)
index = random.randint(0, dataset_size-1)

image, label = dataset[index]
image = image * 0.5 + 0.5
image = image.unsqueeze(0)

conv_img = [0]*(iteration+1)
conv_img[0] = image

for i in range(iteration):


    conv_img[i] = F.conv2d(conv_img[i], kernel_sharp_x, padding=1, groups=3)
    conv_img[i] = F.conv2d(conv_img[i], kernel_sharp_y, padding=1, groups=3)
    conv_img[i+1] = torch.clamp(conv_img[i], 0.0, 1.0)
    conv_img[i] = F.conv2d(conv_img[i], kernel_blur, padding=1, groups=3)
    conv_img[i+1] = torch.clamp(conv_img[i], 0.0, 1.0)
    print(conv_img[i].min(), conv_img[i].max())



image = image.squeeze().permute(1, 2, 0)
conv_img = [img.squeeze(0).permute(1, 2, 0) for img in conv_img]


fig, axes = plt.subplots(2, 3, figsize=(21, 15))
axes[0, 0].imshow(image)
axes[0, 1].imshow(conv_img[1])
axes[0, 2].imshow(conv_img[5])
axes[1, 0].imshow(conv_img[10])
axes[1, 1].imshow(conv_img[50])
axes[1, 2].imshow(conv_img[iteration])

plt.tight_layout()
plt.show()