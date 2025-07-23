# eurosat-alexnet-classification-pytorch
Satellite image classification using AlexNet network implemented in Pytorch

All the updates will be announced on my LinkedIn account [Here](https://www.linkedin.com/in/kacper-litwi%C5%84czyk-0714ab350/)

### Project Overview
Project presents a **Convolutional Neural Network** for classifying satellite images from the [EuroSAT dataset (RGB version)](https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1) into land use and land cover categories. The network is an **Original Implementation of the AlexNet architecture** in PyTorch, *adapted to handle EuroSAT's **RGB images***.
### Dataset and Preprocessing
- **Dataset:** [EuroSAT dataset (RGB version)](https://zenodo.org/records/7711810#.ZAm3k-zMKEA)
- **Number of Classes:** 10 (e.g. Annual Crop, Forest, Residential etc.).
- **Image Size:** 64x64 pixels, RGB.
#### **Preprocessing and Augmentation**
- Images were resized to 224x224 pixels to fit the **AlexNet** input size.
- **RandomHorizontalFlip:** Randomly flips images horizontally with probability 0.5.
- **RandomVerticalFlip:** Randomly flips images vertically with probability 0.5.
- **RandomRotation:** Rotates images randomly within ±30 degrees.
- **ToTensor:** Converts numpy arrays to PyTorch tensors.
- **Normalization:** Standardizes images using ImageNet mean and standard deviation:
`mean=[0.485, 0.456, 0.406]`
`std=[0.229, 0.224, 0.225]`

```python
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```
The data augmentation techniques help prevent overfitting and make the model more robust to variations in satellite imagery.
### Model Architecture
- The model is based on the **original AlexNet** (Krizhevsky et al. 2012).
- The only modification is the number of output classes (10, to match EuroSAT data)
- The architecture consist of **5 convolutional layers** and **3 fully connected layers** with ReLU activation function and Dropout regularization.

```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```
### Key Insights & Future Work
- The custom AlexNet achieved 77.74% accuracy on the test set.
- Data augmentation improved model generalization.<br>
#### **Future Possibilities**
- Hyperparameter tuning
- Using transfer learning with deeper architectures (e.g. ResNet, DenseNet)
- More extensive data augmentation
### Papers
- [ImageNet Classification with Deep Convolutional Neural Networks](https://www.researchgate.net/publication/267960550_ImageNet_Classification_with_Deep_Convolutional_Neural_Networks)
- [EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification](https://ieeexplore.ieee.org/document/8519248)
