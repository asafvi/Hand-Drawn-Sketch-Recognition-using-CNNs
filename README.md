# Hand-Drawn-Sketch-Recognition-using-CNNs
Designed and implemented a pipeline to solve sketch recognition using convolutional neural networks. It makes use of a backbone of pretraind ResNet50, followed by a classifier model that is fine-tuned on the TU-Berlin dataset.

Backbone model: Pretrained ResNet50

```py
backbone_model = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2]).to(device)

for param in backbone_model.parameters():
    param.requires_grad = False

for param in backbone_model[-1].parameters():
    param.requires_grad = True

backbone_model
```
Classifier model: Dense layer with dropout layers.

```py
self.input_size = input_size

    self.backbone = backbone_model

    self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
    )

    self.classfier = nn.Sequential(

            nn.Dropout(0.5),
            nn.Linear(2048, 1500),
            nn.BatchNorm1d(1500, momentum=BN_momentum),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(1500, 1000),
            nn.BatchNorm1d(1000, momentum=BN_momentum),
            nn.ReLU(),

            nn.Dropout(0.5),
            nn.Linear(1000, 500),
            nn.BatchNorm1d(500, momentum=BN_momentum),
            nn.ReLU(),

            nn.Linear(500, classes)
    )
```

Fine-tuned on the [TU-Berlin Sketch dataset](https://drive.google.com/drive/folders/1u9jIxbQ6u5F1LEe7G_rq1sxF4_49JbGr?usp=sharing), as followed:
- Training Set: 40 images per class – 40 x250 = 10,000 images
- Validation Set: 20 images per class – 20 x 250 = 5,000 images
- Test Set: 20 images per class – 20 x 250 = 5,000 images

