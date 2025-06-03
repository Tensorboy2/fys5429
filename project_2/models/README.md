# Models
Within this folder there are three main model architectures with different configurations.

**resnet.py**:
- ResNet$50$
- ResNet$101$

**vit.py**:
- ViT-B$16$

**convnext.py**:
- ConvNeXtTiny
- ConvNeXtSmall

An example of usage is through:
```python
    import torch
    from model_name.py import config_name
    model = config_name(image_size=128, num_classes=4, pretrained=False)
    num_samples = 2
    x = torch.randn((num_samples, 1, 128, 128)) # Two samples, 1 color channels, size = (128,128).
    y = model(x) # shape = (num_samples, num_classes)
```
Switch out <model_name> to script where the model lies, and <config_name> to name of the configuration.