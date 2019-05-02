import torch
from torchvision.models import resnet18 
from torchvision import transforms
import numpy as np 
from torch.autograd import Tensor

scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def get_network_loss(gan_output, true_output, layer):
    model = resnet18(pretrained=True)
    model.eval() 
    
    # Tensor Output
    gan_output_tensor = Tensor(normalize(to_tensor(scaler(gan_output))).unsqueeze(0))
    true_output_tensor = Tensor(normalize(to_tensor(scaler(true_output))).unsqueeze(0))

    if layer == 1:
        gan_vector = torch.zeros(64)
        true_vector = torch.zeros(64)
        model_layer = model._modules.get("layer1")[1].conv2

    elif layer == 2:
        gan_vector = torch.zeros(128)
        true_vector = torch.zeros(128)
        model_layer = model._modules.get("layer2")[1].conv2
 
    elif layer == 3:
        gan_vector = torch.zeros(256)
        true_vector = torch.zeros(256)
        model_layer = model._modules.get("layer3")[1].conv2 

    else:
        print("Layer is wrong: ", layer)

    def copy_gan(m, i, o):
        gan_vector.copy_(o.data)

    gan_hook = model_layer.register_forward_hook(copy_gan)
    model(gan_output_tensor)
    gan_hook.remove()   

    def copy_true(m, i, o):
        true_vector.copy_(o.data)

    true_hook = model_layer.register_forward_hook(copy_true)
    model(true_output_tensor)
    true_hook.remove()   

    return np.linalg.norm(gan_vector - true_vector)
