from torchvision import transforms
from PIL import Image

def init_transform_dict(input_res=224):
    """
    Initializes a dictionary with different torchvision transforms for the train, val, and test sets.
    Parameters:
        input_res (int): The input resolution for the transforms. Defaults to 224.
    Returns:
        dict: A dictionary containing torchvision transforms for the train, val, and test sets.
    """
    transform_dict = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_res, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0, saturation=0, hue=0),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_res, interpolation=Image.BICUBIC),
            transforms.CenterCrop(input_res),
            transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_res, interpolation=Image.BICUBIC),
            transforms.CenterCrop(input_res),
            transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
        ])
    }

    return transform_dict