from config import Config
from torch.utils.data import DataLoader
from data.transform import init_transform_dict
from data.mustard import Mustard
# import datasets

class DataHandler:

    def get_data_loader(config: Config, mode: str):
        img_transforms = init_transform_dict(config.input_resolution)
        train_transform = img_transforms['train']
        val_transform = img_transforms['val']
        test_transform = img_transforms['test']
        
        if mode == 'train':
            train_dataset = Mustard(config, mode='train', img_transforms=train_transform)
            return DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
        elif mode == 'val':
            val_dataset = Mustard(config, mode='val', img_transforms=val_transform)
            return DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        elif mode == 'test':
            test_dataset = Mustard(config, mode='test', img_transforms=test_transform)
            return DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
        else:
            #raise ValueError('mode must be one of train, val, test')
            raise NotImplementedError