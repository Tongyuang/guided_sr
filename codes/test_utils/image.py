# code for image upscale only from LR 
import numpy as np
import options.options as option
import utils.util as util
import data.util as data_util
from models import create_model
import yaml
import os
import torch
import time

TEST_YAML_DIR = "./options/test_farm"

def yml_loader(yml_dir):
    assert os.path.exists(yml_dir)
    opt = option.parse(yml_dir,is_train=False)
    opt = option.dict_to_nonedict(opt)
    return opt

def create_model_by(upscale_factor = 2):
    # upscale factor should only be 2,4 or 8
    try:
        assert upscale_factor in [2,4,8]
    except AssertionError:
        print("Upscale factor should be 2,4 or 8")
        return None
    
    # load yaml file
    parser_dir = os.path.join(TEST_YAML_DIR, "test_IRN_x{}.yml".format(upscale_factor))
    opt = yml_loader(parser_dir)
    
    # load model
    model = create_model(opt)
    
    return model

def load_image_to_device(image_dir=None, image_in=None, device='cpu'):
    if image_dir is None:
        assert image_in is not None
        image = image_in
    elif image_in is None:
        assert image_dir is not None
        image = data_util.read_img(None, image_dir, None)
    image = image[:, :, [2, 1, 0]]
    image = torch.from_numpy(np.ascontiguousarray(np.transpose(image, (2, 0, 1)))).float()
    image = torch.unsqueeze(image,0)
    image = image.to(device)
    return image

def upscale(image_dir=None, image=None, from_dir = True, factor=2):

    # load model
    print('creating model...', end=' ')
    time_now = time.time()
    model = create_model_by(factor)
    print('done in {:.3f}ms'.format(1000*(time.time() - time_now)))
    
    # load image
    if from_dir:
        assert image_dir is not None
        print('loading image from {}...'.format(image_dir), end=' ')
        time_now = time.time()
        device = model.device
        image = load_image_to_device(image_dir = image_dir, image_in=None, device=device)
        print('done in {:.3f}ms'.format(1000*(time.time() - time_now)))
    else:
        assert image is not None
        print('transforming image...', end=' ')
        device = model.device
        image = image = load_image_to_device(image_dir = None, image_in=image, device=device)
        print('done in {:.3f}ms'.format(1000*(time.time() - time_now)))
        
    # upscale image
    print('upscaling image...', end=' ')
    time_now = time.time()
    HR_img = model.upscale(LR_img = image,scale=factor,gaussian_scale=1)
    HR_img = util.tensor2img(HR_img)
    HR_img = HR_img[:, :, [2, 1, 0]]
    print('Done in {:.3f}ms'.format(1000*(time.time() - time_now)))
    return HR_img

def downscale(image_dir=None, image=None, from_dir = True, factor=2):
        # load model
    print('creating model...', end=' ')
    time_now = time.time()
    model = create_model_by(factor)
    print('done in {:.3f}ms'.format(1000*(time.time() - time_now)))
    
    # load image
    if from_dir:
        assert image_dir is not None
        print('loading image from {}...'.format(image_dir), end=' ')
        time_now = time.time()
        device = model.device
        image = load_image_to_device(image_dir = image_dir, image_in=None, device=device)
        print('done in {:.3f}ms'.format(1000*(time.time() - time_now)))
    else:
        assert image is not None
        print('transforming image...', end=' ')
        device = model.device
        image = image = load_image_to_device(image_dir = None, image_in=image, device=device)
        print('done in {:.3f}ms'.format(1000*(time.time() - time_now)))
        
    # upscale image
    print('downscaling image...', end=' ')
    time_now = time.time()
    
    LR_img = model.downscale(HR_img = image)
    LR_img = util.tensor2img(LR_img)
    LR_img = LR_img[:, :, [2, 1, 0]]
    print('Done in {:.3f}ms'.format(1000*(time.time() - time_now)))
    return LR_img

def save_image(image, save_dir):
    util.save_img(image, save_dir)
    