import os
import torch
from torch.utils.data import DataLoader
from utils.data_loading import BasicDataset
from Nofpn_ablution import SPMNet
from utils.path_hyperparameter import ph
from PIL import Image
import numpy as np
import logging

def create_dirs(base_dir):
    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Create subdirectories for T1, T2, labels, and predictions
    t1_dir = os.path.join(base_dir, 'T1')
    t2_dir = os.path.join(base_dir, 'T2')
    label_dir = os.path.join(base_dir, 'Label')
    pred_dir = os.path.join(base_dir, 'Prediction')

    for dir_path in [t1_dir, t2_dir, label_dir, pred_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    return t1_dir, t2_dir, label_dir, pred_dir

def visualize_results(dataset_name, model_path, save_dir):
    # Create directory structure
    t1_dir, t2_dir, label_dir, pred_dir = create_dirs(save_dir)

    # Load the dataset
    test_dataset = BasicDataset(t1_images_dir=f'{dataset_name}/test/t1/',
                                t2_images_dir=f'{dataset_name}/test/t2/',
                                labels_dir=f'{dataset_name}/test/label/',
                                train=False)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=4)

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SPMNet(dims=ph.dims, depths=ph.depths, ssm_d_state=ph.ssm_d_state, ssm_dt_rank=ph.ssm_dt_rank, 
                 ssm_ratio=ph.ssm_ratio, mlp_ratio=ph.mlp_ratio, downsample_version=ph.downsample_version, 
                 patchembed_version=ph.patchembed_version)
    net.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    net.to(device=device)
    net.eval()

    logging.info(f'Model loaded from {model_path}')
    logging.info('Visualizing results')

    # Visualize and save test images
    with torch.no_grad():
        for i, (img1, img2, label, name) in enumerate(test_loader):
            img1 = img1.to(device).float()
            img2 = img2.to(device).float()
            label = label.to(device).float()

            pred = net(img1, img2)
            pred = torch.sigmoid(pred).cpu().numpy()[0, 0]

            img1 = img1.cpu().numpy()[0].transpose((1, 2, 0))
            img2 = img2.cpu().numpy()[0].transpose((1, 2, 0))
            
            # Convert img1 and img2 to uint8 format
            img1 = (img1 * 255).astype(np.uint8)
            img2 = (img2 * 255).astype(np.uint8)
            
            # Adjust label shape to ensure it is in the correct format
            label = label.cpu().numpy()[0]
            if label.ndim == 2:  # If the label is a 2D array, convert it to a 3-channel RGB format
                label = np.stack((label,) * 3, axis=-1)
            elif label.ndim == 3 and label.shape[0] == 1:  # If single channel, convert to 3 channels
                label = np.repeat(label, 3, axis=0).transpose((1, 2, 0))
            elif label.ndim != 3 or label.shape[0] != 3:  # Handle any other unexpected shape
                raise ValueError(f"Unexpected label shape: {label.shape}")
            
            label = (label * 255).astype(np.uint8)

            # Convert pred to uint8 format
            pred = (pred * 255).astype(np.uint8)

            # Save images to respective directories using PIL
            Image.fromarray(img1).save(os.path.join(t1_dir, f'{name[0]}_t1.png'))
            Image.fromarray(img2).save(os.path.join(t2_dir, f'{name[0]}_t2.png'))
            Image.fromarray(label).save(os.path.join(label_dir, f'{name[0]}_label.png'))
            Image.fromarray(pred).save(os.path.join(pred_dir, f'{name[0]}_pred.png'))

            logging.info(f'Saved {name[0]} images to respective directories')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    visualize_results(dataset_name='/root/fpn/Dataset/256levir', 
                      model_path='/root/fpn/train_levircd_256_0.001_best_f1score_model/best_f1score_epoch39_Thu Aug  1 12:36:12 2024.pth', 
                      save_dir='/root/fpn/39outimage')
