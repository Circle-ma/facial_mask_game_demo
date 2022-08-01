"""This script defines the face reconstruction model for Deep3DFaceRecon_pytorch
"""

import numpy as np
import os, torch
from util.load_mats import load_lm3d
from mtcnn import detect_faces
from util.preprocess import align_img
from PIL import Image
from renderer.render import prepare_for_render


facial_mask_type = {'tiger': './facial_mask/tiger.npy', 'cat': './facial_mask/cat.npy'}


def read_data(im, lm3d_std, to_tensor=True):
    # to RGB
    bounding_boxes, landmarks = detect_faces(im)
    horizontal = landmarks[:, :5].reshape((-1, 1))
    vertical = landmarks[:, 5:].reshape((-1, 1))
    combine = np.hstack((horizontal, vertical))
    landmarks = np.round(combine)

    im = im.convert('RGB')
    W,H = im.size
    lm = landmarks[:5, :]
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _, recon_info = align_img(im, lm, lm3d_std)
    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    recon_info.update({'org_w': W, 'org_h': H})
    return im, recon_info 

def compute_visuals(input_img, coeffs, renderer, facial_mask, device='cpu'):
    with torch.no_grad():

        pred_mask, pred_face = renderer(coeffs['id'], coeffs['exp'], coeffs['angle'], coeffs['trans'], coeffs['tex'], coeffs['gamma'], facial_mask)
        
        pred_mask = torch.from_numpy(pred_mask.copy()).to(device)
        pred_face = torch.from_numpy(pred_face).to(device)

        input_img_numpy = 255. * input_img.detach().cpu().permute(0, 2, 3, 1).numpy()
        output_vis = pred_face * pred_mask + (1 - pred_mask) * input_img
        output_vis_numpy = 255. * output_vis.detach().cpu().permute(0, 2, 3, 1).numpy()
        output_vis_numpy = np.clip(output_vis_numpy[0], 0, 255)
        return Image.fromarray(np.uint8(output_vis_numpy))

def split_coeff(coeffs):
    """
    Return:
        coeffs_dict     -- a dict of torch.tensors

    Parameters:
        coeffs          -- torch.tensor, size (B, 256)
    """
    id_coeffs = coeffs[:, :80]
    exp_coeffs = coeffs[:, 80: 144]
    tex_coeffs = coeffs[:, 144: 224]
    angles = coeffs[:, 224: 227]
    gammas = coeffs[:, 227: 254]
    translations = coeffs[:, 254:]
    return {
        'id': id_coeffs,
        'exp': exp_coeffs,
        'tex': tex_coeffs,
        'angle': angles,
        'gamma': gammas,
        'trans': translations
    }

def Run(im, facial_mask_key):
    im = im.convert('RGB')
    device = 'cpu'
    # Loading model
    model_path = './checkpoints/im_3d_torchscript.pth'
    model = torch.jit.load(model_path)

    lm3d_std = load_lm3d('./BFM')
    im_tensor, recon_info = read_data(im, lm3d_std)
    renderer = prepare_for_render(focal=1015, center=112., device=device, camera_distance=10., init_lit=np.array([0.8, 0, 0, 0, 0, 0, 0, 0, 0]), recenter=True, znear=5., zfar=15., bfm_file='./BFM/bfm_compressed.npz')
    facial_mask = facial_mask_type[facial_mask_key]
    with torch.no_grad():
        output_coeff = model(im_tensor)
        pred_coeffs = split_coeff(output_coeff)
        output_im = compute_visuals(im_tensor, pred_coeffs, renderer, facial_mask)

    img_resize, left, up, target_size, org_w, org_h, new_w, new_h = np.asarray(recon_info['img_resize']).copy(), recon_info['left'], recon_info['up'], int(recon_info['target_size']), recon_info['org_w'], recon_info['org_h'], recon_info['new_w'], recon_info['new_h']
    right, down = left+target_size, up+target_size
    output_im = np.asarray(output_im)
    if left < 0:
        output_im = output_im[:, -left:]
        left = 0
    if up < 0:
        output_im = output_im[-up:, :]
        up = 0
    if right > new_w:
        output_im = output_im[:, :new_w - right]
        right = new_w 
    if down > new_h:
        output_im = output_im[:new_h - down, :]
        down = new_h
    img_resize[up:down, left:right] = output_im
    img_resize = Image.fromarray(img_resize)
    img_recon = img_resize.resize((org_w, org_h), resample=Image.BICUBIC)

    return img_recon


