"""This script is the test script for Deep3DFaceRecon_pytorch
"""

import os
from PIL import Image
from facial_mask_game import Run
import numpy as np

def main(img_folder):
    output_dir = os.path.join(img_folder, 'output')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    im_path = [os.path.join(img_folder, i) for i in sorted(os.listdir(img_folder)) if i.endswith('png') or i.endswith('jpg')]
    for i in range(len(im_path)):
        print('Reading', im_path[i])
        # Read an image
        im = Image.open(im_path[i])

        print('Processing', im_path[i])
        # Run inference
        result = Run(im, 'cat') # 'cat' or 'tiger'. There will be more options in the future.

        # Save results
        result.save(os.path.join(output_dir, im_path[i].split('\\')[-1]))

if __name__ == '__main__':
    main('./test_data')
    
