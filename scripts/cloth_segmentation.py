import shutil
import os
import sys
from modules import scripts_postprocessing
from modules.ui_components import FormRow
import gradio as gr
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import infer

def callback():
    shutil.copytree( './extensions/sd-webui-cloth-segmentation/trained_checkpoint', './repositories/clothSegmentation/trained_checkpoint', dirs_exist_ok=True )

class ScriptPostprocessingClothSegmentation(scripts_postprocessing.ScriptPostprocessing):
    name = 'Cloth Segmentation'

    def __init__(self):
        while True:
            if os.path.exists( './repositories/clothSegmentation' ):
                break

        callback()

    def ui( self ):
        with gr.Accordion( 'Cloth segmentation', open=False ):
            mask_index = gr.Radio(
                label='clothes category: Upper body, Lower body and Full body',
                choices=['Upper body', 'Lower body', 'Full body', 'None'],
                type='index',
            )

        return {
            'mask_index': mask_index
        }

    def process( self, img: scripts_postprocessing.PostprocessedImage, mask_index ):
        print(mask_index)
        max_masks_len = 3

        if mask_index is None or mask_index >= max_masks_len:
            return

        masks = infer.run( img.image )
        img.image = masks[mask_index]
        img.info['ClothSegmentation'] = 'Upper body' if mask_index == 0 else 'Lower body' if mask_index == 1 else 'Full body'
