"""
Common operational stuff to support the Condition Sentiment project
"""

from typing import List, Union
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import torch

from ..models.clip import OpenClipModel


def get_image_emeddings(data_dir: Path, 
                        model_name=None, 
                        tokenizer_name=None, 
                        pretrained=None, 
                        batch_size=128,
                        device=None):
  """
  Use openclip, which has different API signatures than huggingface transformers

  data_dir: directory containing images

  Actions: Obtain embeddings of all images in data_dir and save them in a npz file in the same data_dir
  """
  if model_name is None:
    model_name = 'coca_ViT-L-14'
    tokenizer_name = 'coca_ViT-L-14'
  else:
    if tokenizer_name is None:
      tokenizer_name = model_name    # usually they have same names
  
  if pretrained is None:
    pretrained = 'mscoco_finetuned_laion2B-s13B-b90k'

  if device is None:
    device = torch.device('cpu')

  open_clip_model = OpenClipModel(model_name=model_name, tokenizer_name=tokenizer_name, pretrained=pretrained, device=device)

  # assuming data_dir directly contains all images (no recursive subdir op)
  image_paths = data_dir.lf('*.jpg')
  print(f'# of images: {len(image_paths)}')

  # embed image_paths in batch of batch_size
  image_embeddings = []
  for i in tqdm(range(0, len(image_paths), batch_size)):
    image_batch = image_paths[i:i+batch_size]
    image_embeddings.append(open_clip_model.embed_image(image_batch))
  image_embeddings = torch.cat(image_embeddings, dim=0)  

  # store image_paths and image_embeddings in a npz file
  np.savez_compressed(data_dir/'image_embeddings.npz', image_paths=image_paths, image_embeddings=image_embeddings)
