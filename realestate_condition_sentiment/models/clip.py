from typing import List, Tuple, Union
from pathlib import Path

import PIL
import torch
from torch import mps

import open_clip


class OpenClipModel:
  """
  Class for a OpenCLIP model. Provide more convenient interface/methods for using CLIP for embedding,
  zero shot classification, captioning, etc. 
  """

  def __init__(self, model_name='ViT-B-32', tokenizer_name: str='ViT-B-32', pretrained: str='laion2b_s34b_b79k', device=torch.device('cpu')):
    self.model_name = model_name
    self.device = device
    self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    self.model.eval()
    self.model.to(device)

    self.tokenizer = open_clip.get_tokenizer(tokenizer_name)

    if device == torch.device('cuda'):
      self.empty_cache_func = torch.cuda.empty_cache
    elif device == torch.device('mps'):
      self.empty_cache_func = mps.empty_cache
    else:
      self.empty_cache_func = lambda: None

  def embed_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
    """
    Embed text into a normalized tensor of shape (n, embedding_dim), where n is the number of texts.
    """
    if isinstance(texts, str):
      texts = [texts]

    tokenized_texts = self.tokenizer(texts).to(self.device)
    with torch.no_grad():
      text_features = self.model.encode_text(tokenized_texts)
      text_features /= text_features.norm(dim=-1, keepdim=True)

    text_features = text_features.cpu()
    self.empty_cache_func()

    return text_features
  
  def embed_image(self, image_paths: Union[str, List[str], Path, List[Path]]) -> torch.Tensor:
    """
    Embed image into a normalized tensor of shape (n, embedding_dim), where n is the number of images.

    images can be a list of image paths or a single image path (either str or pathlib.Path)
    """
    image_paths = [image_paths] if isinstance(image_paths, str) or isinstance(image_paths, Path) else image_paths
    
    images = torch.stack([self.preprocess(PIL.Image.open(img).convert('RGB')) for img in image_paths], dim=0).to(self.device)

    with torch.no_grad():
      image_features = self.model.encode_image(images)
      image_features /= image_features.norm(dim=-1, keepdim=True)

    image_features = image_features.cpu()
    self.empty_cache_func()

    return image_features
  
  def one_shot_classify(self, text_prompt_features: torch.Tensor, image_features: torch.Tensor):
    """
    Classify image features with text prompts. 
    """
    probs = (100.0 * image_features @ text_prompt_features.T).softmax(dim=-1)   
    
    return probs
  
  def generate_captions(self, image_paths):    
    '''
    For models capability of generateing captions E.g. CoCa
    '''
    image_paths = [image_paths] if isinstance(image_paths, str) or isinstance(image_paths, Path) else image_paths

    images = torch.stack([self.preprocess(PIL.Image.open(img).convert('RGB')) for img in image_paths], dim=0).to(self.device)
    with torch.no_grad():
      captions = self.model.generate(images)

    captions = captions.cpu()
    self.empty_cache_func()

    return [open_clip.decode(caption) for caption in captions]

  @staticmethod
  def cleanup_coca_captions(captions: List[str]) -> List[str]:
    """
    Clean up CoCa captions by removing the <bos> and <eos> tokens.
    """
    # 
    return [caption.split('<end_of_text>')[0].replace('<start_of_text>', '').strip() for caption in captions]

