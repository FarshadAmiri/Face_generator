import sys ; sys.path.insert(0, r"D:\Projects\face_generator")
import os ; os.chdir(r"D:\Projects\face_generator")
import pickle
import numpy as np
import PIL.Image
from IPython.display import Image
import matplotlib.pyplot as plt
import IPython.display
import torch
import dnnlib
import legacy


def seed2vec(gan_model, seed):
  return np.random.RandomState(seed).randn(1, gan_model.z_dim)


def display_image(image):
  plt.axis('off')
  plt.imshow(image)
  plt.show()


def get_label(gan_model, class_idx, device):
  label = torch.zeros([1, gan_model.c_dim], device=device)
  if gan_model.c_dim != 0:
      if class_idx is None:
          print("Must specify class label with --class when using "\
            "a conditional network")
      label[:, class_idx] = 1
  else:
      if class_idx is not None:
          print ("warn: --class=lbl ignored when running on "\
            "an unconditional network")
  return label


def generate_image(gan_model, vec_or_seed, device, truncation_psi=1.0, noise_range=0):
  if type(gan_model) == str:
     with open(gan_model, "rb") as f:
        gan_model = legacy.load_network_pkl(f)['G_ema'].to(device)
    
  if type(vec_or_seed) == int:
    vec = seed2vec(gan_model, vec_or_seed)
  elif type(vec_or_seed) == np.ndarray:
    vec = vec_or_seed
  else:
    raise TypeError("vec_or_seed must be int or np.ndarray")
  
  if noise_range != 0:
    vec = vec + np.random.uniform(-noise_range, noise_range, size=(1, 512))
        
  vec = torch.from_numpy(vec).to(device)
  label = get_label(gan_model, class_idx=None, device=device)
  img = gan_model(vec, label, truncation_psi=truncation_psi, noise_mode='const')
  img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(\
      torch.uint8)
  return PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')