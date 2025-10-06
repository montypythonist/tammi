from transformers import (
    BeitImageProcessor,
    BeitForImageClassification,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)
from PIL import Image
import cv2 as cv
import torch
import torch.nn.functional as F
import sounddevice as sd
import numpy as np
import threading
# why are the dependencies in a file by themselves? i wanted to look cool, leave me alone