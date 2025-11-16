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
import noisereduce as nr
from collections import deque
import whisper

# why are the dependencies in a file by themselves? it makes it easier to organize them, leave me alone