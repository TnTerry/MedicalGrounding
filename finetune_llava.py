import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.language.medllama3 import *
from models.visual.MedSAM import *

llava_model = AutoModelForCausalLM.from_pretrained(
    ""
)