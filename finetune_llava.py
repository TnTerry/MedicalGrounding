import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from lmdeploy import pipeline, TurbomindEngineConfig

from models.language.medllama3 import *
from models.visual.MedSAM import *


engine_config = TurbomindEngineConfig(quant_policy=4, cache_max_entry_count=0.3)
pipe = pipeline("m42-health/Llama3-Med42-8B", backend_config=engine_config)
response = pipe(["Briefly introduce polyp from a professional perspective. Focus on its shape, attribute, location in medical images and possible illnesses."])
print(response[0].text)