# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("ProbeMedicalYonseiMAILab/medllama3-v20")
# model = AutoModelForCausalLM.from_pretrained("ProbeMedicalYonseiMAILab/medllama3-v20")

# # Use a pipeline as a high-level helper
# from transformers import pipeline

# messages = [
#     {"role": "user", "content": "Who are you?"},
# ]
# pipe = pipeline("text-generation", model="ProbeMedicalYonseiMAILab/medllama3-v20")
# pipe(messages)

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class LMInject(nn.Module):
    def __init__(
        self,
        model_name_or_path:str,
    ) -> None:
        super(LMInject).__init__()
        self.model_name_or_path = model_name_or_path
        self.medtokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path
        )
        self.medlm = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path
        )
    
    def generate_knowledge(
        self,
        original_prompt:str
    ) -> str:
        pass