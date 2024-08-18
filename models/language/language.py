# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("ProbeMedicalYonseiMAILab/medllama3-v20")
# model = AutoModelForCausalLM.from_pretrained("ProbeMedicalYonseiMAILab/medllama3-v20")

# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="ProbeMedicalYonseiMAILab/medllama3-v20")
pipe(messages)