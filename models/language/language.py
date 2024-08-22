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
        knowledge_max_length:int
    ) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.knowledge_max_length = knowledge_max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path
        ).to(self.device)
    
    def generate_knowledge(
        self,
        original_prompt:str
    ) -> str:
        inputs = self.tokenizer(
            original_prompt, return_tensors="pt"
        ).to(self.device)

        generate_ids = self.model.generate(
            inputs.input_ids, 
            max_length=self.knowledge_max_length
        )

        message = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return message


if __name__ == "__main__":
    model_lm_inject = LMInject(
        model_name_or_path="m42-health/Llama3-Med42-8B",
        knowledge_max_length=60
    )

    knowledge = model_lm_inject.generate_knowledge(
        "Briefly introduce kidney."
    )

    print(knowledge)