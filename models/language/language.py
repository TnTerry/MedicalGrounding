import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class LMInject(nn.Module):
    def __init__(
        self,
        model_name_or_path:str,
    ) -> None:
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path
        ).to(self.device)
    
    def generate_knowledge(
        self,
        original_prompt:str,
        max_length:int
    ) -> str:
        inputs = self.tokenizer(
            original_prompt, return_tensors="pt"
        ).to(self.device)

        generate_ids = self.model.generate(
            inputs.input_ids, 
            max_length=max_length
        )

        message = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return message


if __name__ == "__main__":
    model_lm_inject = LMInject(
        model_name_or_path="m42-health/Llama3-Med42-8B",
    )

    prompt_message = "Briefly introduce kidney from a professional perspective. Focus on its shape, attribute, location in medical images and possible illnesses."
    result_max_length = len(prompt_message.split()) + 200

    knowledge = model_lm_inject.generate_knowledge(
        original_prompt=prompt_message,
        max_length=result_max_length
    )

    print(knowledge)

    '''
    The image is a non-contrasted computed tomography (CT) scan of the brain, showing the cerebral structures without any medical devices present. 
    The region of interest, located centrally and in the middle of the image, exhibits an area of altered density, which is indicative of a brain hemorrhage. 
    This area is distinct from the surrounding brain tissue, suggesting a possible hematoma or bleeding within the brain parenchyma. 
    The location and characteristics of this abnormality may suggest a relationship with the surrounding brain tissue, potentially causing a mass effect or contributing to increased intracranial pressure.
    '''