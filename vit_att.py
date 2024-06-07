import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image, to_tensor

class CLIPVisionTransformerClassifier(nn.Module):
    def __init__(self, clip_vision_transformer, num_classes):
        super(CLIPVisionTransformerClassifier, self).__init__()
        self.clip_vision_transformer = clip_vision_transformer
        self.classifier = nn.Linear(768, num_classes)
        
        # Register hook for the last attention layer
        self.last_attention_layer = self.clip_vision_transformer.encoder.layers[-1]
        self.hook_handle = self.last_attention_layer.register_forward_hook(self.extract_qk_hook)
        
        # Initialize storage for q and k
        self.captured_q = None
        self.captured_k = None

    def extract_qk_hook(self, module, input, output):
        input_tensor = input[0]
        q = module.self_attn.q_proj(input_tensor)
        k = module.self_attn.k_proj(input_tensor)

        # Store q, k in the module for later retrieval
        self.captured_q = q
        self.captured_k = k

    def forward(self, x):
        x = self.clip_vision_transformer(x)[0]
        cls_token_embedding = x[:, 0]
        logits = self.classifier(cls_token_embedding)
        return logits

    def get_attention_map(self):
        if self.captured_q is None or self.captured_k is None:
            raise ValueError("Attention not captured yet or no forward pass executed.")
        head_dim = self.last_attention_layer.self_attn.head_dim
        attn_scores = torch.matmul(self.captured_q, self.captured_k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        return attn_weights



def resize_vit_att(image_tensor, new_height, new_width):

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((new_height, new_width))
    ])

    # Apply the transformation to each image in the batch
    # Convert tensor to PIL image, apply the transform, then convert back to tensor
    resized_grayscaled_images = torch.stack([
        to_tensor(transform(to_pil_image(img))) for img in image_tensor
    ]).squeeze(1)  # Remove the color channel dimension which is 1 after grayscaling

    return resized_grayscaled_images