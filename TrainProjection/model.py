import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchinfo import summary

class PhiWrapper(nn.Module):
    def __init__(self):
        ## need to pip install flash_attn
        self.frozen_phi = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
        self.phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)


    def forward(self, x):
        outputs = self.frozen_phi.generate(**inputs, max_length=200)
        text = self.phi_tokenizer.batch_decode(outputs)[0]
        return text


if __name__ == "__main__":
    phi = PhiWrapper()

    ## this is all fake, just to see what tokenizer and summary emit about the model
    ## currently running into environment issues with transformers TODO 
    summary(phi.frozen_phi)
    inputs = phi.tokenizer('hi model, whats up?', return_tensors="pt", return_attention_mask=False)
    text = phi(inputs)
