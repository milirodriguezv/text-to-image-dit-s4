import torch
from transformers import CLIPTokenizer, CLIPTextModel


CLIP_NAME = "openai/clip-vit-large-patch14" # modelo de Hugging Face

def load_clip(device: str)-> tuple:
    """
    input: device (string)
    output: tuple[tokenizer, text_model]
    """
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_NAME)
    text_model = CLIPTextModel.from_pretrained(CLIP_NAME)

    text_model.eval() 
    text_model.requires_grad_(False) # Congelar los gradientes de los parámetros
    text_model.to(device)

    return tokenizer, text_model

def encode_texts(texts, tokenizer, text_model, device):
    """
    Divide el texto en tokens rellenando siempre hasta 77 y acorta los captions demasiado largos
    """
    tokens = tokenizer(
        texts, 
        padding ="max_length", 
        max_length = 77, 
        truncation = True,
        return_tensors = "pt")

    # print(tokens)
    # print(tokens.input_ids.shape)

    with torch.no_grad():
        output = text_model(input_ids=tokens.input_ids.to(device))

    return output.pooler_output # shape (len (texts), 768)

# # Pruebas
# clip = load_clip("cpu")
# tokenizer = clip[0]
# text_model = clip[1]


# print(encode_texts(
#     texts= ["a dog on the grass", "a cat on a chair"],
#     tokenizer= tokenizer,
#     text_model= text_model,
#     device= "cpu"
# ).shape)