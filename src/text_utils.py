import torch
from transformers import CLIPTokenizer, CLIPTextModel


CLIP_NAME = "openai/clip-vit-large-patch14" # modelo de Hugging Face

def load_clip(device: str)-> tuple:
    """
    Loads the pretrained CLIP tokenizer and text encoder.

    The text encoder is set to evaluation mode and its parameters are
    frozen to prevent gradient computation during training. The model
    is then moved to the specified device.

    Args:
        device (str): The device on which to load the text model
            (e.g., "cpu" or "cuda").

    Returns:
        tuple: A tuple containing:
            - tokenizer (CLIPTokenizer): The tokenizer used to
              tokenize text inputs for CLIP.
            - text_model (CLIPTextModel): The pretrained CLIP text
              encoder, with gradients disabled and in evaluation mode.
    """
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_NAME)
    text_model = CLIPTextModel.from_pretrained(CLIP_NAME)

    text_model.eval() 
    text_model.requires_grad_(False) # Congelar los gradientes de los parámetros
    text_model.to(device)

    return tokenizer, text_model

def encode_texts(texts: list[str], tokenizer: CLIPTokenizer, text_model: CLIPTextModel, device: str) -> torch.Tensor:
    """
    Encodes a batch of text captions into fixed-size CLIP embeddings.

    The input texts are tokenized with padding to a maximum sequence length
    of 77 tokens. Captions longer than this limit are truncated, while
    shorter captions are padded to the same length.

    The tokenized inputs are passed through the CLIP text encoder without
    computing gradients, since the text model is used as a frozen encoder.

    Args:
        texts (list[str]): A list of text captions to encode.
        tokenizer (CLIPTokenizer): The tokenizer used to convert captions
            into token IDs.
        text_model (CLIPTextModel): The pretrained CLIP text encoder.
        device (str): The device on which to perform the encoding
            (e.g., "cpu" or "cuda").

    Returns:
        torch.Tensor: A tensor containing the pooled text embeddings,
            with shape (batch_size, 768), where batch_size is the number
            of input captions and 768 is the dimensionality of the
            CLIP text embeddings.
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