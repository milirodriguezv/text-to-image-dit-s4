import numpy as np, os, argparse, torch
from text_utils import load_clip, encode_texts

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, text_model = load_clip(device)

    e_null = encode_texts([""], tokenizer, text_model, device)

    arr = e_null.cpu().numpy().astype(np.float16)
    os.makedirs("features/train", exist_ok=True)
    np.save("features/train/null_empty_string.npy", arr)

    ## Pruebas
    # print(arr.shape)
    # print(arr.dtype)
    # print(np.isfinite(arr).all())

    # # Sanity check
    # e_dog = encode_texts(
    #     ["a dog on the grass"],
    #     tokenizer=tokenizer,
    #     text_model=text_model,
    #     device=device
    # )

    # similarity = torch.nn.functional.cosine_similarity(e_null, e_dog)
    # print(similarity)


if __name__ == "__main__":
    main()