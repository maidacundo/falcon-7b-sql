import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Optional
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model, PeftModel
from data.utils import SQL_SPECIAL_TOKENS

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def get_model(model_id: str, bnb_config: BitsAndBytesConfig):
    if bnb_config is None:
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    else:   
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, trust_remote_code=True)
    return model

def get_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, do_lower_case=True)
    return tokenizer

def add_special_tokens_to_tokenizer(tokenizer, SQL_SPECIAL_TOKENS):
    """
    Add special tokens to the tokenizer
    """
    
    tokenizer.add_special_tokens(
        {
            "pad_token": tokenizer.eos_token,
            "sep_token": tokenizer.eos_token,
        }
    )

    additional_special_tokens = (
        []
        if "additional_special_tokens" not in tokenizer.special_tokens_map
        else tokenizer.special_tokens_map["additional_special_tokens"]
    )

    additional_special_tokens = list(set(additional_special_tokens + list(SQL_SPECIAL_TOKENS.values())))

    num_special_tokens = tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})

    return tokenizer, num_special_tokens

def add_embeddings_to_model(model, tokenizer, SQL_SPECIAL_TOKENS):
    """
    Adds the special tokens embeddings to the model.
    Embeddings are calculated as the mean of the sub-words embeddings.
    """

    # map new tokens
    new_tokens_ids = {}

    for value in SQL_SPECIAL_TOKENS.values():
      ids = tokenizer(value)['input_ids']
      if len(ids) > 1:
        new_tokens_ids[value] = ids

    # retrieve embedding weigths
    embeddings_key = ''
    for k in model.state_dict().keys():
        if 'word_embeddings.weight' in k:
            embeddings_key = k
            break
    embeddings_weights = model.state_dict()[embeddings_key].clone()

    # calculate new tokens embeddings weigths as mean of the sub-words embeddings
    new_tokens_emb_weights = {}

    for token in new_tokens_ids.keys():
      token_emb_list = []
      for ids in new_tokens_ids[token]:
        token_emb = embeddings_weights[ids].clone()
        token_emb_list.append(token_emb)
      token_emb_tensor = torch.stack(token_emb_list)
      # mean of the sub-words embeddings
      mean_emb = torch.mean(token_emb_tensor, dim=0)
      new_tokens_emb_weights[token] = mean_emb

    # add new tokens and special tokens to the tokenizer vocab
    tokenizer, num_special_tokens = add_special_tokens_to_tokenizer(tokenizer, SQL_SPECIAL_TOKENS)

    # resize model embedding
    model.resize_token_embeddings(len(tokenizer))

    vocab_size = tokenizer.vocab_size

    new_embs = model.state_dict()[embeddings_key][
        vocab_size : vocab_size + num_special_tokens, :
    ].clone()

    # add new weights to the model.state_dict
    for token in new_tokens_emb_weights.keys():
      new_ids = tokenizer(token)['input_ids']
      model.state_dict()[embeddings_key][new_ids] = new_tokens_emb_weights[token]
    
    return new_embs

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def get_model_and_tokenizer(model_id: str, bnb_config: Optional[BitsAndBytesConfig], lora_config: LoraConfig):
    """
    Returns the model and tokenizer for the given model_id.

    model_id: The model id to load.
    bnb_config: BitsAndBytesConfig to quantize the model.
    lora_config: LoraConfig to configure the model with LoRA.
    """
    tokenizer = get_tokenizer(model_id)
    model = get_model(model_id, bnb_config)
    add_embeddings_to_model(model, tokenizer, SQL_SPECIAL_TOKENS)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    return model, tokenizer

def get_pretrained_model_and_tokenizer(model_id: str, bnb_config: Optional[BitsAndBytesConfig], lora_id: str, add_embeddings: bool = True):
    tokenizer = get_tokenizer(model_id)
    model = get_model(model_id, bnb_config)

    if add_embeddings:
        add_embeddings_to_model(model, tokenizer, SQL_SPECIAL_TOKENS)

    if lora_id:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        model = PeftModel.from_pretrained(model, lora_id, torch_dtype=torch.float16)

    return model, tokenizer