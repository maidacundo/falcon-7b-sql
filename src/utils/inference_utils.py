import transformers
import torch
from tqdm import tqdm
from transformers import StoppingCriteriaList, StoppingCriteria

STOP_WORDS = [';', ');', '\';', '";']

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False

def get_stopping_criteria(tokenizer):
    stop_ids = [tokenizer.encode(w)[0] for w in STOP_WORDS]
    keyword_criteria = KeywordsStoppingCriteria(stop_ids)
    stopping_criteria = StoppingCriteriaList([keyword_criteria])
    return stopping_criteria

def get_pipeline(model, tokenizer):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    return pipeline

def generate(model, tokenizer, inference_dataloader, max_length=512, max_new_tokens=20, limit_generation=None):  
    stopping_criteria = get_stopping_criteria(tokenizer)
    results = []

    generation_config = model.generation_config
    generation_config.max_new_tokens = max_new_tokens
    generation_config.tempeture = 0 
    generation_config.num_return_sequences = 1
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.max_length = max_length
    
    
    for batch in tqdm(inference_dataloader):
        encoding = tokenizer(batch, return_tensors="pt")
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                generation_config=generation_config,
                stopping_criteria=stopping_criteria,
                do_sample=False,
                use_cache=True,
            )
        for output in outputs:
            output = tokenizer.decode(output).split('<|sql|>')[-1]
            results.append(output)
        if limit_generation is not None and len(results) >= limit_generation:
            break
    return results

def generate_pipeline(pipeline, 
                      inference_dataloader, 
                      eos_token_id,
                      pad_token_id,
                      limit_generation=None,
                      max_new_tokens=20,
                      ):
    results = []
    for batch in tqdm(inference_dataloader):
        out = pipeline(batch,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        temperature=0.2,
                        top_k=3,
                        top_p=0.9,
                        repetition_penalty=1.2,
                        num_return_sequences=1,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                        )
        for res in out:
            prediction = res[0]['generated_text'].split('<|sql|>')[-1]
            results.append(prediction)

        if limit_generation is not None and len(results) >= limit_generation:
            break
    return results
    