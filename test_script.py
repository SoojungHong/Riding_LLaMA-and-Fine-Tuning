import torch

import transformers

from transformers import LlamaForCausalLM, LlamaTokenizer

model_dir = "./llama-2-7b-chat-hf"

model = LlamaForCausalLM.from_pretrained(model_dir)

tokenizer = LlamaTokenizer.from_pretrained(model_dir)

#pipeline = transformers.pipeline("text-generation", model=model,tokenizer=tokenizer,torch_dtype=torch.float16, device_map="auto",)
pipeline = transformers.pipeline("text-generation", model=model,tokenizer=tokenizer,torch_dtype=torch.float16, device_map="auto", pad_token_id = 50256)


sequences = pipeline('I have tomatoes, basil and cheese at home. What can I cook for dinner?\n',do_sample=True,top_k=10,num_return_sequences=1,eos_token_id=tokenizer.eos_token_id,max_length=400,)

#sequences = pipeline('What is the business sector of Chevron?\n',do_sample=True,top_k=10,num_return_sequences=1,eos_token_id=tokenizer.eos_token_id,max_length=400,)

#sequences = pipeline('What is the business sector of Chevron in one word?\n',do_sample=True,top_k=10,num_return_sequences=1,eos_token_id=tokenizer.eos_token_id,max_length=400,)

#sequences = pipeline('Write a python code to calculate pibonacci sequence?\n',do_sample=True,top_k=10,num_return_sequences=1,eos_token_id=tokenizer.eos_token_id,max_length=400,)

#sequences = pipeline('Write a python code for following task. Given login and logout times of users (in ints, e.g. [2, 10], [5, 8]) find the peak load interval. \n',do_sample=True,top_k=10,num_return_sequences=1,eos_token_id=tokenizer.eos_token_id,max_length=400,)

for seq in sequences:
    print(f"{seq['generated_text']}")
