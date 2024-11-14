import os
# Ensure the HF_HOME environment variable points to your desired cache location
os.environ["HF_TOKEN"] = "Your HF token"
cache_dir =  'Your cache directory'
os.environ['HF_HOME'] = cache_dir

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from extended_watermark_processor import WatermarkLogitsProcessor
import json

base_model = "Qwen/Qwen2.5-7B" # Choose the model you want to use

# Initialize the tokenizer and model 
tokenizer = AutoTokenizer.from_pretrained(base_model) 
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = AutoModelForCausalLM.from_pretrained(base_model, cache_dir=cache_dir)
model = model.to(device)
model.eval()

# (1) Initialize WatermarkLogitsProcessor
watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                               gamma=0.25,
                                               delta=2.0,
                                               seeding_scheme="selfhash")

# Load your data file
data_path = "c4_prompt_test.pt"
data = torch.load(data_path)

# Initialize an empty list to store generated outputs
generated_data = []

# Counter to keep track of processed inputs
input_counter = 0
start_time = time.time()
saving_freq = 10
max_inp = 200 # Number tokens for prompt input
max_out = 200 # Number of tokens to generate
Ninputs =2000 # Number of data points to process

# Run the model on the data
for idx, text in enumerate(data[0][:Ninputs]):
    print(f"Processing input {idx+1} of {Ninputs}")
    # Encode the first 200 tokens of each text
    prompt_tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=max_inp).to(device)
    prompt_tokens = prompt_tokens['input_ids'].cuda()
    generated_prompt = tokenizer.batch_decode(prompt_tokens, skip_special_tokens=True)[0]
    # Initialize LogitsProcessorList with your watermark_processor
    logits_processor_list = LogitsProcessorList([watermark_processor]) 
    # Generate the next 200 tokens
    outputs = model.generate(
        prompt_tokens,
        max_new_tokens=max_out,  # Generate next 200 tokens
        logits_processor=logits_processor_list,  # (3) Add this parameter to the output
        pad_token_id=tokenizer.eos_token_id
    )
    # Decode the generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_output_text = tokenizer.decode(outputs[0][max_inp:max_inp+max_out], skip_special_tokens=True)

    # Store the input and output in a dictionary
    data_dict = {
        "input": generated_prompt,  # Store only the first 200 tokens of input
        "input_output": generated_text,
        "output_only": generated_output_text,
        "label": 1,
        "train": 0, # 1 for training, 0 for testing
    }
    # Append the dictionary to the list of generated data
    generated_data.append(data_dict)
    # Increment input counter
    input_counter += 1
    # Save the results after processing every 50 inputs
    if input_counter % saving_freq == 0:
        # Check if the file exits
        if os.path.isfile("KGW_qwen_test_" + str(input_counter-saving_freq) +  ".json"):
            os.remove("KGW_qwen_test_" + str(input_counter-saving_freq) +  ".json")
        with open("KGW_qwen_test_" + str(input_counter) +  ".json", "w") as json_file:
            json.dump(generated_data, json_file, indent=4)


# End time
end_time = time.time()
# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Total time taken: {elapsed_time} seconds")