import os
import time
import json
import requests
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from evaluation_functions import evaluate
from sentence_transformers import SentenceTransformer
import bitsandbytes
from transformers import (
    AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, pipeline
)
import torch

from collections import defaultdict


##Variables
MAX_LENGTH = 5000 #maximum lenght of the input
NUM_RUNS = 1 #number of runs
TEMP = 0 #temperature for the model
MAX_TRIES =2
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


# def perform_extraction(model, prompt, text, temperature, max_length):

#     prompt = tokenize(prompt)
#     # model = codellama/CodeLlama-34b-Instruct-hf
#     # Define the payload
#     payload = {
#         "input": prompt.format(text),
#         "model_id": model,
#         "parameters": {
#             "temperature": temperature,
#             "max_length": max_length
#         }
#     }
    
#     #print(prompt.format(text))
#     # Tokenize the input
#     model_input = tokenize(prompt)

#     # Move the input to GPU
#     model_input = model_input.to("cuda")

#     # Generate the output
#     with torch.no_grad():
#         outputs = model.generate(**model_input)  # Adjusting tokens to generate minimal output
#         result = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print(result)

#     return result

pipeline1 = pipeline(
        "text-generation", model=BASE_MODEL, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto", return_full_text=False
    )


def perform_extraction(prompt, text, temperature, max_length):

    prompt = prompt.format(text)

    generate_kwargs = {
        "temperature": temperature,
        "max_new_tokens": max_length,
    }
    result = pipeline1(prompt)
    return result

def perform_cleanup(extraction, openai_api):
    client = OpenAI(api_key=openai_api)
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": """The following text is an extraction of adverse event terms from a drug label. Please remove any preamble or postamble from the list and turn the list of ADEs into a comma separated list. The text: {}""".format(extraction)
            }
        ],
        model="gpt-3.5-turbo-16k",
        temperature=0,
    )
    term = chat_completion.choices[0].message.content
    return term

def extract_ade_terms(config, prompt, text, temperature, max_length):
  extraction = perform_extraction(prompt, text, temperature, max_length)
  if extraction is None:
    raise Exception(f"perform_extraction() return None for {BASE_MODEL}")
  else:
    extraction = perform_cleanup(extraction, config['OpenAI']['openai_api_key'])
    return extraction
  

drug_file = 'data/TAC2017/train_drug_label_text.csv'
manual_file = 'data/TAC2017/train_drug_label_text_manual_ades.csv'

drugs = pd.read_csv(drug_file)
manual_ades = pd.read_csv(manual_file)
set_type = drug_file.split('/')[2].split('_')[0] # assuming file follows format "train_..." or "test...."

all_sections = drugs.query("section_name != 'all-concat'").groupby('drug_name')['section_text'].apply(' '.join).reset_index()
all_sections.insert(1, "section_name", ["all-concat" for _ in range(all_sections.shape[0])])
drugs = pd.concat([drugs, all_sections])

## Run LLama

outputs = {}

config = json.load(open('./config.json'))

# gpt_model = 'code-llama-34b'
# model_id = "codellama/CodeLlama-34b-Instruct-hf"

# model_id = "google/gemma-7b"
# model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

model_name = BASE_MODEL.split('/')[1]

max_length = MAX_LENGTH

nruns = NUM_RUNS
temperature = TEMP

system_options = {
    "no-system-prompt": "",
    "pharmexpert-v0": "You are an expert in pharmacology.",
    "pharmexpert-v1": "You are an expert in medical natural language processing, adverse drug reactions, pharmacology, and clinical trials."
}

prompt_options = {
    "fatal-prompt-v2": """
Extract all adverse reactions as they appear, including all synonyms.
mentioned in the text and provide them as a comma-separated list.
If a fatal event is listed add 'death' to the list.
The text is :'{}' 
""",
    "gpt-written-prompt-llama": """
Extract adverse drug reaction information comprehensively from a given drug label text. Ensure 
inclusivity of sentence-form expressions, reactions in tables, negated reactions, discontinuous 
mentions, hypothetical scenarios, and potentially fatal occurrences. Output the exact mentions 
of adverse reactions in a comma-separated format.
The text is :'{}'
""",
    "gpt-written-prompt-mixtral": """
Please extract adverse drug reactions (ADRs) from the provided structured product label (SPL) information. The SPL contains detailed data on potential reactions to a specific medication. Your task is to identify adverse reactions, including negated, discontinuous, hypothetical, and potentially fatal reactions. The adverse reactions can be presented in sentence form or may appear within tables.

Instructions for Mixtral:

1. Analyze the provided structured product label data thoroughly.
2. Identify adverse reactions associated with the medication mentioned in the SPL.
3. Ensure the extraction covers various scenarios:
 - Negated reactions (e.g., "no adverse reactions")
 - Discontinuous reactions (e.g., "dizziness, nausea, and vomiting")
 - Hypothetical reactions (e.g., "may cause headache")
 - Potentially fatal reactions (e.g., "risk of cardiac arrest")
4. Present the extracted adverse reactions as a comma-separated list of terms.
The text is :'{}'
"""
}

system_name = "pharmexpert-v1"
system_content = system_options[system_name]

user_prompt_name = "fatal-prompt-v2"
user_prompt = prompt_options[user_prompt_name]

gpt_params = [f"temp{temperature}"]

if model_id.split('/')[0] in ("codellama", "mistralai"):
    print("Modifying the prompt to include instruction tags.")
    prefix = ""
    prompt = f"<s>[INST] <<SYS>>\\n{system_content}\\n<</SYS>>\\n\\n{user_prompt}[/INST]{prefix}"
else:
    prompt = system_content + '\n' + user_prompt

output_file_basename = '{}_{}_{}_{}_{}'.format(model_name, user_prompt_name, system_name, '-'.join(gpt_params), set_type)
print(output_file_basename, flush=True)

max_tries = MAX_TRIES
num_tries = defaultdict(int)

#run local
for i in range(nruns):
    run_key = "{}_run{}".format(output_file_basename, i)
    print(run_key)
    
    if run_key in outputs:
        print(f"Run {run_key} already started will pick up from where it was left off.")
    elif os.path.exists('results/extract/{}.csv'.format(run_key)):
        gpt_output = pd.read_csv('results/extract/{}.csv'.format(run_key))
        outputs[run_key] = gpt_output
        print(f"Run {run_key} started, loading from disk and pick up from where it was left off.")
    
    start = time.time()
    results = list()
    for i, row in tqdm(drugs.iterrows(), total=drugs.shape[0]):

        if num_tries[(run_key,i)] >= max_tries:
            print(f"Skipping run {(run_key,i)} because we have tried it {max_tries} times.")
            continue
        
        name, section = row['drug_name'], row['section_name']

        if run_key in outputs:
            prev_run_results = outputs[run_key].query(f"drug_name == '{name}'").query(f"section_name == '{section}'")
            if prev_run_results.shape[0]==1:
                results.append([name, section, prev_run_results.gpt_output.values[0]])
                continue
        
        text = row['section_text'][:15000]
        
        if (name in ('PROLIA', 'ELIQUIS', 'INVOKANA') and section == 'adverse reactions') \
            or (name in ('PROLIA', 'ELIQUIS','INVOKANA') and section == 'all-concat'):
            text = row['section_text'][:14000]
        
        try:
            gpt_out = extract_ade_terms(config, prompt, text, temperature, max_length)
            results.append([name, section, gpt_out])    
        except Exception as err:
            num_tries[(run_key,i)] += 1
            print(f"Encountered an exception for row: '{name}' '{section}'.")
            print(f"This is the {num_tries[(run_key,i)]} time we have tried to run this. Will try {max_tries} times and then skip.")
            print(f"Will save progress, so you can restart from where we left off.")
            gpt_output = pd.DataFrame(
                [r for r in results if r is not None],
                columns=['drug_name', 'section_name', 'gpt_output']
            )
            if gpt_output.shape[0] > 0:
                print("Saved progress successfully.")
                outputs[run_key] = gpt_output
                gpt_output.to_csv('results/extract/{}.csv'.format(run_key))
            
            print(f"Failed for prompt: {prompt.format(text)}")
            raise err
            continue
    
    gpt_output = pd.DataFrame(
        [r for r in results if r is not None],
        columns=['drug_name', 'section_name', 'gpt_output']
    )
    end = time.time()
    
    if gpt_output.shape[0] > 0:
        outputs[run_key] = gpt_output
        gpt_output.to_csv('results/extract/{}.csv'.format(run_key))
    
    print(f"Run: {run_key}, time elapsed: {end-start}s.")


#EVALUATION
evaluate(outputs, manual_ades, 'strict')
evaluate(outputs, manual_ades, 'lenient')


# if using embeddings -- run this once:
# get embeddings for manual annotation --- this part is slow -- but should take <5 min
embed_model_name = 'llmrails/ember-v1'
embed_model = SentenceTransformer(embed_model_name)
man_embeds = embed_model.encode(manual_ades['reaction_string'].tolist())
manual_ades['embeds'] = list(man_embeds)


evaluate(outputs, manual_ades, 'embed', embed_model=embed_model, embed_model_name=embed_model_name)