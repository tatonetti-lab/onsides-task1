from openai import OpenAI, AzureOpenAI

def perform_extraction(api_source, config, gpt_model, system_content, prompt, text, temperature):
    if api_source == 'OpenAI':
        client = OpenAI(api_key=config[api_source]['openai_api_key'])
    elif api_source == 'Azure':
        client = AzureOpenAI(api_key=config[api_source]['openai_api_key'], api_version="2023-12-01-preview", azure_endpoint=config[api_source]['openai_api_endpoint'])
    else:
        raise Exception(f"Unexpected API source requested: {api_source}")
  
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": prompt.format(text)
            }
        ],
        model=gpt_model,
        temperature=temperature,
    )
    term = chat_completion.choices[0].message.content
    return term

def perform_cleanup(extraction, openai_api):
    client = OpenAI(api_key=openai_api)
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": """The following text is an extraction of adverse event terms from a drug label. Please remove any preamble or postamble from the list and turn the list of ADEs into a comma separated list. 
The text: {}""".format(extraction)
            }
        ],
        model="gpt-3.5-turbo-16k",
        temperature=0,
    )
    term = chat_completion.choices[0].message.content
    return term

# function for running GPT
def extract_ade_terms(api_source, config, gpt_model, system_content, prompt, text, temperature):
  extraction = perform_extraction(api_source, config, gpt_model, system_content, prompt, text, temperature)
  if extraction.find('\n') != -1:
    extraction = perform_cleanup(extraction, config['OpenAI']['openai_api_key'])
  return extraction
