import pandas as pd
import numpy as np
import glob

def summ_results(results):
    """
    Function to summarize results of the extracted ADRs from drug labels.
    Result files can either be overall results or granular results.
    """
    all_data = []
    for label in results:
        [api_source, llm_model, prompt, system_prompt, temp, dataset, run, eval_method] = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        relevant_info = label.split('/')[-1].split('_')
        if relevant_info[0] == 'deepcadrme': # ['deepcadrme', '010', 'test', 'strict', 'overall']
            [llm_model, run, dataset, eval_method] = relevant_info[0:4]
            basename = llm_model
        elif relevant_info[0] == 'exact': # ['exact', 'train', 'lenient', 'overall']
            [llm_model, dataset, eval_method] = relevant_info[0:3]
            basename = '_'.join([llm_model])
        elif relevant_info[0] in ['code-llama-34b', 'Meta-Llama-3-8B-Instruct', 'CodeLlama-34b-Instruct-hf']: 
            #['code-llama-34b', 'fatal-prompt-v2', 'pharmexpert-v1', 'temp0', 'train', 'run0', 'strict']
            [llm_model, prompt, system_prompt, temp, dataset, run, eval_method] = relevant_info[:-1]
            basename = '_'.join([llm_model, prompt, system_prompt])
        elif relevant_info[0] == 'Mixtral-8x7B-Instruct-v0.1':
            # ['Mixtral-8x7B-Instruct-v0.1', 'fatal-prompt-v2', 'pharmexpert-v0', 'temp0', 'train', 'run0', 'lenient', 'overall.csv']
            [llm_model, prompt, system_prompt, temp, dataset, run, eval_method] = relevant_info[:-1]
            basename = '_'.join([llm_model, prompt, system_prompt])
        else: 
            # ['OpenAI', 'gpt-4-1106-preview', 'fatal-prompt-v2', 'pharmexpert-v1', 'temp0', 'train', 'run3', 'lenient', 'overall.csv']
            # , 'gpt-written-prompt-llama', 'pharmexpert-v1', 'temp0', 'train', 'run0', 'lenient', 'granular.csv']

            try:
                [api_source, llm_model, prompt, system_prompt, temp, dataset, run, eval_method] = relevant_info[:-1]
                basename = '_'.join([llm_model, prompt, system_prompt])
            except:
                print(relevant_info)
                print(label)
                continue
    
        with open(label, 'r') as f:
            data = pd.read_csv(f)
            data['api_source'] = api_source
            data['llm_model'] = llm_model
            data['prompt'] = prompt
            data['system'] = system_prompt
            data['temp'] = temp
            data['dataset'] = dataset
            data['run'] = run
            data['eval_method'] = eval_method
            data['base_name'] = basename
            all_data.append(data)

    return all_data

#### OVERALL RESULTS ####
# get all overall results
overall_results = glob.glob('results/evals/*_overall.csv')
overall_results[0].split('/')[-1]

## create csv file with summary of results
overall_summary = summ_results(overall_results)
all_overall = pd.concat(overall_summary, axis=0, ignore_index=True)
all_overall.to_csv('results/agg_evals/overall_results_across_models.csv',
                    index=False)

#### GRANULAR RESULTS ####
# get granular results
granular_results = glob.glob('results/evals/*_granular.csv')
granular_results[0].split('/')[-1]

## create csv file with summary of results
granular_summary = summ_results(granular_results)
all_gran_results = pd.concat(granular_summary, axis=0, ignore_index=True)
all_gran_results.to_csv('results/agg_evals/granular_results_across_models.csv'
                         index=False)

