import os
import time
import json
import sys

import concurrent
import numpy as np
import pandas as p d
from tqdm import tqdm
import glob
from evaluation_functions import evaluation_granular
from openai_functions import extract_ade_terms
from sentence_transformers import SentenceTransformer

def run_iteration(input_file, manual_ades, embed_model_name=None, embed_model=None):
    run_key = input_file.split('/')[-1].replace('.csv', '')
    output = pd.read_csv(input_file)

    logf = open("download.log", "a")
    names = open("filenames.log", "a")

    for eval_method in ['strict', 'lenient', 'embed']:
        try:
            if eval_method != 'embed':
                granular_save_filename = 'results/evals/{}_{}_granular.csv'.format(run_key, eval_method)
                overall_save_filename = 'results/evals/{}_{}_overall.csv'.format(run_key, eval_method)
            else:
                granular_save_filename = 'results/evals/{}_{}_granular.csv'.format(run_key.strip('/'), embed_model_name.split('/')[-1])
                overall_save_filename = 'results/evals/{}_{}_overall.csv'.format(run_key.strip('/'), embed_model_name.split('/')[-1])
    
            if os.path.isfile(overall_save_filename):
                continue
            
            results_granular = evaluation_granular(manual_ades, output, eval_method=eval_method, embed_model = embed_model)
            overall_results = results_granular.groupby(['section','ade_type'])[['tp', 'fp', 'fn']].sum(min_count = 1).reset_index()
            overall_results['micro_precision'] = overall_results['tp']/(overall_results['tp']+overall_results['fp'])
            overall_results['micro_recall'] = overall_results['tp']/(overall_results['tp']+overall_results['fn'])
            overall_results['micro_f1'] = (2 * overall_results['micro_precision'] * overall_results['micro_recall'])/(overall_results['micro_precision'] + overall_results['micro_recall']) # 2*tp_total/(2*tp_total+fp_total+fn_total)
            
            macro_results = results_granular.groupby(['section', 'ade_type'])[['precision', 'recall', 'f1']].mean(numeric_only=True).reset_index()
            overall_results['macro_precision'] = macro_results['precision']
            overall_results['macro_recall'] = macro_results['recall']
            overall_results['macro_f1'] = macro_results['f1']
        
            allsections_results = results_granular.query("section != 'all-concat'").groupby(['ade_type'])[['tp', 'fp', 'fn']].sum(min_count = 1).reset_index().query("ade_type == 'all'")
            allsections_results['micro_precision'] = allsections_results['tp']/(allsections_results['tp']+allsections_results['fp'])
            allsections_results['micro_recall'] = allsections_results['tp']/(allsections_results['tp']+allsections_results['fn'])
            allsections_results['micro_f1'] = (2 * allsections_results['micro_precision'] * allsections_results['micro_recall'])/(allsections_results['micro_precision'] + overall_results['micro_recall']) # 2*tp_total/(2*tp_total+fp_total+fn_total)
            
            allsections_macro_results = results_granular.query("section != 'all-concat'").groupby(['ade_type'])[['precision', 'recall', 'f1']].mean(numeric_only=True).reset_index().query("ade_type == 'all'")
            allsections_results['macro_precision'] = allsections_macro_results['precision']
            allsections_results['macro_recall'] = allsections_macro_results['recall']
            allsections_results['macro_f1'] = allsections_macro_results['f1']
            allsections_results['section'] = ['all']
            
            overall_results = pd.concat([overall_results, allsections_results])
            overall_results.to_csv(overall_save_filename)
            results_granular.to_csv(granular_save_filename)
        except:
            logf.write(f'Error with {input_file} and {eval_method}\n')
    
    return None

manual_file = 'data/TAC2017/train_drug_label_text_manual_ades.csv'
manual_ades = pd.read_csv(manual_file)

# embed_model_name = None
# embed_model = None

## add manual embeddings here
embed_model_name = 'llmrails/ember-v1'
embed_model = SentenceTransformer(embed_model_name)
man_embeds = embed_model.encode(manual_ades['reaction_string'].tolist())
manual_ades['embeds'] = list(man_embeds)

overall_results = glob.glob('results/extract/*.csv')


# for input_file in tqdm(overall_results):
#     run_iteration(input_file, manual_ades, embed_model = embed_model, embed_model_name = embed_model_name)
    
results = list()
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as exec:
        results.extend(list(tqdm(
            exec.map(lambda input_file: run_iteration(input_file, manual_ades, embed_model = embed_model, embed_model_name = embed_model_name),
                     overall_results), 
            total=len(overall_results)
        )))

