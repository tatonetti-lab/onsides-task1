import numpy as np
import pandas as pd
from tqdm import tqdm
from common_string import common_lenient_performance
from sentence_transformers.util import cos_sim

def get_manual_data(manual_ades, drug, section='adverse reactions', subtype='all'):
    """
    Get the subset of the manual annotation appropriate for the evaluation
    """

    if not section in ('adverse reactions', 'boxed warnings', 'warnings and precautions', 'all-concat'):
        raise Exception(f"Unexpected section, {section}, provided.")
    
    if section == 'all-concat':
        drug_df = manual_ades.query("drug_name == '{}'".format(drug))
    else:
        drug_df = manual_ades.query("(drug_name == '{}') & (section_name == '{}')".format(drug, section))
    
    if subtype == 'all':
        pass
    elif subtype == 'exact-meddra': 
        drug_df = drug_df[drug_df.meddra_exact_term == 1]
    elif subtype == 'non-meddra': 
        drug_df = drug_df[drug_df.meddra_exact_term == 0]
    elif subtype == 'negated': 
        drug_df = drug_df[drug_df.negated_term == 1]
    elif subtype == 'discontinuous': 
        drug_df = drug_df[drug_df.discontinuous_term == 1]
    else:
        raise Exception(f"Unexpected subtype, {subtype}, provided.")
    

    return drug_df

def get_gpt_drug(gpt_output):
    """
    Get the subset of the GPT output appropriate for the evaluation
    """
    output = gpt_output[['drug_name', 'section_name', 'gpt_output']]
    output['gpt_output'] = output['gpt_output'].str.lower().str.replace('.', '').str.replace('\n-', ', ').str.split(', ')
    output = output.explode('gpt_output').reset_index(drop = True).drop_duplicates()
    output['gpt_output'] = output['gpt_output'].str.strip()

    return output

def embed_evaluation(man_df, gpt_embeds, threshold = 0.6681796):
    TP, FN, FP = 0, 0, 0
    for (term, embed) in man_df.items():
        sims = [float(cos_sim(embed, gpt_emb)) for (term,gpt_emb) in gpt_embeds.items()]
        try:
            if np.max(sims) > threshold:
                TP += 1
            else:
                FN += 1
        except:
            raise Exception('Error in embed_evaluation')
    
    for (term, gpt_emb) in gpt_embeds.items():
        sims = [float(cos_sim(gpt_emb, man_emb)) for (term, man_emb) in man_df.items()]
        try:
            if np.max(sims) < threshold:
                FP += 1
        except:
            raise Exception('Error in embed_evaluation')

    return TP, FN, FP

def evaluation_subtype(manual, gpt_vals, drug, section='adverse reactions', subtype='all', eval_method='strict'):
    '''
    For a given drug, evaluate the performance of GPT on a given subtype of ADEs. 
    '''
            
    if eval_method == 'strict':
        TP = len(manual.intersection(gpt_vals))
        FP = len(gpt_vals.difference(manual))
        FN = len(manual.difference(gpt_vals))
    elif eval_method == 'lenient':
        [TP, FP, FN] = common_lenient_performance(gpt_vals, manual)
    elif eval_method == 'embed':
        [TP, FP, FN] = embed_evaluation(manual, gpt_vals, threshold = 0.6681796)
    
    if subtype != 'all':
        precision = np.nan
        f1 = np.nan
        FP = np.nan
        if TP == 0 and FN == 0:
            recall = np.NAN
        else:
            recall = TP/(TP+FN)
    else:
        if TP == 0 and FP == 0:
            precision = np.NAN
        else:
            precision = TP/(TP+FP)
        if TP == 0 and FN == 0:
            recall = np.NAN
        else:
            recall = TP/(TP+FN)
        if precision != 0 and recall != 0:
            f1 = (2 * precision * recall)/(precision + recall)# 2*TP/(2*TP+FP+FN)
        else:
            f1 = np.NAN
    
    return [drug, section, subtype, len(manual), len(gpt_vals), TP, FP, FN, precision, recall, f1]

def evaluation_granular(manual_ades, gpt_output, eval_method='strict', embed_model=None):
    drugs = gpt_output['drug_name'].unique()
    results = []

    gpt_data = get_gpt_drug(gpt_output)
    if eval_method == 'embed':
        gpt_embeds = list(embed_model.encode(list(gpt_data.gpt_output)))
        gpt_data['embeds'] = gpt_embeds

    for drug in tqdm(drugs):
        for section in ['adverse reactions', 'warnings and precautions', 'boxed warnings', 'all-concat']:
            # subset gpt data
            sub_gpt = gpt_data.query("(drug_name == '{}') & (section_name == '{}')".format(drug, section))

            if sub_gpt.shape[0] == 0:
                continue
            
            if eval_method == 'embed':
                gpt_vals = dict(zip(sub_gpt['gpt_output'], sub_gpt['embeds']))
            else:
                gpt_vals = set(sub_gpt['gpt_output'])
            
            for subtype in ['all', 'exact-meddra', 'non-meddra', 'negated', 'discontinuous']:
                # get manual data
                manual_data = get_manual_data(manual_ades, drug, section=section, subtype=subtype)
                if manual_data.shape[0] == 0:
                    continue
                if eval_method == 'embed':
                    man_vals = dict(zip(manual_data['reaction_string'], manual_data['embeds']))
                else:
                    man_vals = set(manual_data['reaction_string'].to_list())

                results.append(evaluation_subtype(man_vals, gpt_vals, drug,
                                                   eval_method=eval_method, subtype=subtype, section = section))

    results = pd.DataFrame(results, columns=['drug_name', 'section', 'ade_type', 'n_manual', 'n_gpt', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1'])
    return results

def evaluate(outputs, manual_ades, eval_method='strict', embed_model_name=None, embed_model=None):
    print(f"Running {eval_method} evaluation and saving results to disk.")

    for run_key, output in outputs.items():
        if eval_method != 'embed':
            granular_save_filename = 'results/evals/{}_{}_granular.csv'.format(run_key, eval_method)
            overall_save_filename = 'results/evals/{}_{}_overall.csv'.format(run_key, eval_method)
        else:
            granular_save_filename = 'results/evals/{}_{}_granular.csv'.format(run_key.strip('/'), embed_model_name.split('/')[-1])
            overall_save_filename = 'results/evals/{}_{}_overall.csv'.format(run_key.strip('/'), embed_model_name.split('/')[-1])
        
        print(run_key)
        print(f'saving results to {granular_save_filename} and {overall_save_filename}')
        
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
