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
    

    return set(drug_df['reaction_string'].to_list())
    
def embed_evaluation(man_df, gpt_embeds, threshold = 0.6681796):
    TP, FN, FP = 0, 0, 0
    for (term, embed) in man_df.items():
        sims = [float(cos_sim(embed, gpt_emb)) for gpt_emb in gpt_embeds]
        try:
            if np.max(sims) > threshold:
                TP += 1
            else:
                FN += 1
        except:
            raise Exception(len(gpt_emb), man_df.keys(), len(sims))
    
    for gpt_emb in gpt_embeds:
        sims = [float(cos_sim(gpt_emb, man_emb)) for (term, man_emb) in man_df.items()]
        try:
            if np.max(sims) < threshold:
                FP += 1
        except:
            raise Exception(len(gpt_emb), man_df.keys(), len(sims))

    return TP, FN, FP

def evaluation_subtype(manual, gpt_output, drug, section='adverse reactions', subtype='all', eval_method='strict'):
    '''
    For a given drug, evaluate the performance of GPT on a given subtype of ADEs. 
    '''

    gpt_drug = (gpt_output[
        (gpt_output['drug_name'] == drug)
        &
        (gpt_output['section_name'] == section)
        ]["gpt_output"].astype(str)
        .str.lower()
        .str.replace('\n-', ', ')
        .str.replace('<negated>', '')
        .str.split(",").tolist())
    
    try:
        gpt_drug = [x.strip() for x in gpt_drug[0] if x]
        gpt_drug = set(gpt_drug)
    except:
        return [drug, section, subtype, len(manual), len(gpt_drug), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        
    if eval_method == 'strict':
        TP = len(manual.intersection(gpt_drug))
        FP = len(gpt_drug.difference(manual))
        FN = len(manual.difference(gpt_drug))
    elif eval_method == 'lenient':
        [TP, FP, FN] = common_lenient_performance(gpt_drug, manual)
    elif eval_method == 'embed':
        [TP, FP, FN] = [0,0,0]
    
    if subtype != 'all':
            # these can't be computed for the subtypes
            precision = np.nan
            f1 = np.nan
            FP = np.nan
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
    
    return [drug, section, subtype, len(manual), len(gpt_drug), TP, FP, FN, precision, recall, f1]

def evaluation_granular(manual_ades, gpt_output, eval_method='strict', embed_model=None):
    drugs = gpt_output['drug_name'].unique()
    results = []

    for drug in tqdm(drugs):
        for section in ['adverse reactions', 'warnings and precautions','boxed 
warnings', 'all-concat']:
            for subtype in ['all', 'exact-meddra', 'non-meddra', 'negated', 'discontinuous']:
                manual_data = get_manual_data(manual_ades, drug, section=section, subtype=subtype)
                gpt_data = subset_gpt_data(gpt_output, drug, section=section, subtype=subtype, eval_method=eval_method)
                if manual_data.shape == 0:
                    continue
                if eval_method == 'embed':
                    man_embeds = dict(zip(drug_df['reaction_string'], drug_df['embeds']))
                results.append(evaluation_subtype(manual_data, gpt_output, eval_method=eval_method))

    results = pd.DataFrame(results, columns=['drug_name', 'section', 'ade_type', 'n_manual', 'n_gpt', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1'])
    return results

def evaluate(outputs, manual_ades, eval_method='strict', embed_model_name=None):
    print(f"Running {eval_method} evaluation and saving results to disk.")

    for run_key, output in outputs.items():
        if eval_method != 'embed':
            granular_save_filename = 'results/{}_{}_granular.csv'.format(run_key, eval_method)
            overall_save_filename = 'results/{}_{}_overall.csv'.format(run_key, eval_method)
        else:
            granular_save_filename = 'results/{}_{}_granular.csv'.format(run_key, embed_model_name)
            overall_save_filename = 'results/{}_{}_overall.csv'.format(run_key, embed_model_name)
        
        print(run_key)
        
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
