# onsides-task1
Evaluating the ability of generative models to extract adverse reaction terms from structured product labels (SPLs).

**data/**
  - TAC2017/
    - downloaded from [TAC](https://bionlp.nlm.nih.gov/tac2017adversereactions/)
        - train_xml/ : 101 drug labels used as training set in TAC 2017.
        - gold_xml/ : 99 drug labels used as test set in TAC 2017. 
    - generated files:
        - `{train/test}_drug_label_text.csv` : dataframes of drug label text extracted from `train_xml` / `gold_xml` drug labels. 
            - columns : `drug_name`, `section_name`, `section_text`
        - `{train/text}_drug_label_text_subsections.csv` : dataframes of drug label text split into subsections from the `train/test_drug_label_text.csv` files.
            - columns : `drug_name`, `section_name`, `subsection_name`, `subsection_text`
        - `{train/test}_drug_label_text_manual_ades.csv` : dataframes of ades extracted and MedDRA-mapped from `train_xml` / `gold_xml` drug labels. 
            - columns : `drug_name`, `reaction_string`, `meddra_pt`, `meddra_pt_id`, `meddra_llt`, `meddra_llt_id`,
        - `{train/test}_drug_label_mentions.csv` : dataframe containing the manually extracted reaction terms with start index and length
            - columns: `drug_name`, `id`, `section`, `type`, `start`, `len`, `str`
        - `{train/test}_drug_label_label_reactions.csv` : datafram containing the manually extracted string and meddra mappings
          - columns: `drug_name`, `reaction_id`, `str`, `norm_id`, `meddra_pt`, `meddra_pt_id`, `meddra_llt`, `meddra_llt_id`, `flag`
        - `umls_meddra_en.csv` : MedDRA vocabulary downloaded from [NIH UMLS](https://www.nlm.nih.gov/research/umls/index.html).
  - DeepCADREME - see https://www.sciencedirect.com/science/article/pii/S016786552030444X 


**results/**
- agg_evals/: aggregated results
- evals: granular and overall results for each model and parameter set
- extract: the extracted terms for each model and parameter set
  - {model}\_{user_prompt}\_{system_prompt}\_{other_params}\_{dataset}_{run}.csv
    - columns: `drug_name`, `section_name`, `gpt_output` 

**onsides-task1/**
- code
    - `1.extract_data.ipynb` : for ease of use, we build a dataframe of drug label text extracted from the 200 drug labels.
        - input : `train/gold_xml`
        - output : `train/test_drug_label_text.csv`, `train/test_drug_label_text_manual_ades.csv`
    - `2.run_gpt.ipynb` : we run gpt3.5 / gpt4 on each row of the extracted dataframe.
        - input : `train/test_drug_label_text.csv`
        - output : `train/test_drug_label_gpt35/4_output.csv`
        - notes : 
            - the current format of the prompts are:
                `{"role": "system", "content": system_prompt},`
                `{"role": "user", "content": "{prompt}: {}".format(drug_label)}`
    - `3.map_to_meddra.ipynb` : we map the extracted terms to meddra vocabulary (if the extracted term is not found in MedDRA, leave ID as `None`).
        - input : `train/test_drug_label_gpt35/4_output.csv`
        - output : `test/test_drug_label_gpt35/4_ade_terms.csv`
    - `11.parse_data.ipynb` : experimenting with different splitting of the data to allow for less token input. right now we try and split it into smaller subsections. 
        - input : `train/test_drug_label_text.csv`
        - output : `train/test_drug_label_text_subsections.csv`
    - `GPT_run_and_eval.ipynb` : the main file where we run and evaluate the openAI model performance in extracting ADRS
    - `common_string.py` : used for calculating the longest common substring for lenient matching
    - `evaluation_functions.py` : functions used for evaluating the models performance using three methods (1) strict exact matching, (2) lenient lexical matching (longest common substring), and (3) semantic matching using cosine similarity of embedded terms.
    - `openai_functions.py` : update this file to change the format of the GPT prompts
