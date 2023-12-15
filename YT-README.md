## GPT extraction of ADE mentions in drug labels (Yutaro's notes)

- all data / code in this folder will be formatted properly and added into [github repository](https://github.com/tatonetti-lab/onsides-task1) soon. 

- files 
    - downloaded from [TAC](https://bionlp.nlm.nih.gov/tac2017adversereactions/)
        - train_xml : 101 drug labels used as training set in TAC 2017.
        - gold_xml : 99 drug labels used as test set in TAC 2017. 
        - evaluate.py : evaluation script used in TAC 2017.
    - generated
        - `train/test_drug_label_text.csv` : dataframes of drug label text extracted from `train_xml` / `gold_xml` drug labels. 
            - columns : `drug_name`, `section_name`, `section_text`
        - `train/test_drug_label_text_manual_ades.csv` : dataframes of ades extracted and MedDRA-mapped from `train_xml` / `gold_xml` drug labels. 
            - columns : `drug_name`, `reaction_string`, `meddra_pt`, `meddra_pt_id`, `meddra_llt`, `meddra_llt_id`
        - `train/test_drug_label_gpt35/4_output.csv` : dataframes of output from GPT-3.5/4 extraction. 
            - columns : `drug_name`, `section_name`, `gpt35/4_output`
        - `test/test_drug_label_gpt35/4_ade_terms.csv` : from the extraction output, we find individual terms and map them to MedDRA IDs. 
    - other
        - `umls_meddra_en.csv` : MedDRA vocabulary downloaded from [NIH UMLS](https://www.nlm.nih.gov/research/umls/index.html).

- code
    - `1.extract_data.ipynb` : for ease of use, we build a dataframe of drug label text extracted from the 200 drug labels.
        - input : `train/gold_xml`
        - output : `train/test_drug_label_text.csv`
    - `2.run_gpt.ipynb` : we run gpt3.5 / gpt4 on each row of the extracted dataframe.
        - input : `train/test_drug_label_text.csv`
        - output : `train/test_drug_label_gpt35/4_output.csv`
        - notes : 
            - we currently use `gpt-3.5-turbo-1106` and `gpt-4`. different gpt model versions may be better (refer to [OpenAI](https://platform.openai.com/docs/models/overview), especially for longer text `gpt-4-32k` etc. )
            - the current prompt used is 
                `{"role": "system", "content": "You are an expert in pharmacology."},`
                `{"role": "user", "content": "The following is a excerpt of a drug label. Your task is to find the adverse drug event terms in this text. Return only the terms as a comma-separated list. The input is {}".format(text)}`
            - better prompts may be to mention that you're looking for MedDRA terms specifically. (have not tried)
    - `3.map_to_meddra.ipynb` : we map the extracted terms to meddra vocabulary (if the extracted term is not found in MedDRA, leave ID as `None`).
        - input : `train/test_drug_label_gpt35/4_output.csv`
        - output : `test/test_drug_label_gpt35/4_ade_terms.csv`