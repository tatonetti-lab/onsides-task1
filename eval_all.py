import polars as pl
import pandas as pd
from sentence_transformers import SentenceTransformer
import sklearn
import os

def get_manual_embeds(embeds_file, manual_ades):
    if os.path.isfile(embeds_file):
        return pd.read_csv(embeds_file, index_col=0)
    else:
        manual_annots = (
            manual_ades
            .select(pl.col('reaction_string'))
            .unique()
            ['reaction_string'].to_list()
        )
        embed_model = SentenceTransformer('llmrails/ember-v1')

        manual_embeddings = embed_model.encode(manual_annots)
        manual_embeddings_df = pd.DataFrame(manual_embeddings, index=manual_annots)
        manual_embeddings_df.to_csv("data/embeds/manual_train_embeds.csv")
        return manual_embeddings_df

def compute_cosine_df(**kwargs):
    assert len(kwargs) == 2
    x_name, y_name = list(kwargs.keys())
    X = kwargs[x_name]
    Y = kwargs[y_name]
    
    return (
        pd.DataFrame(
            sklearn.metrics.pairwise.cosine_similarity(X, Y),
            index=X.index,
            columns=Y.index,
        )
        .melt(ignore_index=False, var_name=y_name, value_name="cosine_similarity")
        .rename_axis(index=x_name)
        .reset_index()
        .pipe(pl.DataFrame)
    )


def main():

    COSINE_THRESHOLD = 0.7097613
    LENIENT_THRESHOLD = 0.8
    eval_method = 'strict'
    llm_file = 'results/extract/OpenAI_gpt-4-1106-preview_fatal-prompt-v2_pharmexpert-v0_temp0_train_run0.csv'
    llm_results = pl.read_csv(llm_file)
    llm_results = (
        llm_results
        .drop_nulls('gpt_output')
        .select(
            'drug_name',
            'section_name',
            (
                pl.col('gpt_output')
                .str.split(',')
                .list.eval(pl.element().str.strip_chars().str.to_lowercase())
            )
        )
        .explode('gpt_output')
        .filter(pl.col('gpt_output').ne('""') & pl.col('gpt_output').str.len_chars().ne(0))
    )
    print(llm_results.head())
    manual_ades = pl.read_csv('data/TAC2017/train_drug_label_text_manual_ades.csv')
    
    if eval_method == 'ember-v1':
        llm_embeds = get_llm_embeds(llm_results, llm_file)
        embeds_file = 'data/embeds/manual_train_embeds.csv'
        manual_embeds = get_manual_embeds(embeds_file, manual_ades)
        manual_vs_gpt_cosine_df = compute_cosine_df(reaction_string=manual_embeds, gpt_output=gpt_embeddings_df)

    if eval_method == 'strict':

    # ['drug_name', 'section_id', 'reaction_string', 'meddra_pt', 'meddra_pt_id', 'meddra_llt', 'meddra_llt_id',
    # 'section_name', 'section', 'str', 'discontinuous_term', 'negated_term', 'hypothetical_term', 'meddra_exact_term']




    # embed_model = SentenceTransformer('llmrails/ember-v1')


if __name__ == '__main__':
    main()