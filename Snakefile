dirname = 'results'

rule summarize_results:
    input:
        expand("{dirname}/evals/", dirname=dirname)
    output:
        'results/agg_evals/overall_results_across_models.csv',
        'results/agg_evals/granular_results_across_models.csv'
    shell:
        """
        python summarize_results.py
        """

rule update_figures:
    input:
        'results/agg_evals/overall_results_across_models.csv',
        'results/agg_evals/granular_results_across_models.csv'
    output:
        'figures/gpt-4_performance_ade-type.png',
        'figures/granular_compare_extraction.png',
        'figures/granular_embed_compare_extraction.png',
        'figures/granular_exact_compare_extraction.png',
        'figures/granular_performance_label_length.png',
        'figures/granular_performance_label_length_embed.png',
        'figures/section_performance.png'
    shell:
        """
        
        """