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
        'figures/granular-adverse_reactions-strict.png',
        'figures/granular-adverse_reactions-lenient.png',
        'figures/granular-adverse_reactions-ember-v1.png',
        'figures/granular-warnings_and_precautions-strict.png',
        'figures/granular-warnings_and_precautions-lenient.png',
        'figures/granular-warnings_and_precautions-ember-v1.png',
    shell:
        """
        Rscript granular_figures.R
        Rscript granular_figures.R
        """