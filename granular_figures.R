library(tidyverse)
library(ggplot2)
library(cowplot)

##################################################
## Create Granular Performance figures
##################################################

get_top_performin <- function(data, section, ade_type, eval_method) {
        
  results <- data %>%
            filter(section == !!section,
                   ade_type == !!ade_type,
                   eval_method == !!eval_method) %>%
            group_by(llm_model_name, api_source, prompt, system, temp,
                     dataset, run) %>%
            mutate(mean_f1 = mean(f1, na.rm = T)) %>% 
            ungroup() %>% group_by(llm_model_name) %>%
            filter(mean_f1 == max(mean_f1)) %>% ungroup() %>% 
            select(llm_model_name, drug_name, precision, recall, f1) %>%
            pivot_longer(precision:f1, names_to = 'metric', values_to = 'value') %>%
            mutate(metric = metric %>% str_to_sentence)

  return (results)
}

# Unnamed: 0,drug_name,section,ade_type,n_manual,n_gpt,tp,fp,fn,precision,recall,
# f1,api_source,llm_model,llm_model_name,prompt,system,temp,dataset,run,eval_method,base_name


# get granular results i.e. performance per drug
granular <- read_csv('results/agg_evals/granular_results_across_models.csv',
                     show_col_types = FALSE)
sections <- c('adverse_reactions', 'warnings_and_precautions')
section = 'adverse_reactions'
ade_type = 'all'
eval_methods = c('strict', 'lenient', 'ember-v1')

# get top performing parameters per llm_model_name

for (section in sections) {
  for (eval_method in eval_methods) {
    print(eval_method)
    this_section = gsub(pattern = '_', replacement = ' ', x = section)
    top_performs <- get_top_performin(granular,
                                      this_section,
                                      'all',
                                      eval_method)
    top_performs %>% dim() %>% print
    
    top_performs %>% 
      ggplot(mapping = aes(x = llm_model_name, y = value,
                           color = llm_model_name, fill = llm_model_name)) +
      geom_jitter(alpha = 0.5) +
      geom_boxplot(color = 'white', aes(ymin=..lower.., ymax=..upper..),
                   outlier.shape = NA, alpha = 0.8) +
      geom_boxplot(colour = 'white', outlier.shape = NA) +
      facet_wrap(~metric, nrow = 1) +
      theme_cowplot() +
      xlab('') +
      ylab('') + 
      labs(fill = 'Model', color = 'Model') +
      # scale_fill_manual(values=c("#1e4620", "#3E0C48", "#33447D", "#00837F",
      #                            "#44C15E", "#00A5F0", 'yellow')) + 
      # scale_color_manual(values=c("#1e4620", "#3E0C48", "#33447D", "#00837F",
      # "#44C15E", "#00A5F0", 'yellow')) + # "#56B4E9")) + 
      theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
            strip.background = element_rect(fill="white"),
            legend.title = element_blank(),
            legend.position = '') +
      panel_border()
    
    png_name = str_glue('./figures/granular-{section}-{eval_method}.png')
    print(png_name)
    ggsave(png_name,
           dpi = 1200,
           width = 8, height = 6, bg = 'white')
  }
}