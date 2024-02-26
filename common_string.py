import numpy as np
from difflib import SequenceMatcher

def longest_common_substring(s1: str, s2: str) -> str:
    """Computes the longest common substring of s1 and s2"""
    seq_matcher = SequenceMatcher(isjunk=None, a=s1, b=s2)
    match = seq_matcher.find_longest_match(0, len(s1), 0, len(s2))

    if match.size:
        return s1[match.a : match.a + match.size]
    else:
        return ""

def longest_common_substring_percentage(s1 : str, s2 : str) -> float:
    """Computes the longest common substring percentage of s1 and s2"""
    assert min(len(s1), len(s2)) > 0, "One of the given string is empty"
    return len(longest_common_substring(s1, s2))/min(len(s1), len(s2))

def common_lenient_performance(gpt_output: list, manual_output: list) -> list:
    """
    Computes performance using lenient matching

    Input: list of distinct ADE terms from drug label using (1) GPT and (2) manual annotation
    e.g., gpt_output = results[results.drug_name == drug].gpt_output.unique()
    e.g., manual_output = manual[manual.drug_name == drug].reaction_string.unique()
    Output: TP, FP, FN, precision, recall, f1 for each drug
    """
    TP = 0
    FN = 0
    FP = 0
    for gpt_out in gpt_output:
        if gpt_out == '':
            continue
        if gpt_out in manual_output:
            TP += 1
        elif any([longest_common_substring_percentage(gpt_out, x) > 0.8 for x in manual_output]):
            TP += 1
        else:
            FP += 1

    for man_out in manual_output:
        common = [longest_common_substring_percentage(man_out, x) for x in gpt_output]
        if not any([x >= 0.8 for x in common]):
            FN += 1
    
    TP = len(manual_output) - FN
    
    return [TP, FP, FN]