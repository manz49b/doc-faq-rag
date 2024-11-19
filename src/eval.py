import re
import json
import random
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge import Rouge
import nltk

# Download nltk components
# (you need to uncomment this the first time you run (if you have issues loading packages, it is best to include download_dir as your env location path)
# nltk.download('punkt', download_dir='/opt/homebrew/Caskroom/mambaforge/base/envs/doc-faq-rag/lib/nltk_data')
# nltk.download('punkt_tab', download_dir='/opt/homebrew/Caskroom/mambaforge/base/envs/doc-faq-rag/lib/nltk_data')
# nltk.download('stopwords', download_dir='/opt/homebrew/Caskroom/mambaforge/base/envs/doc-faq-rag/lib/nltk_data')
# nltk.download('wordnet', download_dir='/opt/homebrew/Caskroom/mambaforge/base/envs/doc-faq-rag/lib/nltk_data')

def evaluate_gt_containment(gt_terms, generated_answer):
    """
    Checks if any terms from the ground truth (gt_terms) appear in the generated answer.

    Args:
    gt_terms (list of str): A list of ground truth terms/phrases.
    generated_answer (str): The generated answer string.

    Returns:
    bool: True if any terms from gt_terms are contained in the generated answer, False otherwise.
    """
    if isinstance(generated_answer, np.ndarray):
        generated_answer = " ".join(generated_answer.astype(str)) 
    generated_answer = str(generated_answer).lower()  

    for term in gt_terms:
        if len(term) > 2 and term.lower() in generated_answer:
            return True  # Match found, return True
    
    return False  # No match found for any term longer than 2 characters

def apply_gt_containment_evaluation(df, expected_col, generated_col):
    """
    Applies ground truth containment evaluation to a DataFrame's expected and generated columns.
    
    Args:
    df (pd.DataFrame): The dataframe containing the answers.
    expected_col (str): Column name for expected answers.
    generated_col (str): Column name for generated answers.
    
    Returns:
    pd.DataFrame: Updated dataframe with ground truth containment results.
    """
    df['llm_gt_containment'] = df.apply(
        lambda x: evaluate_gt_containment(x[expected_col], str(x[generated_col])), axis=1)
    return df

def tokenize_answer(answer):
    """
    Tokenizes a given answer into a list of unique words, splitting by non-word characters (including punctuation and numbers).

    The function processes the input string by converting it to lowercase and splitting it into tokens based on non-alphanumeric 
    characters (e.g., spaces, punctuation marks, digits). The result is a set of unique tokens.

    Args:
        answer (str): The answer (text) to be tokenized.

    Returns:
        list: A list of unique tokens (words) derived from the input string, excluding punctuation and numbers.

    Example:
        tokenize_answer("Hello, world! It's 2024.")
        ['hello', 'world', 'it', 's']
    """
    return list(set(re.split(r'\W+|\d+', answer.lower())))

def calculate_token_overlap(expected, generated):
    """
    Calculates token-based overlap between two text strings by computing precision, recall, and F1 score.

    This function tokenizes both the expected (reference) and generated (predicted) answers, then calculates:
    - Precision: The proportion of overlapping tokens in the generated answer relative to the total tokens in the generated answer.
    - Recall: The proportion of overlapping tokens in the generated answer relative to the total tokens in the expected answer.
    - F1 Score: The harmonic mean of precision and recall, which balances both metrics.

    Args:
        expected (str): The expected (reference) answer.
        generated (str): The generated (predicted) answer.

    Returns:
        tuple: A tuple containing the precision, recall, and F1 score (in that order).

    Example:
        >>> calculate_token_overlap("The cat sat on the mat", "The cat sat on the rug")
        (0.75, 0.75, 0.75)
    """
    expected_tokens = tokenize_answer(expected)
    generated_tokens = tokenize_answer(generated)

    intersecting_tokens = set(expected_tokens) & set(generated_tokens)
    
    recall = len(intersecting_tokens) / len(set(expected_tokens)) if expected_tokens else 0
    precision = len(intersecting_tokens) / len(set(generated_tokens)) if generated_tokens else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    
    return precision, recall, f1_score

def evaluate_jaccard_similarity(expected, generated):
    """
    Computes the Jaccard similarity between two text strings.

    The Jaccard similarity is a measure of similarity between two sets, defined as the 
    size of the intersection divided by the size of the union of the sets.

    Args:
        expected (str): The expected string (reference text).
        generated (str): The generated string (text to compare against the expected).

    Returns:
        float: The Jaccard similarity coefficient, ranging from 0 (no similarity) to 1 (identical).

    Example:
        >>> evaluate_jaccard_similarity("the cat sat on the mat", "the cat sat on the rug")
        0.75
    """
    expected_set, generated_set = set(expected.split()), set(generated.split())
    return len(expected_set.intersection(generated_set)) / len(expected_set.union(generated_set))

def tokenize_output(text):
    """ 
    Tokenizes the changes and release notes for BLEU (Bilingual Evaluation Understudy) score calculation. 
    The function returns a list of individual tokens (words), which is more suited for BLEU score calculation, 
    as BLEU relies on n-gram matching at the word level, especially for machine translation or sentence generation tasks.
    
    Args:
    output (dict): The output dictionary containing 'changes' and 'release_notes' keys.
    
    Returns:
    list: A list of individual tokens from the changes and release notes.
    """
    return word_tokenize(text.lower())

def evaluate_bleu(expected, generated):
    """
    Evaluates the BLEU score (precision) between the expected and generated outputs.

    The BLEU score is a measure of how similar the generated output is to the expected output based on n-gram precision.
    This implementation uses smoothing to handle cases where there are zero n-gram overlaps.

    Args:
        expected (str): The expected output text (reference).
        generated (str): The generated output text (candidate).

    Returns:
        float: The BLEU score (between 0 and 1) indicating the precision of the generated output relative to the expected output.

    Example:
        evaluate_bleu("The quick brown fox jumps over the lazy dog.", "The fast brown fox leaps over the lazy dog.")
        0.45
    """
    expected_tokens = tokenize_answer(expected)
    generated_tokens = tokenize_answer(generated)

    # Apply smoothing to handle cases with zero n-gram overlaps
    smooth_fn = SmoothingFunction().method1
    score = sentence_bleu([expected_tokens], generated_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn)
    return score

def evaluate_rouge(expected, generated):
    """
    Evaluates the ROUGE scores (recall, precision, F1) between the expected and generated outputs.

    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is commonly used to evaluate the quality of machine-generated text
    by comparing it to a reference (expected) text. This function computes the ROUGE-1, ROUGE-2, and ROUGE-L scores.

    Args:
        expected (str): The expected (reference) output text.
        generated (str): The generated output text (candidate).

    Returns:
        tuple: A tuple containing the following ROUGE scores:
            - ROUGE-1 recall, precision, and F1 score
            - ROUGE-2 recall, precision, and F1 score
            - ROUGE-L recall, precision, and F1 score

    Example:
        >>> evaluate_rouge("The quick brown fox jumps over the lazy dog.", "The fast brown fox leaps over the lazy dog.")
        (0.75, 0.80, 0.77, 0.60, 0.70, 0.65, 0.85, 0.90, 0.87)
    """
    rouge = Rouge()
    scores = rouge.get_scores(expected, generated, avg=True)
    
    r1_recall = scores['rouge-1']['r']
    r1_precision = scores['rouge-1']['p']
    r1_f1 = scores['rouge-1']['f']
    r2_recall = scores['rouge-2']['r']
    r2_precision = scores['rouge-2']['p']
    r2_f1 = scores['rouge-2']['f']
    rl_recall = scores['rouge-l']['r']
    rl_precision = scores['rouge-l']['p']
    rl_f1 = scores['rouge-l']['f']
    
    return r1_recall, r1_precision, r1_f1, r2_recall, r2_precision, r2_f1, rl_recall, rl_precision, rl_f1


def evaluate_exact_match(expected, generated):
    """
    Removed from main tool - Evaluates exact match between expected and generated outputs.

    Args:
    expected (dict): The expected output dictionary.
    generated (dict): The generated output dictionary.

    Returns:
    bool: True if the expected and generated outputs match exactly, False otherwise.
    """
    return expected == generated

def generate_ngrams(text, n=1):
    """
    Generates n-grams (unigrams, bigrams, trigrams, etc.) from a given text.

    The function processes the input text, removes punctuation, converts it to lowercase, 
    and then generates the requested n-grams using the `CountVectorizer` from scikit-learn.

    Args:
        text (str): The input text from which n-grams will be generated.
        n (int, optional): The size of the n-grams to generate (default is 1 for unigrams).

    Returns:
        list: A list of n-grams as strings, or an empty list if the input text is invalid 
              (e.g., empty, contains only "answer is impossible.", or contains only whitespace).
    
    Example:
        generate_ngrams("The quick brown fox", n=2)
        ['quick brown', 'brown fox']
    """
    text = re.sub(r"[^\w\s]", "", text).lower()
    if not text or text == 'answer is impossible.' or not text.strip():  # Handle empty or invalid text
        return []  # Return empty if text is invalid
    # Create n-grams using CountVectorizer
    vectorizer = CountVectorizer(ngram_range=(n, n), analyzer='word', stop_words='english')
    try:
        # Fit the vectorizer to the text and transform it
        ngrams_matrix = vectorizer.fit_transform([text])
        # Get the list of n-grams (features)
        ngram_list = vectorizer.get_feature_names_out()
        return ngram_list
    except ValueError:
        return []  # Handle empty vocabulary error
    
def match_fuzzy_ngram(gt_answers, generated_answer, ngram_type='unigram', threshold=80):
    """
    Matches fuzzy n-grams from the generated answer against one or more ground truth answers.

    This function compares n-grams (unigrams, bigrams, or trigrams) between the generated answer 
    and the ground truth answers, using fuzzy matching to allow for minor differences. It returns 
    the proportion of ground truth answers that have at least one fuzzy n-gram match above the given threshold.

    Args:
        gt_answers (list, str, np.ndarray, pd.Series): The list of ground truth answers (or a single string).
        generated_answer (str): The generated answer to be compared against the ground truth answers.
        ngram_type (str, optional): The type of n-grams to use ('unigram', 'bigram', 'trigram', default is 'unigram').
        threshold (int, optional): The fuzzy matching threshold for considering a match (default is 80).

    Returns:
        float: The proportion of ground truth answers that have at least one fuzzy n-gram match with the generated answer.

    Example:
        match_fuzzy_ngram(["The fox jumped over the fence.", "A quick brown fox."], "A quick fox jumped over.", ngram_type='bigram', threshold=80)
        1.0  # Both ground truth answers have a matching bigram with the generated answer
    """
    matches = 0
    if isinstance(gt_answers, str):
        gt_answers = [gt_answers]  # Convert single string to a list
    elif isinstance(gt_answers, np.ndarray) or isinstance(gt_answers, pd.Series):
        gt_answers = gt_answers.tolist()  # Convert array or series to a list

    if ngram_type == 'unigram':
        ngrams = generate_ngrams(generated_answer, 1)
    elif ngram_type == 'bigram':
        ngrams = generate_ngrams(generated_answer, 2)
    elif ngram_type == 'trigram':
        ngrams = generate_ngrams(generated_answer, 3)

    if isinstance(ngrams, (list, np.ndarray)) and len(ngrams) == 0:
        return 0  

    for gt_answer in gt_answers:
        match_found = False
        gt_clean = re.sub(r"[^\w\s]", "", gt_answer).lower()

        for ngram in ngrams:
            similarity = fuzz.partial_ratio(gt_clean, ngram)
            if similarity >= threshold:
                match_found = True
                break  # We only need one fuzzy match for this gt_answer

        if match_found:
            matches += 1
        # else: # uncomment for debugging
        #     print(f"The term {gt_answer} was not found in generated {generated_answer}")

    if len(gt_answers) > 0:
        return matches / len(gt_answers)
    else:
        return 0

def fuzzmatch_llm_scoring(is_possible, threshold=80):
    """
    Evaluates and calculates the fuzzy match proportions of unigrams, bigrams, and trigrams 
    between ground truth answers and generated answers. The matching is done using fuzzy 
    string matching, where the similarity is measured against a specified threshold.

    This function applies the `match_fuzzy_ngram` function to each row in the input DataFrame 
    to calculate match proportions for unigrams, bigrams, and trigrams. It then computes 
    and prints the average match percentages for each n-gram type.

    Args:
        is_possible (pandas.DataFrame): A DataFrame containing columns 'gt_answers' (ground truth answers) 
                                        and 'answer' (generated answers). The function adds new columns 
                                        for unigram, bigram, and trigram match proportions.
        threshold (int, optional): The fuzzy matching threshold for considering a match (default is 80). 
                                   A higher threshold requires greater similarity to count as a match.

    Returns:
        pandas.DataFrame: The input DataFrame with added columns for unigram, bigram, and trigram match proportions 
                           (columns: 'llm_unigram_match', 'llm_bigram_match', 'llm_trigram_match').

    Example:
        fuzzmatch_llm_scoring(df, threshold=85)
        Average Unigram Match: 92%
        Average Bigram Match: 88%
        Average Trigram Match: 85%
    """
    is_possible['llm_unigram_match'] = is_possible.apply(lambda row: match_fuzzy_ngram(row['gt_answers'], row['answer'], ngram_type='unigram', threshold=threshold), axis=1)
    is_possible['llm_bigram_match'] = is_possible.apply(lambda row: match_fuzzy_ngram(row['gt_answers'], row['answer'], ngram_type='bigram', threshold=threshold), axis=1)
    is_possible['llm_trigram_match'] = is_possible.apply(lambda row: match_fuzzy_ngram(row['gt_answers'], row['answer'], ngram_type='trigram', threshold=threshold), axis=1)

    unigram_match_avg = is_possible['llm_unigram_match'].mean() * 100
    bigram_match_avg = is_possible['llm_bigram_match'].mean() * 100
    trigram_match_avg = is_possible['llm_trigram_match'].mean() * 100

    print(f"Average Unigram Match: {unigram_match_avg:.0f}%")
    print(f"Average Bigram Match: {bigram_match_avg:.0f}%")
    print(f"Average Trigram Match: {trigram_match_avg:.0f}%")
    return is_possible

def evaluate_llm_response(expected, generated):
    """
    Evaluates the generated response against the expected output using multiple evaluation metrics, 
    including token overlap, ROUGE scores, and F1 score.

    This function computes the following:
    - Precision, recall, and F1 score based on token overlap.
    - ROUGE-1, ROUGE-2, and ROUGE-L recall, precision, and F1 scores.

    Args:
        expected (list or np.ndarray): The expected ground truth answer(s). 
                                        If it's a numpy array, it will be converted to a list.
        generated (str or np.ndarray): The model-generated answer. 
                                        If it's a numpy array, it will be converted to a string.

    Returns:
        tuple: A tuple containing the following evaluation metrics:
            - precision (float): Precision based on token overlap.
            - recall (float): Recall based on token overlap.
            - f1_score (float): F1 score based on token overlap.
            - r1_recall (float): ROUGE-1 recall.
            - r1_precision (float): ROUGE-1 precision.
            - r1_f1 (float): ROUGE-1 F1 score.

    Example:
        evaluate_llm_response(["The quick brown fox"], "The quick brown fox jumps")
        (0.75, 0.75, 0.75, 0.6, 0.6, 0.6)
    """
    if isinstance(expected, np.ndarray):  # If it's a numpy array, convert it to a list
        expected = expected.tolist()

    if isinstance(generated, np.ndarray):  # If it's a numpy array, convert it to string
        generated = " ".join(generated.astype(str)) 

    if isinstance(expected, list):
        expected = " ".join(expected)
    if isinstance(generated, list):
        generated = " ".join(generated)

    precision, recall, f1_score = calculate_token_overlap(expected, generated)
    r1_recall, r1_precision, r1_f1, r2_recall, r2_precision, r2_f1, rl_recall, rl_precision, rl_f1 = evaluate_rouge(expected, generated)

    return precision, recall, f1_score, r1_recall, r1_precision, r1_f1

def evaluate_context_response(expected, generated):
    """
    Evaluates the generated response against the expected context using BLEU and ROUGE metrics.

    This function computes the following:
    - BLEU score for n-gram precision.
    - ROUGE-2 recall, precision, and F1 score, which evaluates the overlap of n-grams.

    Args:
        expected (dict or np.ndarray): The expected ground truth answer(s). 
                                       If it's a numpy array, it will be converted to a list.
        generated (dict or np.ndarray): The model-generated answer. 
                                       If it's a numpy array, it will be converted to a string.

    Returns:
        tuple: A tuple containing the following evaluation metrics:
            - bleu_score (float): The BLEU score based on n-gram precision.
            - r2_recall (float): ROUGE-2 recall score.
            - r2_precision (float): ROUGE-2 precision score.
            - r2_f1 (float): ROUGE-2 F1 score.

    Example:
        evaluate_context_response(["The quick brown fox"], "The quick brown fox jumps over")
        (0.6, 0.5, 0.5, 0.5)
    """
    ## Temp debug stage after saving to parquet
    if isinstance(expected, np.ndarray):  # If it's a numpy array, convert it to a list
        expected = expected.tolist()

    if isinstance(generated, np.ndarray):  # If it's a numpy array, convert it to string
        generated = " ".join(generated.astype(str)) 

    if isinstance(expected, list):
        expected = " ".join(expected)
    if isinstance(generated, list):
        generated = " ".join(generated)

    bleu_score = evaluate_bleu(expected, generated)
    r1_recall, r1_precision, r1_f1, r2_recall, r2_precision, r2_f1, rl_recall, rl_precision, rl_f1 = evaluate_rouge(expected, generated)

    return bleu_score, r2_recall, r2_precision, r2_f1

def evaluate_response(expected, generated):
    """
    Draft function - removed from main tool - was used for discovery evaluation to understand the best approach for scoring responses.

    Args:
    expected (dict): The expected output dictionary.
    generated (dict): The generated output dictionary.

    Returns:
    tuple: A tuple containing the BLEU score, ROUGE score, and exact match result.
    """
    if isinstance(expected, list):
        expected = " ".join(expected)
    if isinstance(generated, list):
        generated = " ".join(generated)

    gt_containment = evaluate_gt_containment(expected, generated)  
    precision, recall, f1_score = calculate_token_overlap(expected, generated)
    # jaccard_similarity = evaluate_jaccard_similarity(expected, generated) # Jaccard similarity removed as it is based on the ratio of the intersection to the union of two sets. It works well for finding exact token matches but is less useful when comparing answers that have similar meaning but different phrasing or extra details.
    bleu_score = evaluate_bleu(expected, generated)
    r1_recall, r1_precision, r1_f1, r2_recall, r2_precision, r2_f1, rl_recall, rl_precision, rl_f1 = evaluate_rouge(expected, generated)
    # exact_match = evaluate_exact_match(expected, generated) # Exact match removed as it is not a great measure in this use case
    
    return gt_containment, precision, recall, f1_score, bleu_score, r1_recall, r1_precision, r1_f1, r2_recall, r2_precision, r2_f1, rl_recall, rl_precision, rl_f1

def apply_llm_similarity_evaluation(df,  prefix, expected_col, generated_col):
    """
    Applies LLM similarity evaluation metrics (precision, recall, F1 score, ROUGE-1 scores) to each row in a DataFrame.
    
    This function evaluates the similarity between the expected (ground truth) answers and the generated answers 
    using a set of evaluation metrics. It calculates precision, recall, F1 score based on token overlap, 
    and ROUGE-1 recall, precision, and F1 score. The results are added as new columns to the input DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the expected and generated answers.
        prefix (str): The prefix to be used for the new columns created by this function.
        expected_col (str): The name of the column containing the expected (ground truth) answers.
        generated_col (str): The name of the column containing the generated answers.

    Returns:
        pandas.DataFrame: The original DataFrame with additional columns for the evaluation metrics.
            New columns will include:
            - `{prefix}_precision`: Precision of token overlap.
            - `{prefix}_recall`: Recall of token overlap.
            - `{prefix}_f1_score`: F1 score of token overlap.
            - `{prefix}_r1_recall`: ROUGE-1 recall.
            - `{prefix}_r1_precision`: ROUGE-1 precision.
            - `{prefix}_r1_f1`: ROUGE-1 F1 score.

    Example: 
        df = apply_llm_similarity_evaluation(df, 'llm', 'expected_answer', 'generated_answer')
    """
    df[[f'{prefix}_precision', f'{prefix}_recall', f'{prefix}_f1_score', f'{prefix}_r1_recall', f'{prefix}_r1_precision', f'{prefix}_r1_f1']] = df.apply(
        lambda x: pd.Series(evaluate_llm_response(x[expected_col], str(x[generated_col]))), axis=1
    )
    return df

def apply_context_similarity_evaluation(df,  prefix, expected_col, generated_col):
    """
    Applies context similarity evaluation metrics (BLEU score, ROUGE-2 scores) to each row in a DataFrame.
    
    This function compares the similarity between the expected (ground truth) context and the generated context
    using BLEU and ROUGE-2 metrics. It computes the BLEU score for n-gram precision and ROUGE-2 recall, precision,
    and F1 score. The results are added as new columns to the input DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the expected and generated contexts.
        prefix (str): The prefix to be used for the new columns created by this function.
        expected_col (str): The name of the column containing the expected (ground truth) context.
        generated_col (str): The name of the column containing the generated context.

    Returns:
        pandas.DataFrame: The original DataFrame with additional columns for the evaluation metrics.
            New columns will include:
            - `{prefix}_bleu_score`: BLEU score of n-gram precision.
            - `{prefix}_r2_recall`: ROUGE-2 recall.
            - `{prefix}_r2_precision`: ROUGE-2 precision.
            - `{prefix}_r2_f1`: ROUGE-2 F1 score.

    Example:
        df = apply_context_similarity_evaluation(df, 'context', 'expected_context', 'generated_context')
    """
    df[[f'{prefix}_bleu_score', f'{prefix}_r2_recall', f'{prefix}_r2_precision', f'{prefix}_r2_f1']] = df.apply(
        lambda x: pd.Series(evaluate_context_response(x[expected_col], str(x[generated_col]))), axis=1
    )
    return df

def mean_reciprocal_rank(df, k=5):
    """
    Calculate the Mean Reciprocal Rank (MRR) for the top k retrieved results.

    This function computes the Mean Reciprocal Rank, which is a metric used to evaluate the ranking of relevant documents 
    based on their relevance scores. For each row in the DataFrame, it calculates the reciprocal rank of the first relevant 
    result (if any), then averages these values across the dataset.

    Args:
        df (pandas.DataFrame): The DataFrame containing relevance scores. It must include a column named 'gt_relevance', 
                                where the relevance score is typically 1 for relevant documents and 0 for non-relevant ones.
        k (int, optional): The number of top results to consider for each query. The default value is 5. This parameter 
                           is currently unused as the implementation assumes a single relevance score per document.

    Returns:
        float: The Mean Reciprocal Rank (MRR) score, which is the average of reciprocal ranks for the relevant documents.
    
    Raises:
        KeyError: If the 'gt_relevance' column is not present in the input DataFrame.

    Example:
        mrr = mean_reciprocal_rank(df)
        print(mrr)
    """
    mrr_scores = []
    
    # Ensure that 'gt_relevance' is in the DataFrame passed to this function
    if 'gt_relevance' not in df.columns:
        raise KeyError("'gt_relevance' column is missing from the DataFrame.")
    
    for idx, row in df.iterrows():
        # Since 'gt_relevance' is expected to be an integer, just check its value directly
        relevance = row['gt_relevance']
        
        if relevance == 1:  # If the relevance score is 1, consider it relevant
            mrr_scores.append(1)  # Reciprocal rank for relevant document
        else:
            mrr_scores.append(0)  # If not relevant, reciprocal rank is 0.
    
    return sum(mrr_scores) / len(mrr_scores)

def precision_at_k(df, k=5):
    """
    Calculate Precision at k (P@k) for the top k retrieved results.

    This function computes Precision at k, a metric used to evaluate the precision of the top k retrieved documents based 
    on their relevance scores. It calculates the proportion of relevant documents in the top k results.

    Args:
        df (pandas.DataFrame): The DataFrame containing relevance scores. It must include a column named 'gt_relevance', 
                                where the relevance score is typically 1 for relevant documents and 0 for non-relevant ones.
        k (int, optional): The number of top results to consider for each query. The default value is 5. In this implementation, 
                           the value of k is currently unused as the function assumes a binary relevance score for each document.

    Returns:
        float: The Precision at k (P@k) score, which is the proportion of relevant documents in the top k retrieved results.
    
    Raises:
        KeyError: If the 'gt_relevance' column is not present in the input DataFrame.

    Example:
        precision = precision_at_k(df)
        print(precision)
    """
    precision_scores = []
    
    # Ensure that 'gt_relevance' is in the DataFrame passed to this function
    if 'gt_relevance' not in df.columns:
        raise KeyError("'gt_relevance' column is missing from the DataFrame.")
    
    for idx, row in df.iterrows():
        # Get the top k relevance scores for each query (no need to iterate if it's just a binary score)
        top_k_relevance = row['gt_relevance']  # We expect a single value per row, not a list
        precision = top_k_relevance  # Just using the value directly since it's binary
        precision_scores.append(precision)
    
    return sum(precision_scores) / len(precision_scores)

def calculate_gt_relevance(df, threshold):
    """
    Calculate ground truth relevance scores based on the BLEU score or other similarity metrics.

    This function computes a relevance score for each entry in the DataFrame by comparing the 
    BLEU score (or any other specified similarity metric) against a given threshold. If the score 
    exceeds or equals the threshold, the entry is considered relevant (marked as 1); otherwise, 
    it is considered non-relevant (marked as 0).

    Args:
        df (pandas.DataFrame): A DataFrame containing the 'embeddings_bleu_score' column, 
                                which holds the similarity scores (e.g., BLEU score) for each entry.
        threshold (float): The threshold value for determining relevance. Entries with a score 
                           greater than or equal to this value are marked as relevant (1), 
                           otherwise as non-relevant (0).

    Returns:
        pandas.DataFrame: The original DataFrame with an additional column 'gt_relevance' that contains 
                          the calculated relevance scores (1 for relevant, 0 for non-relevant).

    Raises:
        KeyError: If the 'embeddings_bleu_score' column is missing from the DataFrame.

    Example:
        df = calculate_gt_relevance(df, threshold=0.5)
        print(df[['embeddings_bleu_score', 'gt_relevance']])
    """
    if 'embeddings_bleu_score' not in df.columns:
        raise KeyError("'embeddings_bleu_score' column is missing from the DataFrame.")
    df['gt_relevance'] = df['embeddings_bleu_score'].apply(lambda x: 1 if x >= threshold else 0)
    return df

def evaluate_rag_with_ranking(df, k=5, threshold=0.1):  # Lower threshold due to incredibly low bleu scores
    """
    Calculate ground truth relevance scores based on the BLEU score or other similarity metrics.

    This function computes a relevance score for each entry in the DataFrame by comparing the 
    BLEU score (or any other specified similarity metric) against a given threshold. If the score 
    exceeds or equals the threshold, the entry is considered relevant (marked as 1); otherwise, 
    it is considered non-relevant (marked as 0).

    Args:
        df (pandas.DataFrame): A DataFrame containing the 'embeddings_bleu_score' column, 
                                which holds the similarity scores (e.g., BLEU score) for each entry.
        threshold (float): The threshold value for determining relevance. Entries with a score 
                           greater than or equal to this value are marked as relevant (1), 
                           otherwise as non-relevant (0).

    Returns:
        pandas.DataFrame: The original DataFrame with an additional column 'gt_relevance' that contains 
                          the calculated relevance scores (1 for relevant, 0 for non-relevant).

    Raises:
        KeyError: If the 'embeddings_bleu_score' column is missing from the DataFrame.

    Example:
        df = calculate_gt_relevance(df, threshold=0.5)
        print(df[['embeddings_bleu_score', 'gt_relevance']])
    """
    is_possible = df[df.gt_is_impossible == False]
    impossible = df[df.gt_is_impossible == True]
    
    apply_llm_similarity_evaluation(is_possible, 'llm', 'gt_answers', 'answer')
    apply_gt_containment_evaluation(is_possible, 'gt_answers', 'answer')
    is_possible = fuzzmatch_llm_scoring(is_possible, threshold=80)
    apply_context_similarity_evaluation(is_possible, 'embeddings', 'gt_contexts', 'context')

    is_possible = calculate_gt_relevance(is_possible, threshold)

    results = []

    for collection, group in is_possible.groupby('collection_name'):
        mrr_score = mean_reciprocal_rank(group, k)
        precision_score = precision_at_k(group, k)

        collection_results = {
            'collection_name': collection,
            'Mean Reciprocal Rank (MRR)': mrr_score,
            'Precision at 5 (P@5)': precision_score,
            'llm_gt_containment_proportion': group['llm_gt_containment'].mean(),
            'llm_unigram_match': group['llm_unigram_match'].mean(),
            'llm_bigram_match': group['llm_bigram_match'].mean(),
            'llm_trigram_match': group['llm_trigram_match'].mean(),
            'llm_precision': group['llm_precision'].mean(),
            'llm_recall': group['llm_recall'].mean(),
            'llm_f1_score': group['llm_f1_score'].mean(),
            'llm_r1_recall': group['llm_r1_recall'].mean(),
            'llm_r1_precision': group['llm_r1_precision'].mean(),
            'llm_r1_f1': group['llm_r1_f1'].mean(),
            'embeddings_bleu_score': group['embeddings_bleu_score'].mean(),
            'embeddings_r2_recall': group['embeddings_r2_recall'].mean(),
            'embeddings_r2_precision': group['embeddings_r2_precision'].mean(),
            'embeddings_r2_f1': group['embeddings_r2_f1'].mean(),
            'time_taken': group['time_taken'].mean(),  
        }

        impossible_subset = impossible[impossible['collection_name'] == collection]
        
        if not impossible_subset.empty:
            impossible_score = (impossible_subset['answer'] == 'Answer is impossible.').mean()
        else:
            impossible_score = 0.0  # If no impossible answers for this collection
        
        collection_results['impossible_score'] = impossible_score

        results.append(collection_results)

    summary_df = pd.DataFrame(results)

    return summary_df