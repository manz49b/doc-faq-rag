import json
import time
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from base import BASE_DIR
from utils import safe_load_json, save_to_parquet
from embeddings import call_vo_embeddings, count_tokens, chunk_text
from claude import call_claude
from gpt import call_openai_gpt
from prompt import system_prompt, user_prompt
from utils import RateLimitTracker

rate_tracker = RateLimitTracker()

def get_long_keywords():
    with open(f'{BASE_DIR}/steer/question_keywords_long_answers.json', 'r') as f:
        return json.load(f)

def preprocess_question(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return tokens

def contains_keywords(question, keywords):
    question_tokens = set(preprocess_question(question)) 
    return any(keyword in question_tokens for keyword in keywords)

def retrieve_answer(question, n_results, document_title, collection):
    question_embedding = call_vo_embeddings([question])
    
    results = collection.query(
        query_embeddings=question_embedding.embeddings,
        n_results=n_results,  # Return the closest match based on the number of results
        where={"doc_id": document_title}  # Filter by document title
    )
    
    context = results["documents"][0][0] if results["documents"] else None
    context_embedding = collection.get(ids=results['ids'][0], include=['embeddings'])['embeddings']

    if context:
        print(f"Context tokens: {count_tokens(context)}")
        prompt = user_prompt(question, context)
        
        print(f"Prompt + context tokens: {count_tokens(prompt)}")
        # answer = call_claude(system_prompt, prompt)
        # answer = answer[0].text
        
        answer = call_openai_gpt(system_prompt(), prompt)
        print("GPT SUCCESS")
        print(prompt)
        print(answer)

        return answer, context, context_embedding
    else:
        print(f"No relevant chunk found for document {document_title}.")
        return None, None, None
    
def find_gt_chunk(chunks, answer_starts):
    current_pos = 0
    valid_chunks = []  # To collect all valid chunks for different answer_start

    # Loop through all answer starts
    for answer_start in answer_starts:
        for chunk in chunks:
            chunk_length = len(chunk)

            if answer_start <= chunk_length:
                valid_chunks.append(chunk)  # If answer_start is within this chunk, keep it
                break  # No need to check further chunks for this answer_start
            else:
                current_pos += chunk_length

                if answer_start <= current_pos:
                    valid_chunks.append(chunk)  # Found the chunk with the answer_start
                    break

    return list(set(valid_chunks))  # Return all valid chunks that could match

def run_faq_rag(data, collection_name, db_client):
    collection = db_client.get_or_create_collection(name=collection_name)
    print(f"Created ChromaDB collection: {collection_name}")

    dfs = []

    for document in data[:2]:
        document_title = document["title"]  # Store document title for relevant retrieval
        for paragraph in document["paragraphs"]:
            context = paragraph["context"]
            
            # token_count = int(collection_name.split('_')[-1])
            token_count = 4096
            chunks = chunk_text(context, max_tokens=token_count)

            for qa in paragraph["qas"]:
                question = qa["question"]
                gt_answers = [answer["text"] for answer in qa["answers"]]
                gt_answer_starts = [answer["answer_start"] for answer in qa["answers"]] 
                gt_is_impossible = qa.get("is_impossible", False)

                if not gt_is_impossible:  # Should we filter for impossible to answer for all
                    gt_contexts = find_gt_chunk(chunks, gt_answer_starts)
                    gt_context_embedding = call_vo_embeddings(gt_contexts).embeddings[0]
                else:
                    gt_contexts = []
                    gt_context_embedding = []

                if contains_keywords(question, get_long_keywords()):
                    n_results = 5  # Retrieve more documents for long-answer questions
                else:
                    n_results = 1  # Retrieve only one document for short-answer questions

                try:
                    start_time = time.time()
                    answer, best_chunk, best_chunk_embedding = retrieve_answer(question, n_results, document_title, collection)
                    end_time = time.time()
                    time_taken = end_time - start_time
                    # rate_tracker.calculate_delay(count_tokens(str(answer))) # was required for claude

                    dfs.append({
                    "document_title": document_title,
                    "question": question,
                    "context": best_chunk,
                    "context_embedding": best_chunk_embedding,
                    "answer": answer,
                    "gt_contexts": gt_contexts,
                    "gt_context_embedding": gt_context_embedding,
                    "gt_answer_starts": gt_answer_starts,
                    "gt_answers": gt_answers,
                    "gt_is_impossible": gt_is_impossible,
                    "time_taken": time_taken
                })
                    
                except Exception as e:
                    df = pd.DataFrame(dfs)  # df with partial results
                    df["collection_name"] = collection_name
                    save_to_parquet(df, f"{BASE_DIR}/output/partial/data.parquet")
                    if "rate_limit_error" in str(e):  # Check if it's a rate limit error
                        print("Rate limit exceeded, saving the results so far.")
                        break 
                    else:
                        # Handle other exceptions as needed
                        print(f"An error occurred: {e}")
                        save_to_parquet(df, f"{BASE_DIR}/output/partial/data.parquet")
                        break
    df = pd.DataFrame(dfs)
    df["collection_name"] = collection_name
    return df

def find_collections(client):
    collections = client.list_collections()
    available_collections = [c.name for c in collections]
    if 'legal_docs_voiyage2' in available_collections:
        available_collections.remove('legal_docs_voiyage2')
    return available_collections

def retrieve_rag_results(available_collections, data, db_client):
    dfs = []

    for collection_name in available_collections:
        df = run_faq_rag(data, collection_name, db_client)
        dfs.append(df)

        out = pd.concat(dfs)
    out.answer = [safe_load_json(x)['answer'][0] for x in out.answer]

    out['answer'] = out['answer'].replace(
        to_replace=r'^Answer is impossible.*', 
        value='Answer is impossible.', 
        regex=True
    )
    return out