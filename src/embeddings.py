import os
import chromadb
import voyageai
from transformers import AutoTokenizer
from dotenv import load_dotenv
load_dotenv()
from base import BASE_DIR

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
vo = voyageai.Client(api_key=VOYAGE_API_KEY)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
db_client = chromadb.PersistentClient(path=f"{BASE_DIR}/embeddings/voiyage-2")

def count_tokens(text):
    tokens = tokenizer.tokenize(text)
    return len(tokens)

def find_max_question_length(data):
    question_tokens = []
    for document in data:
        for paragraph in document["paragraphs"]:
            for qa in paragraph["qas"]:
                question_length = count_tokens(qa["question"])
                question_tokens.append(question_length)
    return max(question_tokens)

def call_vo_embeddings(texts):
    embeddings_response = vo.embed(texts, model="voyage-2", input_type="document")
    return embeddings_response

def preprocess_text(text):
    tokens = tokenizer.tokenize(text.lower())
    return tokens

def chunk_text(text, max_tokens=512):
    tokens = preprocess_text(text)
    chunks = []
    
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]  # Adjust to avoid exceeding max tokens
        
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        chunks.append(chunk_text)
    
    return chunks

def generate_embeddings(data, chunk_size):
    collection_name = f"legal_docs_voiyage2_{chunk_size}"
    collection = db_client.get_or_create_collection(name=collection_name)
    print(f"Created ChromaDB collection: {collection}")

    MAX_TOKENS = chunk_size - find_max_question_length(data)  
    for doc_id, document in enumerate(data):
        for paragraph in document["paragraphs"]:
            context = paragraph["context"]
            chunked_text = chunk_text(context, MAX_TOKENS)
            
            for i, chunk in enumerate(chunked_text):
                qa_id = f"{document['title']}_chunk_{i}"

                embeddings_response = call_vo_embeddings([chunk])
                embedding = embeddings_response.embeddings[0]

                print(f"Adding {qa_id} to ChromaDB collection.")
                collection.add(
                    ids=[qa_id],  
                    documents=[chunk],
                    metadatas=[{"doc_id": document["title"], "chunk_id": qa_id}],
                    embeddings=[embedding]
                )