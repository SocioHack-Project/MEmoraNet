from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

import torch
import faiss
import json
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from rank_bm25 import BM25Okapi
from collections import OrderedDict
import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

DATA_FILE = "search_data.json"
MODEL_NAME = "microsoft/phi-2"
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
CACHE_SIZE = 100
MAX_NEW_TOKENS = 150

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    logging.info(f"Successfully loaded language model: {MODEL_NAME}")
except Exception as e:
    logging.critical(f"Critical error loading language model: {e}")
    exit(1)

try:
    embedding_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    logging.info(f"Successfully loaded sentence transformer model: {SENTENCE_TRANSFORMER_MODEL}")
except Exception as e:
    logging.critical(f"Critical error loading sentence transformer model: {e}")
    exit(1)

class LRUCache(OrderedDict):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity

    def __getitem__(self, key):
        self.move_to_end(key)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        super().__setitem__(key, value)
        if len(self) > self.capacity:
            self.popitem(last=False)

cache = LRUCache(capacity=CACHE_SIZE)


index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
corpus = []
embeddings = []


try:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
            corpus = data.get('corpus', [])
            embeddings = data.get('embeddings', [])
            if embeddings:
                index.add(np.array(embeddings))
except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
    logging.warning(f"Error loading data from {DATA_FILE}: {e}. Starting with an empty database.")


def build_bm25_index(corpus):
    try:
        if corpus:
            return BM25Okapi([doc.split() for doc in corpus])
        else:
            return None
    except Exception as e:
        logging.error(f"Error building BM25 index: {e}")
        return None


bm25_index = build_bm25_index(corpus)


def hybrid_search(query, top_k=3):
    query_embedding = embedding_model.encode(query, normalize_embeddings=True)

    bm25_results = []
    if bm25_index:
        bm25_results = bm25_index.get_top_n(query.split(), corpus, n=min(len(corpus),top_k))

    dense_results = []
    if index.ntotal > 0:
        D, I = index.search(np.array([query_embedding], dtype=np.float32), k=top_k)
        dense_results = [corpus[i] for i in I[0] if 0 <= i < len(corpus)]

    results = list(set(bm25_results + dense_results))
    return results[:top_k]


def generate_response(prompt):
    cached_response = cache.get(prompt)
    if cached_response:
        logging.info(f"Cache hit for prompt: {prompt}")
        return cached_response

    relevant_docs = hybrid_search(prompt)

    system_prompt = "You are a knowledgeable AI assistant. Answer based on provided context."
    context = "\n".join(relevant_docs) if relevant_docs else ""
    full_prompt = f"{system_prompt}\n\nContext: {context}\n\nUser: {prompt}\nAI:"

    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, padding=True).to("cuda" if torch.cuda.is_available() else "cpu")
    try:
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.1
            )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        cache[prompt] = response 


        if prompt:
            embedding = embedding_model.encode(prompt, normalize_embeddings=True)
            corpus.append(prompt)
            embeddings.append(embedding.tolist())
            index.add(np.array([embedding], dtype=np.float32))
            bm25_index = build_bm25_index(corpus) 
            save_data()
        return response
    except Exception as e:
        logging.exception(f"Error during response generation: {e}")
        return "I'm having trouble generating a response right now. Please try again later."



def save_data():
    try:
        with open(DATA_FILE, "w") as f:
            json.dump({"corpus": corpus, "embeddings": embeddings}, f, indent=4)
        logging.info(f"Data saved to {DATA_FILE}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")


while True:
    user_input = input("\nUser: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye! 👋")
        break
    try:
        bot_response = generate_response(user_input)
        print(f"Chatbot: {bot_response}")
    except Exception as e:
        logging.exception(f"A serious error occurred: {e}")
        print("Chatbot: I encountered a serious error. Please try again later.")

app = FastAPI()


origins = ["http://localhost:3000", "https://memora-net.vercel.app/"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query")
async def query_route(query: dict):
    """Handles query requests."""
    try:
        user_query = query["query"]
        response = generate_response(user_query)  # Call the function directly
        return {"response": response}
    except Exception as e:
        logging.exception(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Use 0.0.0.0 for Docker