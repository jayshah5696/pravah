import os
import asyncio
import numpy as np
from bm25s import BM25  # Updated import
import bm25s
from typing import List
import faiss
import fastavro
from dotenv import load_dotenv
import tiktoken
import re
import litellm
from rerankers import Reranker
from functools import lru_cache 
from cachetools import LRUCache, cached  
from cachetools.keys import hashkey  
from .regax_pattern import combined_pattern
from functools import lru_cache
import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import OpenAIEmbeddings
import uuid
from langsmith import traceable, Client
load_dotenv()

class LiteLLMEmbeddingClient:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key
    async def embed_document(self, text: str) -> List[float]:
        response = litellm.embedding(input=[text], model=self.model, api_key=self.api_key)
        return response['data'][0]['embedding']

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        tasks = [self.embed_document(text) for text in texts]
        return await asyncio.gather(*tasks)

    async def embed_query(self, query: str) -> List[float]:
        return await self.embed_document(query)

@traceable
class RetrievalEngine:
    def __init__(self, texts: List[dict],
                uuid_input: str,
                chunk_size: int = 500,
                overlap: int = 100,
                chunking_method: str = 'tokens',
                tokens: bool = False,
                use_lancedb: bool = False,
                embed_client = LiteLLMEmbeddingClient(model= "text-embedding-3-small",
                                                    api_key=os.environ['OPENAI_API_KEY']),
                reranker = Reranker('flashrank')):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokens = tokens
        self.use_lancedb = use_lancedb
        self.chunking_method = chunking_method
        self.chunks = self.chunk_texts(texts, uuid_input=uuid_input)
        self.embed_client = embed_client
        self.embeddings = None
        self.index = None

        if self.use_lancedb:
            self.lancedb = lancedb.connect(".my_db")
            self.tbl = asyncio.run(self.create_lancedb_table())  # Use asyncio.run to call the async function
            self.add_chunks_to_lancedb()
        else:
            self.bm25 = self.create_bm25()
            self.reranker = reranker

    def chunk_texts(self, texts: List[dict], uuid_input: str) -> List[dict]:
        chunks = []
        for item in texts:
            text = item['content']
            url = item['url']
            if self.chunking_method == 'tokens':  
                chunked_texts = self.chunk_text_by_tokens(text, self.chunk_size, self.overlap)
            elif self.chunking_method == 'regax': 
                chunked_texts = self.chunk_regax(text, self.chunk_size, self.overlap)
            else:  # And this line
                chunked_texts = self.chunk_text(text, self.chunk_size, self.overlap)
            for chunk in chunked_texts:
                chunks.append({'content': chunk, 'url': url, 'uuid': uuid_input})
        self.uuid_input = uuid_input
        return chunks

    def chunk_text(self, text, max_char_length=1000, overlap=0):
        chunks = []
        current_chunk = ""
        sentences = re.split(r'(\.|\?|!)', text.replace('\n', ' '))

        for sentence in sentences:
            trimmed_sentence = sentence.strip()
            if not trimmed_sentence:
                continue

            chunk_length = len(current_chunk) + len(trimmed_sentence) + 1
            lower_bound = max_char_length - max_char_length * 0.5
            upper_bound = max_char_length + max_char_length * 0.5

            if lower_bound <= chunk_length <= upper_bound and current_chunk:
                current_chunk = re.sub(r'^\.\s+', "", current_chunk).strip()
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = ""
            elif chunk_length > upper_bound:
                current_chunk = re.sub(r'^\.\s+', "", current_chunk).strip()
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = trimmed_sentence
            else:
                current_chunk += f" {trimmed_sentence}"

        if current_chunk:
            chunks.append(current_chunk)

        if overlap > 0:
            overlapped_chunks = []
            for i in range(len(chunks)):
                start = max(0, i - overlap)
                end = min(len(chunks), i + 1)
                overlapped_chunks.append(' '.join(chunks[start:end]))
            return overlapped_chunks

        return chunks

    def chunk_text_by_tokens(self, text, max_token_length=100, overlap=0):
        enc = tiktoken.get_encoding("o200k_base")
        tokens = enc.encode(text,disallowed_special=())
        chunks = []

        for i in range(0, len(tokens), max_token_length - overlap):
            chunk_tokens = tokens[i:i + max_token_length]
            chunk_text = enc.decode(chunk_tokens)
            chunks.append(chunk_text)

        return chunks

    def create_bm25(self):
        tokenized_chunks = bm25s.tokenize([chunk['content'] for chunk in self.chunks])

        bm25 = BM25()
        bm25.index(tokenized_chunks)
        return bm25

    @lru_cache(maxsize=128)
    async def create_embeddings(self):
        if self.embeddings is None:
            texts = [chunk['content'] for chunk in self.chunks]
            self.embeddings = await self.embed_client.embed_documents(texts)
        return self.embeddings
    
    @lru_cache(maxsize=128)
    async def semantic_query_run(self, query: str, top_k: int = 5) -> List[dict]:
        query_embedding = await self.embed_client.embed_documents([query])
        index = await self.create_faiss_index()  # Ensure this is awaited
        distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)
        return distances, indices
    
    @lru_cache(maxsize=128)
    async def semantic_search(self, query: str, top_k: int = 5) -> List[dict]:
        distance, indices = await self.semantic_query_run(query, top_k)
        return [self.chunks[i] for i in indices[0]]

    @lru_cache(maxsize=128)
    async def create_faiss_index(self):
        if self.index is None:
            embeddings = await self.create_embeddings()
            dimension = len(embeddings[0])
            embeddings = np.array(embeddings)
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
        return self.index

    @lru_cache(maxsize=128)
    async def keyword_search(self, query: str, top_k: int = 5) -> List[dict]:
        if self.use_lancedb:
            return await self.lancedb_keyword_search(query, top_k)
        else:
            # Get BM25 scores for the query
            query_tokens = bm25s.tokenize(query)
            docs, scores = self.bm25.retrieve(query_tokens, k=top_k)
            return [self.chunks[doc_id] for doc_id in docs[0]]


    @lru_cache(maxsize=128)
    async def combined_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[dict]:
        if self.use_lancedb:
            return await self.lancedb_combined_search(query, top_k)
        else:
            query_tokens = bm25s.tokenize(query)
            keyword_docs, keyword_scores = self.bm25.retrieve(query_tokens, k=len(self.chunks))
            keyword_scores = self.normalize_scores(keyword_scores[0])  # Normalize keyword scores
            distances, indices = await self.semantic_query_run(query, len(self.chunks))
            semantic_scores = np.zeros(len(self.chunks))  # Initialize semantic scores
            semantic_scores[indices[0]] = 1 / (1 + distances[0])  # Calculate semantic scores
            combined_scores = alpha * keyword_scores + (1 - alpha) * semantic_scores  # Combine scores
            top_indices = np.argsort(combined_scores)[-top_k:][::-1]  # Get top indices
            return [self.chunks[i] for i in top_indices]

    
    def normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize an array of scores to a range between 0 and 1.
        Args:
            scores (np.ndarray): The array of scores to normalize.
        Returns:
            np.ndarray: The normalized scores."""
        # Find the minimum and maximum scores
        min_score = np.min(scores)
        max_score = np.max(scores)
        # Normalize the scores to a range between 0 and 1
        return (scores - min_score) / (max_score - min_score)
    
    async def save_faiss_index(self, file_path: str):
        """Save the FAISS index to a file.
        
        Args:
            file_path (str): The path to the file where the index will be saved.
        """
        if self.index is None:
            await self.create_faiss_index()
        faiss.write_index(self.index, file_path)

    async def save_bm25_index_avro(self, file_path: str):
        """Save the BM25 index to an Avro file.
        
        Args:
            file_path (str): The path to the file where the index will be saved.
        """
        schema = {
            "type": "record",
            "name": "BM25Index",
            "fields": [
                {"name": "doc_freqs", "type": {"type": "array", "items": {"type": "map", "values": "int"}}},
                {"name": "idf", "type": {"type": "array", "items": "double"}},
                {"name": "doc_len", "type": {"type": "array", "items": "int"}},
                {"name": "avgdl", "type": "double"}
            ]
        }
        bm25_data = {
            'doc_freqs': self.bm25.doc_freqs,
            'idf': self.bm25.idf,
            'doc_len': self.bm25.doc_len,
            'avgdl': self.bm25.avgdl
        }
        with open(file_path, 'wb') as f:
            fastavro.writer(f, schema, [bm25_data])

    async def load_bm25_index_avro(self, file_path: str):
        """Load the BM25 index from an Avro file.
        
        Args:
            file_path (str): The path to the file from which the index will be loaded.
        """
        with open(file_path, 'rb') as f:
            reader = fastavro.reader(f)
            bm25_data = next(reader)
        self.bm25 = BM25(corpus=[])  # Initialize an empty BM25
        self.bm25.doc_freqs = bm25_data['doc_freqs']
        self.bm25.idf = bm25_data['idf']
        self.bm25.doc_len = bm25_data['doc_len']
        self.bm25.avgdl = bm25_data['avgdl']

    async def rerank_chunks(self, query: str, chunks: List[dict], top_k: int = 5) -> List[dict]:
        """Rerank chunks of text using the Reranker library.
        
        Args:
            query (str): The query string.
            chunks (List[dict]): The list of chunks to rerank.
            top_k (int): The number of top results to return.
        
        Returns:
            List[dict]: The top-k ranked chunks.
        """
        # Extract texts from chunks
        texts = [chunk['content'] for chunk in chunks]
        # Perform reranking
        results = await self.reranker.rank_async(query=query, docs=texts)
        # Extract top-k results
        top_results = results.top_k(top_k)
        # Map results back to chunks
        ranked_chunks = [chunks[result.document.doc_id] for result in top_results]
        return ranked_chunks
    
    def chunk_regax(self, text, max_char_length=1000, overlap=0):
        matches = re.findall(combined_pattern, text, re.MULTILINE)
        chunks = []
        current_chunk = ""    
        for match in matches:
            if len(match) > max_char_length:
                # Split the match into smaller chunks
                for i in range(0, len(match), max_char_length - overlap):
                    chunk = match[i:i + max_char_length]
                    chunks.append(chunk)
            else:
                if len(current_chunk) + len(match) <= max_char_length:
                    current_chunk += match
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = match
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
        
    async def create_lancedb_table(self):
        model = OpenAIEmbeddings(name='text-embedding-3-small',dim=1536)
        class Document(LanceModel):
            content: str = model.SourceField()
            vector: Vector(1536) = model.VectorField()
            url: str
            uuid: str

        table_mode = "append" if "pravah_chunks" in self.lancedb.table_names() else "create"
        if table_mode=='create':
            return await self.lancedb.create_table("pravah_chunks", schema=Document)
        else:
            return self.lancedb.open_table("pravah_chunks")

    def add_chunks_to_lancedb(self):
        self.tbl.add(self.chunks)
        self.tbl.create_fts_index("content",replace=True)

    async def lancedb_keyword_search(self, query: str, top_k: int = 5) -> List[dict]:
        results = self.tbl.search(query, query_type='fts').where(f"uuid={self.uuid_input}").limit(top_k).to_list()
        return results

    async def lancedb_semantic_search(self, query: str, top_k: int = 5) -> List[dict]:
        results = self.tbl.search(query, query_type='vector').where(f"uuid={self.uuid_input}").limit(top_k).to_list()
        return results

    async def lancedb_hybrid_search(self, query: str, top_k: int = 5) -> List[dict]:
        results = self.tbl.search(query, query_type='hybrid').where(f"uuid={self.uuid_input}").limit(top_k).to_list()
        return results

    async def lancedb_combined_search(self, query: str, top_k: int = 5) -> List[dict]:
        from lancedb.rerankers import CohereReranker
        reranker = CohereReranker(column='content')
        results = (self.tbl.search(query, query_type='hybrid')
                   .limit(top_k*2)
                   .rerank(reranker=reranker).limit(top_k))
        return results.to_list()