import numpy as np
import faiss
import pickle
import os
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

class HybridRetriever:
    def __init__(self, model_name: str = 'intfloat/multilingual-e5-small', index_path: str = "data/vector_store"):
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # In-memory stores
        self.faiss_index = faiss.IndexFlatIP(self.dimension) # Inner product for cosine sim (E5 embeddings are normalized)
        self.metadata: List[Dict[str, Any]] = []
        self.bm25: BM25Okapi = None
        self.corpus_tokens: List[List[str]] = []
        self.original_texts: List[str] = []
        
        # Create directory if it doesn't exist
        os.makedirs(self.index_path, exist_ok=True)
        self._faiss_path = os.path.join(self.index_path, "faiss.index")
        self._meta_path = os.path.join(self.index_path, "meta.pkl")
        self._try_load()

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-10)

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]], batch_size: int = 32):
        """
        Embeds texts in batches and adds them to FAISS and BM25 index.
        Prefixes texts with 'passage: ' as required by E5 models.
        """
        if not texts:
            return
            
        # Add 'passage: ' prefix for E5
        prefixed_texts = [f"passage: {t}" for t in texts]
        
        # Dense indexing
        all_embeddings = []
        for i in range(0, len(prefixed_texts), batch_size):
            batch = prefixed_texts[i:i+batch_size]
            emb = self.model.encode(batch, convert_to_numpy=True)
            all_embeddings.append(emb)
            
        embeddings_np = np.vstack(all_embeddings)
        embeddings_np = self._normalize_embeddings(embeddings_np)
        
        self.faiss_index.add(embeddings_np.astype(np.float32))
        self.metadata.extend(metadatas)
        self.original_texts.extend(texts)
        
        # Sparse indexing (BM25)
        new_tokens = [t.lower().split() for t in texts]
        self.corpus_tokens.extend(new_tokens)
        self.bm25 = BM25Okapi(self.corpus_tokens)
        self.save()

    def _rrf(self, dense_results: List[Tuple[int, float]], sparse_results: List[Tuple[int, float]], k: int = 60) -> List[int]:
        """
        Reciprocal Rank Fusion.
        Returns sorted indices based on RRF score.
        """
        rrf_scores = {}
        
        for rank, (idx, _) in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
            
        for rank, (idx, _) in enumerate(sparse_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
            
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return sorted_indices

    def search(self, query: str, product_id: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Hybrid search combining Dense (FAISS) and Sparse (BM25).
        Prefixes query with 'query: ' for E5.
        Filters by product_id if provided.
        """
        if self.faiss_index.ntotal == 0:
            return []
            
        # 1. Dense Search
        query_emb = self.model.encode([f"query: {query}"], convert_to_numpy=True)
        query_emb = self._normalize_embeddings(query_emb)
        
        # Retrieve more candidates to account for filtering
        fetch_k = top_k * 5 if product_id else top_k * 2
        
        # Check if we have enough elements in the index
        if self.faiss_index.ntotal < fetch_k:
            fetch_k = self.faiss_index.ntotal
            
        distances, indices = self.faiss_index.search(query_emb.astype(np.float32), fetch_k)
        
        dense_results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            # Filter by product_id
            if product_id and self.metadata[idx].get("product_id") != product_id:
                continue
            dense_results.append((int(idx), float(distances[0][i])))

        # 2. Sparse Search
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Get top candidates from BM25, also filtering by product_id
        sparse_candidates = []
        for idx, score in enumerate(bm25_scores):
            if score > 0:
                if product_id and self.metadata[idx].get("product_id") != product_id:
                    continue
                sparse_candidates.append((idx, score))
                
        sparse_candidates.sort(key=lambda x: x[1], reverse=True)
        sparse_candidates = sparse_candidates[:fetch_k]
        
        # 3. RRF Merge
        merged_indices = self._rrf(dense_results, sparse_candidates)
        
        # Return top_k enriched with metadata
        results = []
        for idx in merged_indices[:top_k]:
            results.append({
                "content": self.original_texts[idx],
                "metadata": self.metadata[idx]
            })
            
        return results

    def save(self):
        """Persist FAISS + metadata for 4-person scale (10K+ docs)."""
        try:
            faiss.write_index(self.faiss_index, self._faiss_path)
            with open(self._meta_path, "wb") as f:
                pickle.dump({"metadata": self.metadata, "texts": self.original_texts, "tokens": self.corpus_tokens}, f)
        except Exception as e:
            print(f"[HybridRetriever] save failed: {e}")

    def _try_load(self):
        if os.path.exists(self._faiss_path) and os.path.exists(self._meta_path):
            try:
                self.faiss_index = faiss.read_index(self._faiss_path)
                with open(self._meta_path, "rb") as f:
                    data = pickle.load(f)
                    self.metadata = data.get("metadata", [])
                    self.original_texts = data.get("texts", [])
                    self.corpus_tokens = data.get("tokens", [])
                    if self.corpus_tokens:
                        self.bm25 = BM25Okapi(self.corpus_tokens)
                print(f"[HybridRetriever] loaded {self.faiss_index.ntotal} docs from {self.index_path}")
            except Exception as e:
                print(f"[HybridRetriever] load failed: {e}")
