import re
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticChunker:
    def __init__(self, model_name: str = 'intfloat/multilingual-e5-small', threshold: float = 0.6):
        """
        Initializes the Semantic Chunker with a multilingual embedding model.
        Using multilingual-E5 as requested.
        """
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text: str, max_chunk_chars: int = 600) -> List[str]:
        """
        Growing-window strategy:
        Compares the next sentence against the running embedding of the current chunk.
        """
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_chunk = [sentences[0]]
        
        # Pre-compute sentence embeddings to avoid multiple encoding calls for single sentences
        # Though the running chunk embedding needs to be updated.
        
        # Start the running embedding
        running_embedding = self.model.encode(sentences[0])
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_emb = self.model.encode(sentence)
            
            # Semantic check against running embedding
            sim = cosine_similarity([running_embedding], [sentence_emb])[0][0]
            
            current_text = " ".join(current_chunk)
            if sim >= self.threshold and (len(current_text) + len(sentence)) <= max_chunk_chars:
                # Add to chunk
                current_chunk.append(sentence)
                # Update running embedding (simple average of current text for the new running chunk)
                new_chunk_text = " ".join(current_chunk)
                running_embedding = self.model.encode(new_chunk_text)
            else:
                # Split chunk
                chunks.append(current_text)
                current_chunk = [sentence]
                running_embedding = sentence_emb

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
