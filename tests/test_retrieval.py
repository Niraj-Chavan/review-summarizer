import pytest
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.chunker import SemanticChunker
from src.retrieval.hybrid_search import HybridRetriever

def test_preprocessing_and_retrieval():
    # 1. Initialize components
    cleaner = TextCleaner()
    chunker = SemanticChunker(model_name="intfloat/multilingual-e5-small", threshold=0.5)
    retriever = HybridRetriever(model_name="intfloat/multilingual-e5-small")

    # 2. Sample data across languages
    raw_reviews = [
        {
            "review_id": "r1",
            "product_id": "p1",
            "text": "<p>The battery life is amazing! I charge it once a day. tbh it is gr8 👍</p>"
        },
        {
            "review_id": "r2",
            "product_id": "p1",
            "text": "La duración de la batería es increíble, me dura todo el día. Pero el envío fue muy lento."
        },
        {
            "review_id": "r3",
            "product_id": "p1",
            "text": "L'écran est magnifique, les couleurs sont très vives. Je recommande vivement."
        },
        {
            "review_id": "r4",
            "product_id": "p2",
            "text": "Terrible battery life on this model. Avoid it."
        }
    ]

    all_chunks = []
    all_metadatas = []

    # 3. Process each review
    for rev in raw_reviews:
        # Clean text
        cleaned = cleaner.clean_text(rev["text"])
        
        # Detect language
        lang = cleaner.detect_language(cleaned)
        
        # Chunk text
        chunks = chunker.chunk_text(cleaned)
        
        for c_idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                "review_id": rev["review_id"],
                "product_id": rev["product_id"],
                "language": lang,
                "chunk_index": c_idx
            })

    # 4. Index in Vector Store
    retriever.add_documents(all_chunks, all_metadatas)
    
    # 5. Hybrid Search
    query = "battery life"
    
    # We expect english and spanish battery reviews for product p1
    results = retriever.search(query=query, product_id="p1", top_k=2)
    
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    
    # Asserting that both results belong to product p1
    for res in results:
        assert res["metadata"]["product_id"] == "p1"
        assert "battery" in res["content"].lower() or "batería" in res["content"].lower()
        
    print("Test passed! Results found:")
    for res in results:
        print(f"[{res['metadata']['language']}] {res['content']}")

if __name__ == "__main__":
    test_preprocessing_and_retrieval()
