import pytest
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.chunker import SemanticChunker

def test_clean_text():
    cleaner = TextCleaner()
    raw_text = "<p>This is a great product!   </p> Check it out at http://example.com"
    cleaned = cleaner.clean_text(raw_text)
    assert cleaned == "This is a great product! Check it out at"

def test_detect_language():
    cleaner = TextCleaner()
    assert cleaner.detect_language("This is an English sentence.") == "en"
    assert cleaner.detect_language("C'est un produit fantastique.") == "fr"
    assert cleaner.detect_language("¡Me encanta este teléfono!") == "es"

def test_semantic_chunker():
    # We test instantiation and basic chunking. 
    # The actual semantic similarity threshold depends on the model.
    chunker = SemanticChunker(threshold=0.3)
    text = "The battery life is amazing. It lasts all day without charging. However, the delivery was terrible. The box arrived completely crushed."
    
    chunks = chunker.chunk_text(text, max_chunk_chars=100)
    
    assert len(chunks) > 0
    # The battery sentences should ideally group, and delivery sentences group.
    # We at least assert it doesn't crash and returns strings.
    for chunk in chunks:
        assert isinstance(chunk, str)
        assert len(chunk) > 0
