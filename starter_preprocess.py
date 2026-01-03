"""
starter_preprocess.py
Starter code for text preprocessing - focus on the statistics, not the regex!
"""

import re
import json
from typing import List, Dict, Tuple
from collections import Counter
import string

class TextPreprocessor:
    """Handles all the annoying text cleaning so you can focus on the fun stuff"""
    
    def __init__(self):
        # Gutenberg markers (these are common, add more if needed)
        self.gutenberg_markers = [
            "*** START OF THIS PROJECT GUTENBERG",
            "*** END OF THIS PROJECT GUTENBERG",
            "*** START OF THE PROJECT GUTENBERG",
            "*** END OF THE PROJECT GUTENBERG",
            "*END*THE SMALL PRINT",
            "<<THIS ELECTRONIC VERSION"
        ]
    
    def clean_gutenberg_text(self, raw_text: str) -> str:
        """Remove Project Gutenberg headers/footers"""
        lines = raw_text.split('\n')
        
        # Find start and end markers
        start_idx = 0
        end_idx = len(lines)
        
        for i, line in enumerate(lines):
            if any(marker in line for marker in self.gutenberg_markers[:4]):
                if "START" in line:
                    start_idx = i + 1
                elif "END" in line:
                    end_idx = i
                    break
        
        # Join the cleaned lines
        cleaned = '\n'.join(lines[start_idx:end_idx])
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = re.sub(r' {2,}', ' ', cleaned)
        
        return cleaned.strip()
    
    def normalize_text(self, text: str, preserve_sentences: bool = True) -> str:
        """
        Normalize text while preserving sentence boundaries

        Args:
            text: Input text
            preserve_sentences: If True, keeps . ! ? for sentence detection
        """
        # Convert to lowercase
        text = text.lower()

        # Standardize quotes and dashes
        text = re.sub(r"[“”]", '"', text)       # fixed
        text = re.sub(r"[‘’´`]", "'", text)     # fixed
        text = re.sub(r"—|–", "-", text)

        # normalize newlines and spaces
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t\f\v]+", " ", text)

        if preserve_sentences:
            # Keep . ! ? but remove other punctuation
            text = re.sub(r"[^\w\s.!?\'-]", " ", text)
        else:
            # Remove all punctuation except apostrophes in contractions
            text = re.sub(r"(?<!\w)'(?!\w)|[^\w\s]", " ", text)

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def tokenize_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter (you can make this fancier with NLTK)
        sentences = re.split(r'[.!?]+', text)
        
        # Clean up and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def tokenize_words(self, text: str) -> List[str]:
        """Split text into words"""
        # Remove sentence endings for word tokenization
        text_for_words = re.sub(r'[.!?]', '', text)
        
        # Split on whitespace and filter empty strings
        words = text_for_words.split()
        words = [w for w in words if w]
        
        return words
    
    def tokenize_chars(self, text: str, include_space: bool = True) -> List[str]:
        """Split text into characters"""
        if include_space:
            # Replace multiple spaces with single space
            text = re.sub(r'\s+', ' ', text)
            return list(text)
        else:
            return [c for c in text if c != ' ']
    
    def get_sentence_lengths(self, sentences: List[str]) -> List[int]:
        """Get word count for each sentence"""
        return [len(self.tokenize_words(sent)) for sent in sentences]
    

class FrequencyAnalyzer:
    """Calculate n-gram frequencies from tokenized text"""
    
    def calculate_ngrams(self, tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
        """
        Calculate n-gram frequencies
        
        Args:
            tokens: List of tokens (words or characters)
            n: Size of n-gram (1=unigram, 2=bigram, 3=trigram)
        
        Returns:
            Dictionary mapping n-grams to their counts
        """
        if n == 1:
            # Special case for unigrams (return as single strings, not tuples)
            return dict(Counter(tokens))
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.append(ngram)
        
        return dict(Counter(ngrams))
    
    def calculate_probabilities(self, ngram_counts: Dict, smoothing: float = 0.0) -> Dict:
        """
        Convert counts to probabilities
        
        Args:
            ngram_counts: Dictionary of n-gram counts
            smoothing: Laplace smoothing parameter (0 = no smoothing)
        """
        total = sum(ngram_counts.values()) + smoothing * len(ngram_counts)
        
        probabilities = {}
        for ngram, count in ngram_counts.items():
            probabilities[ngram] = (count + smoothing) / total
        
        return probabilities
    
    def save_frequencies(self, frequencies: Dict, filename: str):
        """Save frequency dictionary to JSON file"""
        # Convert tuples to strings for JSON serialization
        json_friendly = {}
        for key, value in frequencies.items():
            if isinstance(key, tuple):
                json_friendly['||'.join(key)] = value
            else:
                json_friendly[key] = value
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_friendly, f, indent=2, ensure_ascii=False)
    
    def load_frequencies(self, filename: str) -> Dict:
        """Load frequency dictionary from JSON file"""
        with open(filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # Convert string keys back to tuples where needed
        frequencies = {}
        for key, value in json_data.items():
            if '||' in key:
                frequencies[tuple(key.split('||'))] = value
            else:
                frequencies[key] = value
        
        return frequencies


# Example usage to test your setup
if __name__ == "__main__":
    # Test with a small example
    sample_text = """
    This is a test. This is only a test! 
    If this were a real emergency, you would be informed.
    """
    
    preprocessor = TextPreprocessor()
    analyzer = FrequencyAnalyzer()
    
    # Clean and normalize
    clean_text = preprocessor.normalize_text(sample_text)
    print(f"Cleaned text: {clean_text}\n")
    
    # Get sentences
    sentences = preprocessor.tokenize_sentences(clean_text)
    print(f"Sentences: {sentences}\n")
    
    # Get words
    words = preprocessor.tokenize_words(clean_text)
    print(f"Words: {words}\n")
    
    # Calculate bigrams
    bigrams = analyzer.calculate_ngrams(words, 2)
    print(f"Word bigrams: {bigrams}\n")
    
    # Calculate character trigrams
    chars = preprocessor.tokenize_chars(clean_text)
    char_trigrams = analyzer.calculate_ngrams(chars, 3)
    print(f"Character trigrams (first 5): {dict(list(char_trigrams.items())[:5])}")