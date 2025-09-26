#!/usr/bin/env python3
"""
Word Neighborhood Analysis Tool

Takes news text as input and shows top 10 neighbors for each word
from both real and fake news vocabulary subgraphs.
"""

import sys
sys.path.append('.')

from load_data_vocab import split_news_by_binary_label, tokenizer
import pickle
import numpy as np
from collections import Counter, defaultdict
import json

class VocabularyGraphBuilder:
    """Build vocabulary graphs from news corpora."""
    
    def __init__(self, window_size=5, min_cooccur=3):
        self.window_size = window_size
        self.min_cooccur = min_cooccur
        self.real_graph = {}
        self.fake_graph = {}
        self.real_word_counts = Counter()
        self.fake_word_counts = Counter()
    
    def build_cooccurrence_graph(self, texts, min_word_freq=10, remove_stop_words=False, remove_punct=True):
        """Build co-occurrence graph from texts with flexible filtering options."""
        from nltk.corpus import stopwords
        import string
        
        # Get stop words if needed
        if remove_stop_words:
            try:
                stop_words = set(stopwords.words('english'))
            except:
                print("NLTK stopwords not found, using basic stop word list")
                stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
        else:
            stop_words = set()
        
        # Punctuation to remove
        if remove_punct:
            punct_to_remove = set(string.punctuation) | {"''", "``", "'s", "'t", "'re", "'ve", "'ll", "'d"}
        else:
            punct_to_remove = set()
        
        cooccur_counts = defaultdict(Counter)
        word_counts = Counter()
        
        print(f"Processing {len(texts)} texts...")
        print(f"Filters: Remove stop words: {remove_stop_words}, Remove punctuation: {remove_punct}")
        
        for i, text in enumerate(texts):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(texts)} texts")
            
            # Tokenize and clean
            tokens = tokenizer.tokenize(text.lower())
            
            # Apply filters
            filtered_tokens = []
            for token in tokens:
                # Skip if too short
                if len(token) < 2:
                    continue
                    
                # Skip punctuation if requested
                if remove_punct and token in punct_to_remove:
                    continue
                    
                # Skip stop words if requested
                if remove_stop_words and token in stop_words:
                    continue
                    
                # Keep alphabetic tokens and some punctuation
                if token.isalpha() or (not remove_punct and token in {'.', ',', ';', ':', '!', '?'}):
                    filtered_tokens.append(token)
            
            # Count word frequencies
            word_counts.update(filtered_tokens)
            
            # Build co-occurrence matrix
            for j, word in enumerate(filtered_tokens):
                start = max(0, j - self.window_size)
                end = min(len(filtered_tokens), j + self.window_size + 1)
                
                for k in range(start, end):
                    if k != j:
                        neighbor = filtered_tokens[k]
                        cooccur_counts[word][neighbor] += 1
        
        # Filter by minimum frequency
        frequent_words = {word for word, count in word_counts.items() 
                         if count >= min_word_freq}
        
        print(f"Vocabulary size after filtering: {len(frequent_words)}")
        
        # Build final graph with probability distributions
        graph = {}
        for word in frequent_words:
            if word in cooccur_counts:
                # Get co-occurring words and their counts
                neighbors = cooccur_counts[word]
                total_cooccur = sum(neighbors.values())
                
                if total_cooccur >= self.min_cooccur:
                    # Convert to probabilities
                    prob_dist = {
                        neighbor: count / total_cooccur 
                        for neighbor, count in neighbors.items()
                        if count >= 2  # Filter rare co-occurrences
                    }
                    
                    if len(prob_dist) >= 5:  # Need reasonable neighborhood size
                        graph[word] = prob_dist
        
        return graph, word_counts
    
    def build_graphs_from_data(self, dataset='politifact', remove_stop_words=False, remove_punct=True):
        """Build both real and fake news graphs with filtering options."""
        print(f"Loading {dataset} data...")
        train_dict = pickle.load(open(f'data/news_articles/{dataset}_train.pkl', 'rb'))
        news_real, news_fake = split_news_by_binary_label(train_dict)
        
        print(f"Building real news graph from {len(news_real)} articles...")
        self.real_graph, self.real_word_counts = self.build_cooccurrence_graph(
            news_real, remove_stop_words=remove_stop_words, remove_punct=remove_punct)
        
        print(f"Building fake news graph from {len(news_fake)} articles...")
        self.fake_graph, self.fake_word_counts = self.build_cooccurrence_graph(
            news_fake, remove_stop_words=remove_stop_words, remove_punct=remove_punct)
        
        print(f"Real graph vocabulary size: {len(self.real_graph)}")
        print(f"Fake graph vocabulary size: {len(self.fake_graph)}")
        
        # Save graphs for reuse
        suffix = ""
        if remove_stop_words:
            suffix += "_no_stopwords"
        if remove_punct:
            suffix += "_no_punct"
        
        graphs_data = {
            'real_graph': self.real_graph,
            'fake_graph': self.fake_graph,
            'real_word_counts': dict(self.real_word_counts),
            'fake_word_counts': dict(self.fake_word_counts),
            'dataset': dataset,
            'window_size': self.window_size,
            'remove_stop_words': remove_stop_words,
            'remove_punct': remove_punct
        }
        
        filename = f'vocabulary_graphs_{dataset}{suffix}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(graphs_data, f)
        
        print(f"Graphs saved to {filename}")
    
    def load_graphs(self, dataset='politifact', remove_stop_words=False, remove_punct=True):
        """Load pre-built graphs with specific filtering options."""
        suffix = ""
        if remove_stop_words:
            suffix += "_no_stopwords"
        if remove_punct:
            suffix += "_no_punct"
        
        filename = f'vocabulary_graphs_{dataset}{suffix}.pkl'
        
        try:
            with open(filename, 'rb') as f:
                graphs_data = pickle.load(f)
            
            self.real_graph = graphs_data['real_graph']
            self.fake_graph = graphs_data['fake_graph']
            self.real_word_counts = Counter(graphs_data['real_word_counts'])
            self.fake_word_counts = Counter(graphs_data['fake_word_counts'])
            
            print(f"Loaded graphs for {dataset} (stop_words={remove_stop_words}, punct={remove_punct})")
            print(f"Real graph vocabulary: {len(self.real_graph)} words")
            print(f"Fake graph vocabulary: {len(self.fake_graph)} words")
            return True
        
        except FileNotFoundError:
            print(f"No pre-built graphs found: {filename}")
            print("Run build_graphs_from_data() with matching parameters first.")
            return False

class WordNeighborhoodAnalyzer:
    """Analyze word neighborhoods in input text."""
    
    def __init__(self, graph_builder):
        self.graph_builder = graph_builder
    
    def get_top_neighbors(self, word, graph, top_k=10):
        """Get top K neighbors for a word from a graph."""
        if word not in graph:
            return []
        
        neighbors = graph[word]
        sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
        return sorted_neighbors[:top_k]
    
    def analyze_text(self, input_text, show_details=True):
        """Analyze word neighborhoods for all words in input text, showing only neighbors present in the text."""
        print(f"\n" + "="*80)
        print("WORD NEIGHBORHOOD ANALYSIS - FILTERED TO TEXT VOCABULARY")
        print("="*80)
        
        # Tokenize input text
        tokens = tokenizer.tokenize(input_text.lower())
        tokens = [t for t in tokens if t.isalpha() and len(t) > 2]
        unique_words = sorted(set(tokens))
        text_vocab = set(unique_words)
        
        print(f"Input text contains {len(tokens)} tokens, {len(unique_words)} unique words")
        print(f"Analyzing neighborhoods for {len(unique_words)} words...")
        print(f"Filtering neighbors to only show words present in input text")
        
        results = {}
        
        for word in unique_words:
            # Get all neighbors from graphs
            all_real_neighbors = self.get_top_neighbors(word, self.graph_builder.real_graph, top_k=50)
            all_fake_neighbors = self.get_top_neighbors(word, self.graph_builder.fake_graph, top_k=50)
            
            # Filter to only neighbors that appear in the input text
            real_neighbors = [(neighbor, prob) for neighbor, prob in all_real_neighbors 
                             if neighbor in text_vocab and neighbor != word]
            fake_neighbors = [(neighbor, prob) for neighbor, prob in all_fake_neighbors 
                             if neighbor in text_vocab and neighbor != word]
            
            # Take top 10 from filtered results
            real_neighbors = real_neighbors[:10]
            fake_neighbors = fake_neighbors[:10]
            
            # Get word frequencies in each corpus
            real_freq = self.graph_builder.real_word_counts.get(word, 0)
            fake_freq = self.graph_builder.fake_word_counts.get(word, 0)
            
            results[word] = {
                'real_neighbors': real_neighbors,
                'fake_neighbors': fake_neighbors,
                'real_frequency': real_freq,
                'fake_frequency': fake_freq,
                'in_real_graph': word in self.graph_builder.real_graph,
                'in_fake_graph': word in self.graph_builder.fake_graph,
                'neighbors_in_text': True  # Flag to indicate filtering was applied
            }
            
            if show_details:
                self._print_word_analysis(word, results[word])
        
        return results
    
    def _print_word_analysis(self, word, analysis):
        """Print detailed analysis for a single word."""
        print(f"\n{'-'*60}")
        print(f"WORD: '{word.upper()}'")
        print(f"{'-'*60}")
        
        print(f"Frequency - Real: {analysis['real_frequency']:,}, "
              f"Fake: {analysis['fake_frequency']:,}")
        
        print(f"In graphs - Real: {analysis['in_real_graph']}, "
              f"Fake: {analysis['in_fake_graph']}")
        
        if analysis['real_neighbors'] or analysis['fake_neighbors']:
            print(f"\nTOP NEIGHBORS (from words in this text only):")
            
            # Print side by side
            real_neighbors = analysis['real_neighbors']
            fake_neighbors = analysis['fake_neighbors']
            
            print(f"{'REAL NEWS CONTEXT':<35} | {'FAKE NEWS CONTEXT':<35}")
            print(f"{'-'*35}|{'-'*36}")
            
            max_len = max(len(real_neighbors), len(fake_neighbors))
            
            if max_len == 0:
                print("No in-text neighbors found in either context")
            else:
                for i in range(max_len):
                    real_str = ""
                    fake_str = ""
                    
                    if i < len(real_neighbors):
                        neighbor, prob = real_neighbors[i]
                        real_str = f"{i+1:2d}. {neighbor:<20} ({prob:.4f})"
                    
                    if i < len(fake_neighbors):
                        neighbor, prob = fake_neighbors[i]
                        fake_str = f"{i+1:2d}. {neighbor:<20} ({prob:.4f})"
                    
                    print(f"{real_str:<35}| {fake_str:<35}")
        
        else:
            print("No neighbors found in either graph (word too rare)")
    
    def analyze_text_with_summary(self, input_text):
        """Analyze text and provide a summary of differential patterns."""
        results = self.analyze_text(input_text, show_details=True)
        
        print(f"\n" + "="*80)
        print("SUMMARY: DIFFERENTIAL WORD ASSOCIATION PATTERNS")
        print("="*80)
        
        words_with_neighbors = {word: data for word, data in results.items() 
                              if data['real_neighbors'] or data['fake_neighbors']}
        
        if not words_with_neighbors:
            print("No intra-text word associations found.")
            return results
        
        print(f"Words with intra-text associations: {len(words_with_neighbors)}/{len(results)}")
        
        # Find words with different patterns
        differential_patterns = []
        similar_patterns = []
        
        for word, data in words_with_neighbors.items():
            real_neighbors = set(n[0] for n in data['real_neighbors'])
            fake_neighbors = set(n[0] for n in data['fake_neighbors'])
            
            if real_neighbors and fake_neighbors:
                overlap = len(real_neighbors & fake_neighbors)
                total_unique = len(real_neighbors | fake_neighbors)
                overlap_ratio = overlap / total_unique if total_unique > 0 else 0
                
                if overlap_ratio < 0.5:
                    differential_patterns.append((word, overlap_ratio, real_neighbors, fake_neighbors))
                else:
                    similar_patterns.append((word, overlap_ratio))
        
        if differential_patterns:
            print(f"\nWords showing DIFFERENTIAL patterns (different neighbors in real vs fake):")
            for word, overlap, real_set, fake_set in differential_patterns:
                print(f"  • {word}: {overlap:.2f} overlap")
                print(f"    Real context: {list(real_set)[:5]}")
                print(f"    Fake context: {list(fake_set)[:5]}")
        
        if similar_patterns:
            print(f"\nWords showing SIMILAR patterns in both contexts: {len(similar_patterns)}")
            for word, overlap in similar_patterns[:5]:
                print(f"  • {word}: {overlap:.2f} overlap")
        
        return results
    
    def compare_neighborhoods(self, word1, word2):
        """Compare neighborhoods of two words."""
        print(f"\n" + "="*80)
        print(f"COMPARING NEIGHBORHOODS: '{word1}' vs '{word2}'")
        print("="*80)
        
        for word in [word1, word2]:
            real_neighbors = self.get_top_neighbors(word, self.graph_builder.real_graph, 5)
            fake_neighbors = self.get_top_neighbors(word, self.graph_builder.fake_graph, 5)
            
            print(f"\nWord: '{word}'")
            print(f"Real top 5: {[n[0] for n in real_neighbors]}")
            print(f"Fake top 5: {[n[0] for n in fake_neighbors]}")
    
    def find_differential_words(self, threshold=0.01):
        """Find words with very different neighborhoods in real vs fake graphs."""
        print(f"\nFinding words with differential neighborhoods...")
        
        differential_words = []
        common_words = set(self.graph_builder.real_graph.keys()) & set(self.graph_builder.fake_graph.keys())
        
        for word in common_words:
            real_neighbors = dict(self.get_top_neighbors(word, self.graph_builder.real_graph, 10))
            fake_neighbors = dict(self.get_top_neighbors(word, self.graph_builder.fake_graph, 10))
            
            # Calculate overlap
            real_top_words = set(real_neighbors.keys())
            fake_top_words = set(fake_neighbors.keys())
            
            if len(real_top_words) >= 5 and len(fake_top_words) >= 5:
                overlap = len(real_top_words & fake_top_words)
                total_unique = len(real_top_words | fake_top_words)
                
                if total_unique > 0:
                    overlap_ratio = overlap / total_unique
                    
                    if overlap_ratio < threshold:  # Low overlap = high difference
                        differential_words.append((word, overlap_ratio, real_top_words, fake_top_words))
        
        # Sort by most differential (lowest overlap)
        differential_words.sort(key=lambda x: x[1])
        
        print(f"Found {len(differential_words)} words with neighborhood overlap < {threshold}")
        
        # Print top 10 most differential
        print(f"\nTop 10 most differential words:")
        for i, (word, overlap, real_words, fake_words) in enumerate(differential_words[:10]):
            print(f"{i+1:2d}. {word:<15} (overlap: {overlap:.3f})")
            print(f"    Real: {list(real_words)[:5]}")
            print(f"    Fake: {list(fake_words)[:5]}")
        
        return differential_words

def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze word neighborhoods in news text')
    parser.add_argument('--dataset', choices=['politifact', 'gossipcop', 'lun'], 
                       default='politifact', help='Dataset to use')
    parser.add_argument('--build-graphs', action='store_true',
                       help='Build vocabulary graphs from data')
    parser.add_argument('--remove-stop-words', action='store_true',
                       help='Remove stop words when building graphs')
    parser.add_argument('--remove-punct', action='store_true', default=True,
                       help='Remove punctuation when building graphs (default: True)')
    parser.add_argument('--keep-punct', action='store_true',
                       help='Keep punctuation when building graphs')
    parser.add_argument('--text', type=str,
                       help='Input text to analyze')
    parser.add_argument('--file', type=str,
                       help='File containing text to analyze')
    parser.add_argument('--compare', nargs=2,
                       help='Compare neighborhoods of two words')
    parser.add_argument('--find-differential', action='store_true',
                       help='Find words with most differential neighborhoods')
    parser.add_argument('--summary', action='store_true',
                       help='Show summary analysis of text patterns')
    
    args = parser.parse_args()
    
    # Handle punctuation settings
    remove_punct = args.remove_punct
    if args.keep_punct:
        remove_punct = False
    
    # Initialize graph builder
    graph_builder = VocabularyGraphBuilder(window_size=5, min_cooccur=3)
    
    # Build or load graphs
    if args.build_graphs:
        graph_builder.build_graphs_from_data(
            args.dataset, 
            remove_stop_words=args.remove_stop_words,
            remove_punct=remove_punct
        )
    else:
        if not graph_builder.load_graphs(
            args.dataset,
            remove_stop_words=args.remove_stop_words,
            remove_punct=remove_punct
        ):
            print("Please run with --build-graphs first, or ensure graph files exist")
            return
    
    # Initialize analyzer
    analyzer = WordNeighborhoodAnalyzer(graph_builder)
    
    # Handle different modes
    if args.compare:
        analyzer.compare_neighborhoods(args.compare[0], args.compare[1])
    
    elif args.find_differential:
        differential_words = analyzer.find_differential_words()
    
    elif args.text:
        if args.summary:
            results = analyzer.analyze_text_with_summary(args.text)
        else:
            results = analyzer.analyze_text(args.text)
    
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text_content = f.read()
            if args.summary:
                results = analyzer.analyze_text_with_summary(text_content)
            else:
                results = analyzer.analyze_text(text_content)
        except FileNotFoundError:
            print(f"File not found: {args.file}")
    
    else:
        # Interactive mode
        print("Interactive mode - enter text to analyze (or 'quit' to exit)")
        
        while True:
            user_input = input("\nEnter text: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input:
                if args.summary:
                    results = analyzer.analyze_text_with_summary(user_input)
                else:
                    results = analyzer.analyze_text(user_input, show_details=True)
            else:
                print("Please enter some text to analyze")

# Example usage functions
def example_usage():
    """Example of how to use the tool."""
    
    # Build/load graphs
    graph_builder = VocabularyGraphBuilder()
    
    # Try to load existing graphs, build if they don't exist
    if not graph_builder.load_graphs('politifact'):
        print("Building graphs from PolitiFact data...")
        graph_builder.build_graphs_from_data('politifact')
    
    # Initialize analyzer
    analyzer = WordNeighborhoodAnalyzer(graph_builder)
    
    # Example 1: Analyze a news headline
    sample_text = "President announces new government officials investigation according to sources"
    print("Analyzing sample text:")
    results = analyzer.analyze_text(sample_text)
    
    # Example 2: Compare two words
    print("\nComparing 'president' and 'government':")
    analyzer.compare_neighborhoods('president', 'government')
    
    # Example 3: Find most differential words
    print("\nFinding words with most different neighborhoods:")
    differential_words = analyzer.find_differential_words(threshold=0.3)

if __name__ == "__main__":
    main()