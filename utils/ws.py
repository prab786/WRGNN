import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import torch
from transformers import BertTokenizer
from scipy import stats
import random

class StrategicWordSelector:
    """
    Strategic word selection for comprehensive validation of the Differential Distributional Hypothesis.
    
    This class implements multiple selection strategies to ensure robust testing across
    different word categories and linguistic patterns.
    """
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained('bert-base-uncased')
        self.word_categories = {}
        self.selection_results = {}
        
    def comprehensive_word_selection(self, real_texts, fake_texts, target_size=50):
        """
        Select words using multiple strategic approaches to ensure comprehensive coverage.
        
        Args:
            real_texts: List of real news texts
            fake_texts: List of fake news texts
            target_size: Target number of words to select
            
        Returns:
            Dict with selected words organized by selection strategy
        """
        
        print(f"🎯 Selecting {target_size} words using comprehensive strategy...")
        
        # Strategy 1: High-frequency words (journalism basics)
        journalism_words = self._select_journalism_words(real_texts, fake_texts, n=12)
        
        # Strategy 2: Domain-specific terms
        domain_words = self._select_domain_specific_words(real_texts, fake_texts, n=10)
        
        # Strategy 3: Attribution and sourcing words
        attribution_words = self._select_attribution_words(real_texts, fake_texts, n=8)
        
        # Strategy 4: Emotional/sentiment words
        emotional_words = self._select_emotional_words(real_texts, fake_texts, n=8)
        
        # Strategy 5: Statistical divergence-based selection
        divergent_words = self._select_by_statistical_divergence(real_texts, fake_texts, n=8)
        
        # Strategy 6: Random baseline words
        random_words = self._select_random_baseline(real_texts, fake_texts, n=4)
        
        # Combine and deduplicate
        all_selected = {
            'journalism': journalism_words,
            'domain_specific': domain_words,
            'attribution': attribution_words,
            'emotional': emotional_words,
            'statistical_divergent': divergent_words,
            'random_baseline': random_words
        }
        
        # Create final deduplicated list
        final_words = self._create_final_word_list(all_selected, target_size)
        
        return {
            'category_selections': all_selected,
            'final_word_list': final_words,
            'selection_rationale': self._generate_selection_rationale(all_selected, final_words)
        }
    
    def _select_journalism_words(self, real_texts, fake_texts, n=12):
        """Select core journalism and news reporting words."""
        
        journalism_candidates = [
            # Attribution words
            'according', 'sources', 'officials', 'spokesman', 'spokesperson',
            'authorities', 'experts', 'analysts', 'investigators',
            
            # Reporting verbs
            'reported', 'confirmed', 'announced', 'stated', 'revealed',
            'disclosed', 'indicated', 'suggested', 'claimed', 'alleged',
            
            # Institutional words
            'government', 'administration', 'department', 'agency', 'bureau',
            'committee', 'commission', 'council', 'board',
            
            # Process words
            'investigation', 'statement', 'report', 'study', 'analysis',
            'findings', 'evidence', 'documents', 'records'
        ]
        
        # Filter by frequency and relevance
        selected = self._filter_by_frequency_and_relevance(
            journalism_candidates, real_texts, fake_texts, n
        )
        
        return selected
    
    def _select_domain_specific_words(self, real_texts, fake_texts, n=10):
        """Select domain-specific words relevant to your datasets."""
        
        # Political domain (for PolitiFact)
        political_words = [
            'president', 'congress', 'senate', 'house', 'senator',
            'representative', 'politician', 'campaign', 'election',
            'voters', 'policy', 'legislation', 'bill', 'law'
        ]
        
        # Entertainment domain (for GossipCop)
        entertainment_words = [
            'celebrity', 'actor', 'actress', 'movie', 'film',
            'director', 'producer', 'studio', 'premiere', 'awards',
            'performance', 'character', 'role', 'series'
        ]
        
        # Health/general news
        health_words = [
            'health', 'medical', 'doctor', 'hospital', 'patients',
            'treatment', 'symptoms', 'vaccine', 'drug', 'therapy'
        ]
        
        all_domain_words = political_words + entertainment_words + health_words
        
        selected = self._filter_by_frequency_and_relevance(
            all_domain_words, real_texts, fake_texts, n
        )
        
        return selected
    
    def _select_attribution_words(self, real_texts, fake_texts, n=8):
        """Select words specifically related to information attribution."""
        
        attribution_candidates = [
            # Direct attribution
            'said', 'told', 'explained', 'noted', 'added', 'continued',
            'emphasized', 'insisted', 'maintained', 'argued',
            
            # Indirect attribution  
            'according', 'citing', 'quoting', 'referencing', 'based',
            
            # Uncertainty markers
            'allegedly', 'reportedly', 'supposedly', 'apparently',
            'presumably', 'seemingly',
            
            # Certainty markers
            'confirmed', 'verified', 'established', 'proven', 'documented'
        ]
        
        selected = self._filter_by_frequency_and_relevance(
            attribution_candidates, real_texts, fake_texts, n
        )
        
        return selected
    
    def _select_emotional_words(self, real_texts, fake_texts, n=8):
        """Select emotional and sentiment-bearing words."""
        
        emotional_candidates = [
            # Strong positive
            'amazing', 'incredible', 'fantastic', 'wonderful', 'excellent',
            
            # Strong negative
            'terrible', 'awful', 'horrible', 'devastating', 'shocking',
            'outrageous', 'disgusting', 'unbelievable',
            
            # Intensity markers
            'extremely', 'totally', 'completely', 'absolutely', 'definitely',
            
            # Conspiracy/sensational
            'secret', 'hidden', 'conspiracy', 'cover', 'expose', 'reveal',
            'truth', 'lies', 'deception', 'manipulation'
        ]
        
        selected = self._filter_by_frequency_and_relevance(
            emotional_candidates, real_texts, fake_texts, n
        )
        
        return selected
    
    def _select_by_statistical_divergence(self, real_texts, fake_texts, n=8):
        """Select words based on preliminary statistical analysis."""
        
        print("🔍 Computing preliminary divergences for statistical selection...")
        
        # Get vocabulary from both corpora
        real_vocab = self._extract_vocabulary(real_texts[:100])  # Sample for efficiency
        fake_vocab = self._extract_vocabulary(fake_texts[:100])
        common_vocab = real_vocab & fake_vocab
        
        # Filter for reasonable words
        filtered_vocab = [
            word for word in common_vocab 
            if len(word) > 3 and word.isalpha() and not word.startswith('##')
        ]
        
        # Compute quick divergence estimates
        divergence_estimates = {}
        
        for word in filtered_vocab[:200]:  # Limit for computational efficiency
            try:
                # Quick co-occurrence analysis
                real_cooccur = self._quick_cooccurrence_analysis(word, real_texts[:50])
                fake_cooccur = self._quick_cooccurrence_analysis(word, fake_texts[:50])
                
                if real_cooccur and fake_cooccur:
                    # Simple divergence estimate
                    divergence = self._estimate_divergence(real_cooccur, fake_cooccur)
                    divergence_estimates[word] = divergence
                    
            except:
                continue
        
        # Select top divergent words
        top_divergent = sorted(divergence_estimates.items(), 
                              key=lambda x: x[1], reverse=True)[:n]
        
        selected = [word for word, div in top_divergent]
        
        print(f"📊 Selected {len(selected)} statistically divergent words")
        
        return selected
    
    def _select_random_baseline(self, real_texts, fake_texts, n=4):
        """Select random words as baseline/control."""
        
        # Get common vocabulary
        real_vocab = self._extract_vocabulary(real_texts[:100])
        fake_vocab = self._extract_vocabulary(fake_texts[:100])
        common_vocab = real_vocab & fake_vocab
        
        # Filter for reasonable words
        filtered_vocab = [
            word for word in common_vocab 
            if len(word) > 3 and word.isalpha() and not word.startswith('##')
        ]
        
        # Random selection
        selected = random.sample(filtered_vocab, min(n, len(filtered_vocab)))
        
        return selected
    
    def _filter_by_frequency_and_relevance(self, candidates, real_texts, fake_texts, n):
        """Filter candidate words by frequency and relevance."""
        
        # Get frequency counts
        real_freq = self._get_word_frequencies(candidates, real_texts)
        fake_freq = self._get_word_frequencies(candidates, fake_texts)
        
        # Score words by combined frequency and presence in both corpora
        word_scores = {}
        for word in candidates:
            real_count = real_freq.get(word, 0)
            fake_count = fake_freq.get(word, 0)
            
            # Must appear in both corpora with minimum frequency
            if real_count >= 5 and fake_count >= 5:
                # Score by geometric mean of frequencies
                score = np.sqrt(real_count * fake_count)
                word_scores[word] = score
        
        # Select top-scoring words
        top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        selected = [word for word, score in top_words]
        
        return selected
    
    def _extract_vocabulary(self, texts):
        """Extract vocabulary from texts."""
        vocab = set()
        for text in texts:
            tokens = self.tokenizer.tokenize(text.lower())
            vocab.update(tokens)
        return vocab
    
    def _get_word_frequencies(self, words, texts):
        """Get frequency counts for specific words in texts."""
        frequencies = defaultdict(int)
        
        for text in texts:
            tokens = self.tokenizer.tokenize(text.lower())
            for word in words:
                if word in tokens:
                    frequencies[word] += tokens.count(word)
        
        return frequencies
    
    def _quick_cooccurrence_analysis(self, word, texts):
        """Quick co-occurrence analysis for preliminary divergence estimation."""
        cooccurrences = defaultdict(int)
        total_contexts = 0
        
        for text in texts:
            tokens = self.tokenizer.tokenize(text.lower())
            if word in tokens:
                word_indices = [i for i, token in enumerate(tokens) if token == word]
                for idx in word_indices:
                    # Simple window-based co-occurrence
                    start = max(0, idx - 5)
                    end = min(len(tokens), idx + 6)
                    context = tokens[start:end]
                    
                    for coword in context:
                        if coword != word:
                            cooccurrences[coword] += 1
                    total_contexts += 1
        
        # Convert to probabilities
        if total_contexts > 0:
            probs = {word: count/total_contexts for word, count in cooccurrences.items()}
            return probs
        return {}
    
    def _estimate_divergence(self, real_cooccur, fake_cooccur):
        """Estimate divergence between co-occurrence distributions."""
        
        all_words = set(real_cooccur.keys()) | set(fake_cooccur.keys())
        
        if len(all_words) < 2:
            return 0.0
        
        # Simple KL divergence estimate with smoothing
        divergence = 0.0
        for word in all_words:
            p = real_cooccur.get(word, 0) + 1e-10
            q = fake_cooccur.get(word, 0) + 1e-10
            divergence += p * np.log(p / q)
        
        return divergence
    
    def _create_final_word_list(self, category_selections, target_size):
        """Create final deduplicated word list."""
        
        # Collect all words with their categories
        word_to_categories = defaultdict(list)
        
        for category, words in category_selections.items():
            for word in words:
                word_to_categories[word].append(category)
        
        # Priority order for categories
        category_priority = {
            'journalism': 1,
            'attribution': 2,
            'domain_specific': 3,
            'statistical_divergent': 4,
            'emotional': 5,
            'random_baseline': 6
        }
        
        # Sort words by category priority
        sorted_words = sorted(
            word_to_categories.items(),
            key=lambda x: min(category_priority[cat] for cat in x[1])
        )
        
        # Select final words
        final_words = []
        for word, categories in sorted_words:
            if len(final_words) < target_size:
                final_words.append({
                    'word': word,
                    'categories': categories,
                    'primary_category': min(categories, key=lambda c: category_priority[c])
                })
        
        return final_words[:target_size]
    
    def _generate_selection_rationale(self, category_selections, final_words):
        """Generate rationale for word selection."""
        
        rationale = {
            'total_words': len(final_words),
            'category_distribution': {},
            'selection_criteria': {
                'journalism_words': 'Core news reporting and attribution terms',
                'domain_specific': 'Terms specific to political and entertainment domains',
                'attribution': 'Words related to information sourcing and attribution',
                'emotional': 'Sentiment and emotional intensity markers',
                'statistical_divergent': 'Words with preliminary high divergence estimates',
                'random_baseline': 'Random control words for baseline comparison'
            }
        }
        
        # Count distribution
        for word_info in final_words:
            primary_cat = word_info['primary_category']
            rationale['category_distribution'][primary_cat] = \
                rationale['category_distribution'].get(primary_cat, 0) + 1
        
        return rationale
    
    def validate_word_selection(self, selected_words, real_texts, fake_texts):
        """Validate the quality of word selection."""
        
        print("🔍 Validating word selection quality...")
        
        validation_results = {
            'frequency_analysis': {},
            'coverage_analysis': {},
            'diversity_analysis': {}
        }
        
        # Frequency validation
        for word_info in selected_words:
            word = word_info['word']
            
            real_freq = self._count_word_frequency(word, real_texts)
            fake_freq = self._count_word_frequency(word, fake_texts)
            
            validation_results['frequency_analysis'][word] = {
                'real_frequency': real_freq,
                'fake_frequency': fake_freq,
                'min_frequency': min(real_freq, fake_freq),
                'sufficient_data': real_freq >= 10 and fake_freq >= 10
            }
        
        # Coverage analysis
        total_words = len(selected_words)
        sufficient_data_count = sum(
            1 for word, freq in validation_results['frequency_analysis'].items()
            if freq['sufficient_data']
        )
        
        validation_results['coverage_analysis'] = {
            'total_selected': total_words,
            'sufficient_data': sufficient_data_count,
            'coverage_ratio': sufficient_data_count / total_words,
            'recommendation': 'good' if sufficient_data_count / total_words > 0.8 else 'needs_improvement'
        }
        
        # Diversity analysis
        categories = [word_info['primary_category'] for word_info in selected_words]
        category_counts = Counter(categories)
        
        validation_results['diversity_analysis'] = {
            'category_distribution': dict(category_counts),
            'num_categories': len(category_counts),
            'balanced': max(category_counts.values()) / min(category_counts.values()) < 3
        }
        
        return validation_results
    
    def _count_word_frequency(self, word, texts):
        """Count frequency of word in texts."""
        count = 0
        for text in texts:
            tokens = self.tokenizer.tokenize(text.lower())
            count += tokens.count(word)
        return count
    
    def generate_selection_report(self, selection_results, validation_results):
        """Generate comprehensive selection report."""
        
        print("\n" + "="*80)
        print("📋 WORD SELECTION REPORT")
        print("="*80)
        
        final_words = selection_results['final_word_list']
        rationale = selection_results['selection_rationale']
        
        print(f"\n📊 SELECTION SUMMARY:")
        print(f"Total words selected: {len(final_words)}")
        print(f"Category distribution:")
        for category, count in rationale['category_distribution'].items():
            print(f"  {category}: {count} words")
        
        print(f"\n🔍 VALIDATION RESULTS:")
        coverage = validation_results['coverage_analysis']
        print(f"Words with sufficient data: {coverage['sufficient_data']}/{coverage['total_selected']}")
        print(f"Coverage ratio: {coverage['coverage_ratio']:.2f}")
        print(f"Quality assessment: {coverage['recommendation']}")
        
        print(f"\n📝 SELECTED WORDS BY CATEGORY:")
        
        # Group by category
        by_category = defaultdict(list)
        for word_info in final_words:
            by_category[word_info['primary_category']].append(word_info['word'])
        
        for category, words in by_category.items():
            print(f"\n{category.upper().replace('_', ' ')}:")
            for word in words:
                freq_info = validation_results['frequency_analysis'][word]
                status = "✅" if freq_info['sufficient_data'] else "⚠️"
                print(f"  {status} {word} (R:{freq_info['real_frequency']}, F:{freq_info['fake_frequency']})")
        
        print(f"\n💡 SELECTION RATIONALE:")
        for category, description in rationale['selection_criteria'].items():
            if category in rationale['category_distribution']:
                print(f"  {category}: {description}")
        
        return True


def run_comprehensive_word_selection(real_texts, fake_texts, target_size=50):
    """
    Main function to run comprehensive word selection.
    
    Args:
        real_texts: List of real news texts
        fake_texts: List of fake news texts
        target_size: Target number of words to select
    
    Returns:
        Dictionary with selection results and validation
    """
    
    print("🎯 Starting Comprehensive Word Selection Process...")
    
    # Initialize selector
    selector = StrategicWordSelector()
    
    # Run selection
    selection_results = selector.comprehensive_word_selection(
        real_texts, fake_texts, target_size
    )
    
    # Validate selection
    validation_results = selector.validate_word_selection(
        selection_results['final_word_list'], real_texts, fake_texts
    )
    
    # Generate report
    selector.generate_selection_report(selection_results, validation_results)
    
    # Extract final word list
    final_words = [word_info['word'] for word_info in selection_results['final_word_list']]
    
    return {
        'selected_words': final_words,
        'word_details': selection_results['final_word_list'],
        'selection_results': selection_results,
        'validation_results': validation_results,
        'summary': {
            'total_words': len(final_words),
            'sufficient_data_ratio': validation_results['coverage_analysis']['coverage_ratio'],
            'recommendation': validation_results['coverage_analysis']['recommendation']
        }
    }

import pickle
# Example usage
if __name__ == "__main__":
    # This would use your existing data
    from load_data_vocab import split_news_by_binary_label
    train_dict = pickle.load(open('data/news_articles/' + 'politifact' + '_train.pkl', 'rb'))
    news_o, news_1 = split_news_by_binary_label(train_dict)  # or 'gossipcop'
    # Example call:
    results = run_comprehensive_word_selection(news_o, news_1, target_size=200)
    final_word_list = results['selected_words']
    
    print("Word selection framework ready. Use run_comprehensive_word_selection() with your data.")