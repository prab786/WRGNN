import pickle
import numpy as np
import pandas as pd
import torch
from collections import defaultdict, Counter
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import entropy
import warnings
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer
import random
from typing import List, Dict, Tuple, Any
warnings.filterwarnings('ignore')

class StrongerValidation:
    """
    Comprehensive validation suite to test if Context-Specific Association Divergence
    represents genuine linguistic differences vs dataset biases.
    """
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained('bert-base-uncased')
        self.results = {}
        
    def temporal_matching_validation(self, real_texts: List[str], fake_texts: List[str], 
                                   real_dates: List[str] = None, fake_dates: List[str] = None,
                                   test_words: List[str] = None) -> Dict:
        """
        Test if divergences hold when controlling for temporal effects.
        
        Args:
            real_texts: Real news articles
            fake_texts: Fake news articles  
            real_dates: Publication dates for real articles (optional)
            fake_dates: Publication dates for fake articles (optional)
            test_words: Words to test for divergence
        
        Returns:
            Validation results with temporal controls
        """
        print("🕒 Running Temporal Matching Validation...")
        print("="*60)
        
        # If no dates provided, try to extract from text or simulate
        if real_dates is None or fake_dates is None:
            print("⚠️  No dates provided. Simulating temporal matching...")
            return self._simulate_temporal_matching(real_texts, fake_texts, test_words)
        
        # Parse dates and create time periods
        real_periods = self._extract_time_periods(real_dates)
        fake_periods = self._extract_time_periods(fake_dates)
        
        # Find overlapping time periods
        common_periods = set(real_periods) & set(fake_periods)
        
        if len(common_periods) < 3:
            print(f"⚠️  Only {len(common_periods)} common time periods found. Using simulation.")
            return self._simulate_temporal_matching(real_texts, fake_texts, test_words)
        
        print(f"Found {len(common_periods)} common time periods")
        
        # Test divergence within each time period
        period_results = []
        
        for period in common_periods:
            print(f"Testing period: {period}")
            
            # Get articles from this period
            real_period_texts = [text for text, p in zip(real_texts, real_periods) if p == period]
            fake_period_texts = [text for text, p in zip(fake_texts, fake_periods) if p == period]
            
            if len(real_period_texts) < 10 or len(fake_period_texts) < 10:
                continue
                
            # Test divergence for this period
            period_divergences = self._test_word_divergences(
                real_period_texts, fake_period_texts, test_words or self._get_default_test_words()
            )
            
            period_results.append({
                'period': period,
                'real_count': len(real_period_texts),
                'fake_count': len(fake_period_texts),
                'divergences': period_divergences
            })
        
        # Analyze consistency across periods
        consistency_analysis = self._analyze_temporal_consistency(period_results)
        
        return {
            'validation_type': 'temporal_matching',
            'period_results': period_results,
            'consistency_analysis': consistency_analysis,
            'conclusion': self._conclude_temporal_validation(consistency_analysis)
        }
    
    def topic_matching_validation(self, real_texts: List[str], fake_texts: List[str],
                                test_words: List[str] = None, n_topics: int = 10) -> Dict:
        """
        Test if divergences hold when controlling for topical differences.
        """
        print("🎯 Running Topic Matching Validation...")
        print("="*60)
        
        # Extract topics using TF-IDF and clustering
        print("Extracting topics...")
        all_texts = real_texts + fake_texts
        labels = [0] * len(real_texts) + [1] * len(fake_texts)
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', max_df=0.95, min_df=2)
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Clustering to identify topics
        kmeans = KMeans(n_clusters=n_topics, random_state=42)
        topic_assignments = kmeans.fit_predict(tfidf_matrix)
        
        # Analyze topic distribution
        topic_analysis = []
        for topic_id in range(n_topics):
            topic_indices = np.where(topic_assignments == topic_id)[0]
            topic_real_count = sum(1 for idx in topic_indices if labels[idx] == 0)
            topic_fake_count = sum(1 for idx in topic_indices if labels[idx] == 1)
            
            if topic_real_count >= 20 and topic_fake_count >= 20:  # Sufficient data
                topic_real_texts = [all_texts[idx] for idx in topic_indices if labels[idx] == 0]
                topic_fake_texts = [all_texts[idx] for idx in topic_indices if labels[idx] == 1]
                
                # Test divergence within this topic
                topic_divergences = self._test_word_divergences(
                    topic_real_texts, topic_fake_texts, test_words or self._get_default_test_words()
                )
                
                topic_analysis.append({
                    'topic_id': topic_id,
                    'real_count': topic_real_count,
                    'fake_count': topic_fake_count,
                    'divergences': topic_divergences,
                    'topic_keywords': self._get_topic_keywords(vectorizer, kmeans, topic_id)
                })
        
        print(f"Found {len(topic_analysis)} topics with sufficient data")
        
        # Analyze consistency across topics
        consistency_analysis = self._analyze_topic_consistency(topic_analysis)
        
        return {
            'validation_type': 'topic_matching',
            'topic_analysis': topic_analysis,
            'consistency_analysis': consistency_analysis,
            'conclusion': self._conclude_topic_validation(consistency_analysis)
        }
    
    def source_diversity_validation(self, real_texts: List[str], fake_texts: List[str],
                                  real_sources: List[str] = None, fake_sources: List[str] = None,
                                  test_words: List[str] = None) -> Dict:
        """
        Test if divergences hold across diverse source types.
        """
        print("📰 Running Source Diversity Validation...")
        print("="*60)
        
        if real_sources is None or fake_sources is None:
            print("⚠️  No source information provided. Simulating source diversity...")
            return self._simulate_source_diversity(real_texts, fake_texts, test_words)
        
        # Group by source
        real_by_source = defaultdict(list)
        fake_by_source = defaultdict(list)
        
        for text, source in zip(real_texts, real_sources):
            real_by_source[source].append(text)
            
        for text, source in zip(fake_texts, fake_sources):
            fake_by_source[source].append(text)
        
        # Test divergences for each source type
        source_results = []
        
        for source in set(real_sources) & set(fake_sources):
            if len(real_by_source[source]) >= 20 and len(fake_by_source[source]) >= 20:
                source_divergences = self._test_word_divergences(
                    real_by_source[source], fake_by_source[source], 
                    test_words or self._get_default_test_words()
                )
                
                source_results.append({
                    'source': source,
                    'real_count': len(real_by_source[source]),
                    'fake_count': len(fake_by_source[source]),
                    'divergences': source_divergences
                })
        
        # Analyze consistency across sources
        consistency_analysis = self._analyze_source_consistency(source_results)
        
        return {
            'validation_type': 'source_diversity',
            'source_results': source_results,
            'consistency_analysis': consistency_analysis,
            'conclusion': self._conclude_source_validation(consistency_analysis)
        }
    
    def cross_dataset_validation(self, datasets: Dict[str, Dict]) -> Dict:
        """
        Test if divergences replicate across different datasets.
        
        Args:
            datasets: Dict with keys as dataset names and values containing 'real' and 'fake' text lists
        """
        print("🔄 Running Cross-Dataset Validation...")
        print("="*60)
        
        dataset_results = {}
        test_words = self._get_default_test_words()
        
        for dataset_name, data in datasets.items():
            print(f"Testing dataset: {dataset_name}")
            
            real_texts = data.get('real', [])
            fake_texts = data.get('fake', [])
            
            if len(real_texts) < 50 or len(fake_texts) < 50:
                print(f"⚠️  Insufficient data for {dataset_name}")
                continue
            
            # Sample for computational efficiency
            real_sample = random.sample(real_texts, min(200, len(real_texts)))
            fake_sample = random.sample(fake_texts, min(200, len(fake_texts)))
            
            dataset_divergences = self._test_word_divergences(real_sample, fake_sample, test_words)
            
            dataset_results[dataset_name] = {
                'real_count': len(real_sample),
                'fake_count': len(fake_sample),
                'divergences': dataset_divergences
            }
        
        # Analyze cross-dataset consistency
        consistency_analysis = self._analyze_cross_dataset_consistency(dataset_results, test_words)
        
        return {
            'validation_type': 'cross_dataset',
            'dataset_results': dataset_results,
            'consistency_analysis': consistency_analysis,
            'conclusion': self._conclude_cross_dataset_validation(consistency_analysis)
        }
    
    def synthetic_validation(self, real_texts: List[str], fake_texts: List[str],
                           test_words: List[str] = None) -> Dict:
        """
        Test using synthetic fake news created by systematically altering real news.
        """
        print("🧪 Running Synthetic Validation...")
        print("="*60)
        
        # Create synthetic fake news by applying systematic transformations
        synthetic_fake_texts = []
        transformation_methods = [
            self._add_emotional_words,
            self._add_conspiracy_terms, 
            self._add_sensational_phrases,
            self._modify_attribution_words
        ]
        
        print("Creating synthetic fake news...")
        for i, real_text in enumerate(real_texts[:200]):  # Limit for efficiency
            if i % 4 == 0:  # Apply different transformations
                method = transformation_methods[i % len(transformation_methods)]
                synthetic_text = method(real_text)
                synthetic_fake_texts.append(synthetic_text)
        
        print(f"Created {len(synthetic_fake_texts)} synthetic fake articles")
        
        # Test divergence between real and synthetic fake
        test_words = test_words or self._get_default_test_words()
        synthetic_divergences = self._test_word_divergences(
            real_texts[:len(synthetic_fake_texts)], synthetic_fake_texts, test_words
        )
        
        # Compare with original real vs fake divergences  
        original_divergences = self._test_word_divergences(
            real_texts[:200], fake_texts[:200], test_words
        )
        
        # Analyze patterns
        comparison_analysis = self._compare_synthetic_vs_original(
            synthetic_divergences, original_divergences, test_words
        )
        
        return {
            'validation_type': 'synthetic',
            'synthetic_divergences': synthetic_divergences,
            'original_divergences': original_divergences,
            'comparison_analysis': comparison_analysis,
            'conclusion': self._conclude_synthetic_validation(comparison_analysis)
        }
    
    def same_event_validation(self, real_texts: List[str], fake_texts: List[str],
                            test_words: List[str] = None) -> Dict:
        """
        Test divergences when real and fake news cover the same events.
        """
        print("📅 Running Same-Event Validation...")
        print("="*60)
        
        # Find articles covering the same events using keyword similarity
        event_pairs = self._find_same_event_pairs(real_texts, fake_texts)
        
        if len(event_pairs) < 10:
            print("⚠️  Insufficient same-event pairs found. Using high-similarity pairs.")
            event_pairs = self._find_high_similarity_pairs(real_texts, fake_texts)
        
        print(f"Found {len(event_pairs)} same-event article pairs")
        
        # Extract real and fake texts for same events
        same_event_real = [pair['real_text'] for pair in event_pairs]
        same_event_fake = [pair['fake_text'] for pair in event_pairs]
        
        # Test divergences for same events
        test_words = test_words or self._get_default_test_words()
        same_event_divergences = self._test_word_divergences(
            same_event_real, same_event_fake, test_words
        )
        
        # Compare with random pairs
        random_real = random.sample(real_texts, len(same_event_real))
        random_fake = random.sample(fake_texts, len(same_event_fake))
        random_divergences = self._test_word_divergences(random_real, random_fake, test_words)
        
        # Analyze differences
        comparison_analysis = self._compare_same_event_vs_random(
            same_event_divergences, random_divergences, test_words
        )
        
        return {
            'validation_type': 'same_event',
            'event_pairs_count': len(event_pairs),
            'same_event_divergences': same_event_divergences,
            'random_divergences': random_divergences,
            'comparison_analysis': comparison_analysis,
            'conclusion': self._conclude_same_event_validation(comparison_analysis)
        }
    
    def style_transfer_validation(self, real_texts: List[str], fake_texts: List[str],
                                test_words: List[str] = None) -> Dict:
        """
        Test by creating style-transferred versions that preserve content but change style.
        """
        print("🎭 Running Style Transfer Validation...")
        print("="*60)
        
        # Create style-transferred versions
        # Transfer real news to fake style and vice versa
        real_to_fake_style = []
        fake_to_real_style = []
        
        print("Applying style transfers...")
        
        # Sample for efficiency
        real_sample = real_texts[:100]
        fake_sample = fake_texts[:100]
        
        for real_text in real_sample:
            transferred = self._transfer_to_fake_style(real_text)
            real_to_fake_style.append(transferred)
        
        for fake_text in fake_sample:
            transferred = self._transfer_to_real_style(fake_text)
            fake_to_real_style.append(transferred)
        
        print(f"Created {len(real_to_fake_style)} real→fake and {len(fake_to_real_style)} fake→real transfers")
        
        # Test divergences
        test_words = test_words or self._get_default_test_words()
        
        # Original divergences
        original_divergences = self._test_word_divergences(real_sample, fake_sample, test_words)
        
        # Style-transferred divergences
        transferred_divergences = self._test_word_divergences(
            fake_to_real_style, real_to_fake_style, test_words
        )
        
        # Cross-style divergences
        cross_style_divergences = self._test_word_divergences(
            real_sample, real_to_fake_style, test_words
        )
        
        # Analyze style effects
        style_analysis = self._analyze_style_effects(
            original_divergences, transferred_divergences, cross_style_divergences, test_words
        )
        
        return {
            'validation_type': 'style_transfer',
            'original_divergences': original_divergences,
            'transferred_divergences': transferred_divergences,
            'cross_style_divergences': cross_style_divergences,
            'style_analysis': style_analysis,
            'conclusion': self._conclude_style_validation(style_analysis)
        }
    
    def comprehensive_validation_suite(self, real_texts: List[str], fake_texts: List[str],
                                     additional_datasets: Dict = None,
                                     test_words: List[str] = None) -> Dict:
        """
        Run the complete validation suite and provide overall conclusions.
        """
        print("🔬 Running Comprehensive Validation Suite...")
        print("="*80)
        
        all_results = {}
        
        # 1. Topic Matching Validation
        all_results['topic_matching'] = self.topic_matching_validation(real_texts, fake_texts, test_words)
        
        # 2. Synthetic Validation  
        all_results['synthetic'] = self.synthetic_validation(real_texts, fake_texts, test_words)
        
        # 3. Same Event Validation
        all_results['same_event'] = self.same_event_validation(real_texts, fake_texts, test_words)
        
        # 4. Style Transfer Validation
        all_results['style_transfer'] = self.style_transfer_validation(real_texts, fake_texts, test_words)
        
        # 5. Cross-Dataset Validation (if additional datasets provided)
        if additional_datasets:
            all_results['cross_dataset'] = self.cross_dataset_validation(additional_datasets)
        
        # Overall analysis
        overall_conclusion = self._synthesize_validation_results(all_results)
        
        # Generate comprehensive report
        self._generate_validation_report(all_results, overall_conclusion)
        
        return {
            'individual_validations': all_results,
            'overall_conclusion': overall_conclusion,
            'validity_score': self._calculate_validity_score(all_results),
            'recommendations': self._generate_recommendations(overall_conclusion)
        }
    
    # Helper methods
    def _get_default_test_words(self) -> List[str]:
        """Default set of words to test across validations."""
        return [
            'officials', 'sources', 'reported', 'confirmed', 'according',
            'statement', 'investigation', 'administration', 'president',
            'government', 'authorities', 'experts', 'researchers', 'study'
        ]
    
    def _test_word_divergences(self, real_texts: List[str], fake_texts: List[str], 
                              test_words: List[str]) -> Dict:
        """Test KL divergences for given words between real and fake texts."""
        from load_data_vocab import compute_word_cooccurrence_distribution, compute_kl_divergence
        
        divergences = {}
        
        for word in test_words:
            try:
                P_real = compute_word_cooccurrence_distribution(word, real_texts, self.tokenizer)
                P_fake = compute_word_cooccurrence_distribution(word, fake_texts, self.tokenizer)
                
                if P_real and P_fake:
                    kl_div = compute_kl_divergence(P_real, P_fake)
                    divergences[word] = {
                        'kl_divergence': kl_div,
                        'real_vocab_size': len(P_real),
                        'fake_vocab_size': len(P_fake),
                        'P_real': P_real,
                        'P_fake': P_fake
                    }
            except Exception as e:
                continue
        
        return divergences
    
    def _simulate_temporal_matching(self, real_texts: List[str], fake_texts: List[str], 
                                  test_words: List[str]) -> Dict:
        """Simulate temporal matching by random time-based splits."""
        print("Simulating temporal periods...")
        
        # Create artificial time periods
        n_periods = 4
        period_results = []
        
        for period in range(n_periods):
            # Create temporal splits
            real_period = real_texts[period::n_periods]
            fake_period = fake_texts[period::n_periods]
            
            if len(real_period) >= 20 and len(fake_period) >= 20:
                period_divergences = self._test_word_divergences(
                    real_period, fake_period, test_words or self._get_default_test_words()
                )
                
                period_results.append({
                    'period': f'simulated_period_{period}',
                    'real_count': len(real_period),
                    'fake_count': len(fake_period),
                    'divergences': period_divergences
                })
        
        consistency_analysis = self._analyze_temporal_consistency(period_results)
        
        return {
            'validation_type': 'temporal_matching_simulated',
            'period_results': period_results,
            'consistency_analysis': consistency_analysis,
            'conclusion': self._conclude_temporal_validation(consistency_analysis)
        }
    
    def _extract_time_periods(self, dates: List[str]) -> List[str]:
        """Extract time periods (e.g., year-month) from dates."""
        periods = []
        for date_str in dates:
            try:
                # Try to parse common date formats
                if isinstance(date_str, str):
                    # Extract year-month pattern
                    match = re.search(r'(\d{4})-?(\d{2})', date_str)
                    if match:
                        periods.append(f"{match.group(1)}-{match.group(2)}")
                    else:
                        periods.append("unknown")
                else:
                    periods.append("unknown")
            except:
                periods.append("unknown")
        return periods
    
    def _find_same_event_pairs(self, real_texts: List[str], fake_texts: List[str]) -> List[Dict]:
        """Find real-fake article pairs covering the same events."""
        event_pairs = []
        
        # Use TF-IDF to find similar articles
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        
        # Sample for efficiency
        real_sample = real_texts[:200] if len(real_texts) > 200 else real_texts
        fake_sample = fake_texts[:200] if len(fake_texts) > 200 else fake_texts
        
        all_texts = real_sample + fake_sample
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Compute similarities between real and fake articles
        real_matrix = tfidf_matrix[:len(real_sample)]
        fake_matrix = tfidf_matrix[len(real_sample):]
        
        similarities = cosine_similarity(real_matrix, fake_matrix)
        
        # Find high-similarity pairs (likely same events)
        threshold = 0.3  # Adjust based on data
        
        for i in range(len(real_sample)):
            for j in range(len(fake_sample)):
                if similarities[i, j] > threshold:
                    event_pairs.append({
                        'real_text': real_sample[i],
                        'fake_text': fake_sample[j],
                        'similarity': similarities[i, j]
                    })
                    break  # One pair per real article
        
        return event_pairs
    
    def _find_high_similarity_pairs(self, real_texts: List[str], fake_texts: List[str]) -> List[Dict]:
        """Find high-similarity pairs as proxy for same events."""
        return self._find_same_event_pairs(real_texts, fake_texts)  # Same implementation
    
    def _add_emotional_words(self, text: str) -> str:
        """Add emotional/sensational words to create synthetic fake news."""
        emotional_words = ['shocking', 'unbelievable', 'outrageous', 'devastating', 'explosive']
        
        # Insert emotional words at strategic positions
        sentences = text.split('.')
        if len(sentences) > 1:
            # Add to first sentence
            first_sentence = sentences[0]
            word_to_add = random.choice(emotional_words)
            modified_first = f"{first_sentence.strip()}, which is {word_to_add},"
            sentences[0] = modified_first
            return '. '.join(sentences)
        return text
    
    def _add_conspiracy_terms(self, text: str) -> str:
        """Add conspiracy-related terms."""
        conspiracy_terms = ['they dont want you to know', 'hidden agenda', 'cover-up', 
                           'mainstream media wont tell you', 'the truth is']
        
        # Add conspiracy framing
        term = random.choice(conspiracy_terms)
        return f"{term.title()}: {text}"
    
    def _add_sensational_phrases(self, text: str) -> str:
        """Add sensational phrases."""
        sensational_phrases = ['BREAKING:', 'URGENT:', 'EXCLUSIVE:', 'BOMBSHELL:', 'LEAKED:']
        
        phrase = random.choice(sensational_phrases)
        return f"{phrase} {text}"
    
    def _modify_attribution_words(self, text: str) -> str:
        """Modify attribution words to be less credible."""
        # Replace credible sources with vague ones
        replacements = {
            'officials': 'sources',
            'confirmed': 'alleged',
            'reported': 'claimed',
            'according to': 'rumors suggest',
            'study shows': 'some say'
        }
        
        modified_text = text
        for original, replacement in replacements.items():
            modified_text = modified_text.replace(original, replacement)
        
        return modified_text
    
    def _transfer_to_fake_style(self, real_text: str) -> str:
        """Transfer real news to fake style."""
        # Apply multiple transformations
        text = self._add_emotional_words(real_text)
        text = self._modify_attribution_words(text)
        return text
    
    def _transfer_to_real_style(self, fake_text: str) -> str:
        """Transfer fake news to real style."""
        # Remove sensational elements and make more formal
        text = fake_text
        
        # Remove caps
        text = text.lower().capitalize()
        
        # Replace sensational words
        sensational_replacements = {
            'shocking': 'notable',
            'unbelievable': 'significant',
            'outrageous': 'concerning',
            'devastating': 'serious',
            'explosive': 'important'
        }
        
        for sensational, neutral in sensational_replacements.items():
            text = text.replace(sensational, neutral)
        
        return text
    
    def _get_topic_keywords(self, vectorizer, kmeans, topic_id: int) -> List[str]:
        """Get top keywords for a topic cluster."""
        feature_names = vectorizer.get_feature_names_out()
        center = kmeans.cluster_centers_[topic_id]
        top_indices = center.argsort()[-10:][::-1]
        return [feature_names[i] for i in top_indices]
    
    def _analyze_temporal_consistency(self, period_results: List[Dict]) -> Dict:
        """Analyze consistency of divergences across time periods."""
        if not period_results:
            return {'error': 'No period results to analyze'}
        
        # Collect divergences for each word across periods
        word_divergences = defaultdict(list)
        
        for period_result in period_results:
            for word, div_data in period_result['divergences'].items():
                word_divergences[word].append(div_data['kl_divergence'])
        
        # Calculate consistency metrics
        consistency_metrics = {}
        
        for word, divergences in word_divergences.items():
            if len(divergences) >= 2:
                consistency_metrics[word] = {
                    'mean_divergence': np.mean(divergences),
                    'std_divergence': np.std(divergences),
                    'coefficient_of_variation': np.std(divergences) / np.mean(divergences) if np.mean(divergences) > 0 else float('inf'),
                    'min_divergence': np.min(divergences),
                    'max_divergence': np.max(divergences),
                    'n_periods': len(divergences)
                }
        
        # Overall consistency score
        cv_values = [metrics['coefficient_of_variation'] for metrics in consistency_metrics.values() 
                    if metrics['coefficient_of_variation'] != float('inf')]
        
        overall_consistency = {
            'mean_coefficient_of_variation': np.mean(cv_values) if cv_values else float('inf'),
            'consistent_words': sum(1 for cv in cv_values if cv < 0.5),  # Low variation
            'total_words': len(cv_values),
            'consistency_ratio': sum(1 for cv in cv_values if cv < 0.5) / len(cv_values) if cv_values else 0
        }
        
        return {
            'word_consistency': consistency_metrics,
            'overall_consistency': overall_consistency
        }
    
    def _analyze_topic_consistency(self, topic_analysis: List[Dict]) -> Dict:
        """Analyze consistency across topics."""
        return self._analyze_temporal_consistency(topic_analysis)  # Same logic
    
    def _analyze_source_consistency(self, source_results: List[Dict]) -> Dict:
        """Analyze consistency across sources."""
        return self._analyze_temporal_consistency(source_results)  # Same logic
    
    def _analyze_cross_dataset_consistency(self, dataset_results: Dict, test_words: List[str]) -> Dict:
        """Analyze consistency across datasets."""
        # Convert format for consistency analysis
        period_results = []
        for dataset_name, results in dataset_results.items():
            period_results.append({
                'period': dataset_name,
                'divergences': results['divergences']
            })
        
        return self._analyze_temporal_consistency(period_results)
    
    def _compare_synthetic_vs_original(self, synthetic_divergences: Dict, 
                                     original_divergences: Dict, test_words: List[str]) -> Dict:
        """Compare synthetic vs original divergences to test linguistic hypothesis."""
        comparison = {}
        
        for word in test_words:
            if word in synthetic_divergences and word in original_divergences:
                synthetic_kl = synthetic_divergences[word]['kl_divergence']
                original_kl = original_divergences[word]['kl_divergence']
                
                comparison[word] = {
                    'synthetic_kl': synthetic_kl,
                    'original_kl': original_kl,
                    'ratio': synthetic_kl / original_kl if original_kl > 0 else float('inf'),
                    'difference': abs(synthetic_kl - original_kl),
                    'both_significant': synthetic_kl > 0.1 and original_kl > 0.1
                }
        
        # Overall analysis
        ratios = [comp['ratio'] for comp in comparison.values() if comp['ratio'] != float('inf')]
        
        overall = {
            'mean_ratio': np.mean(ratios) if ratios else 0,
            'correlation': np.corrcoef([comp['synthetic_kl'] for comp in comparison.values()],
                                    [comp['original_kl'] for comp in comparison.values()])[0,1] if len(comparison) > 1 else 0,
            'both_significant_count': sum(1 for comp in comparison.values() if comp['both_significant']),
            'total_words': len(comparison)
        }
        
        return {
            'word_comparisons': comparison,
            'overall_analysis': overall
        }
    
    def _compare_same_event_vs_random(self, same_event_divergences: Dict, 
                                    random_divergences: Dict, test_words: List[str]) -> Dict:
        """Compare same-event vs random article divergences."""
        comparison = {}
        
        for word in test_words:
            if word in same_event_divergences and word in random_divergences:
                same_event_kl = same_event_divergences[word]['kl_divergence']
                random_kl = random_divergences[word]['kl_divergence']
                
                comparison[word] = {
                    'same_event_kl': same_event_kl,
                    'random_kl': random_kl,
                    'ratio': same_event_kl / random_kl if random_kl > 0 else float('inf'),
                    'difference': same_event_kl - random_kl,
                    'same_event_higher': same_event_kl > random_kl
                }
        
        # Statistical test
        same_event_values = [comp['same_event_kl'] for comp in comparison.values()]
        random_values = [comp['random_kl'] for comp in comparison.values()]
        
        if len(same_event_values) > 1:
            stat, p_value = stats.wilcoxon(same_event_values, random_values, alternative='greater')
        else:
            stat, p_value = None, 1.0
        
        overall = {
            'mean_same_event_kl': np.mean(same_event_values) if same_event_values else 0,
            'mean_random_kl': np.mean(random_values) if random_values else 0,
            'same_event_higher_count': sum(1 for comp in comparison.values() if comp['same_event_higher']),
            'statistical_test': {'statistic': stat, 'p_value': p_value},
            'conclusion': 'same_event_higher' if p_value and p_value < 0.05 else 'no_difference'
        }
        
        return {
            'word_comparisons': comparison,
            'overall_analysis': overall
        }
    
    def _analyze_style_effects(self, original_divergences: Dict, transferred_divergences: Dict,
                             cross_style_divergences: Dict, test_words: List[str]) -> Dict:
        """Analyze the effects of style transfer on divergences."""
        analysis = {}
        
        for word in test_words:
            if (word in original_divergences and word in transferred_divergences and 
                word in cross_style_divergences):
                
                orig_kl = original_divergences[word]['kl_divergence']
                trans_kl = transferred_divergences[word]['kl_divergence']
                cross_kl = cross_style_divergences[word]['kl_divergence']
                
                analysis[word] = {
                    'original_kl': orig_kl,
                    'transferred_kl': trans_kl,
                    'cross_style_kl': cross_kl,
                    'style_preserved': abs(trans_kl - orig_kl) < 0.1 * orig_kl,
                    'content_vs_style_ratio': cross_kl / orig_kl if orig_kl > 0 else 0
                }
        
        # Overall style analysis
        style_preserved_count = sum(1 for comp in analysis.values() if comp['style_preserved'])
        content_ratios = [comp['content_vs_style_ratio'] for comp in analysis.values()]
        
        overall = {
            'style_preserved_ratio': style_preserved_count / len(analysis) if analysis else 0,
            'mean_content_style_ratio': np.mean(content_ratios) if content_ratios else 0,
            'style_dominates': np.mean(content_ratios) > 0.7 if content_ratios else False
        }
        
        return {
            'word_analysis': analysis,
            'overall_analysis': overall
        }
    
    def _conclude_temporal_validation(self, consistency_analysis: Dict) -> Dict:
        """Draw conclusions from temporal validation."""
        if 'error' in consistency_analysis:
            return {'conclusion': 'insufficient_data', 'confidence': 'low'}
        
        consistency_ratio = consistency_analysis['overall_consistency']['consistency_ratio']
        
        if consistency_ratio > 0.7:
            return {
                'conclusion': 'patterns_robust_across_time',
                'confidence': 'high',
                'interpretation': 'Divergences are consistent across time periods, supporting linguistic hypothesis'
            }
        elif consistency_ratio > 0.4:
            return {
                'conclusion': 'patterns_moderately_consistent',
                'confidence': 'medium',
                'interpretation': 'Some temporal variation exists but core patterns persist'
            }
        else:
            return {
                'conclusion': 'patterns_vary_by_time',
                'confidence': 'low',
                'interpretation': 'High temporal variation suggests potential bias rather than linguistic differences'
            }
    
    def _conclude_topic_validation(self, consistency_analysis: Dict) -> Dict:
        """Draw conclusions from topic validation."""
        consistency_ratio = consistency_analysis['overall_consistency']['consistency_ratio']
        
        if consistency_ratio > 0.6:
            return {
                'conclusion': 'patterns_robust_across_topics',
                'confidence': 'high',
                'interpretation': 'Divergences persist across different topics, supporting linguistic hypothesis'
            }
        else:
            return {
                'conclusion': 'patterns_topic_dependent',
                'confidence': 'low',
                'interpretation': 'Divergences vary by topic, suggesting content rather than linguistic differences'
            }
    
    def _conclude_source_validation(self, consistency_analysis: Dict) -> Dict:
        """Draw conclusions from source validation."""
        consistency_ratio = consistency_analysis['overall_consistency']['consistency_ratio']
        
        if consistency_ratio > 0.6:
            return {
                'conclusion': 'patterns_robust_across_sources',
                'confidence': 'high',
                'interpretation': 'Divergences consistent across sources, supporting linguistic hypothesis'
            }
        else:
            return {
                'conclusion': 'patterns_source_dependent',
                'confidence': 'low',
                'interpretation': 'Divergences vary by source, suggesting source bias rather than linguistic differences'
            }
    
    def _conclude_cross_dataset_validation(self, consistency_analysis: Dict) -> Dict:
        """Draw conclusions from cross-dataset validation."""
        consistency_ratio = consistency_analysis['overall_consistency']['consistency_ratio']
        
        if consistency_ratio > 0.5:
            return {
                'conclusion': 'patterns_replicate_across_datasets',
                'confidence': 'high',
                'interpretation': 'Divergences replicate across independent datasets, strong evidence for linguistic differences'
            }
        else:
            return {
                'conclusion': 'patterns_dataset_specific',
                'confidence': 'low',
                'interpretation': 'Poor replication suggests dataset-specific biases'
            }
    
    def _conclude_synthetic_validation(self, comparison_analysis: Dict) -> Dict:
        """Draw conclusions from synthetic validation."""
        correlation = comparison_analysis['overall_analysis']['correlation']
        both_significant = comparison_analysis['overall_analysis']['both_significant_count']
        total = comparison_analysis['overall_analysis']['total_words']
        
        if correlation > 0.6 and both_significant / total > 0.5:
            return {
                'conclusion': 'synthetic_patterns_match_real',
                'confidence': 'high',
                'interpretation': 'Synthetic manipulation produces similar patterns, supporting causal linguistic hypothesis'
            }
        else:
            return {
                'conclusion': 'synthetic_patterns_differ',
                'confidence': 'medium',
                'interpretation': 'Synthetic patterns differ from real data, suggesting complex factors beyond simple linguistic manipulation'
            }
    
    def _conclude_same_event_validation(self, comparison_analysis: Dict) -> Dict:
        """Draw conclusions from same-event validation."""
        conclusion = comparison_analysis['overall_analysis']['conclusion']
        p_value = comparison_analysis['overall_analysis']['statistical_test']['p_value']
        
        if conclusion == 'same_event_higher' and p_value and p_value < 0.05:
            return {
                'conclusion': 'patterns_persist_same_events',
                'confidence': 'high',
                'interpretation': 'Divergences persist even for same events, supporting linguistic differences over content differences'
            }
        else:
            return {
                'conclusion': 'patterns_content_dependent',
                'confidence': 'medium',
                'interpretation': 'Divergences reduce for same events, suggesting content rather than linguistic differences'
            }
    
    def _conclude_style_validation(self, style_analysis: Dict) -> Dict:
        """Draw conclusions from style transfer validation."""
        style_dominates = style_analysis['overall_analysis']['style_dominates']
        style_preserved_ratio = style_analysis['overall_analysis']['style_preserved_ratio']
        
        if style_dominates and style_preserved_ratio > 0.5:
            return {
                'conclusion': 'style_drives_divergences',
                'confidence': 'high',
                'interpretation': 'Style transfer preserves divergences, indicating style-based linguistic differences'
            }
        else:
            return {
                'conclusion': 'content_drives_divergences',
                'confidence': 'medium',
                'interpretation': 'Style transfer changes divergences, suggesting content-based rather than style-based differences'
            }
    
    def _synthesize_validation_results(self, all_results: Dict) -> Dict:
        """Synthesize all validation results into overall conclusion."""
        high_confidence_results = []
        medium_confidence_results = []
        low_confidence_results = []
        
        for validation_type, results in all_results.items():
            conclusion = results.get('conclusion', {})
            confidence = conclusion.get('confidence', 'low')
            
            if confidence == 'high':
                high_confidence_results.append((validation_type, conclusion))
            elif confidence == 'medium':
                medium_confidence_results.append((validation_type, conclusion))
            else:
                low_confidence_results.append((validation_type, conclusion))
        
        # Count supporting vs contradicting evidence
        linguistic_support = 0
        bias_support = 0
        
        for _, conclusion in high_confidence_results + medium_confidence_results:
            if any(keyword in conclusion.get('conclusion', '') for keyword in 
                  ['robust', 'persist', 'replicate', 'match']):
                linguistic_support += 2 if _ in [item[0] for item in high_confidence_results] else 1
            else:
                bias_support += 2 if _ in [item[0] for item in high_confidence_results] else 1
        
        # Overall conclusion
        if linguistic_support > bias_support * 1.5:
            overall_conclusion = {
                'conclusion': 'linguistic_differences_supported',
                'confidence': 'high' if len(high_confidence_results) >= 2 else 'medium',
                'evidence_ratio': linguistic_support / (linguistic_support + bias_support),
                'interpretation': 'Multiple validations support genuine linguistic differences between real and fake news'
            }
        elif bias_support > linguistic_support * 1.5:
            overall_conclusion = {
                'conclusion': 'dataset_bias_likely',
                'confidence': 'high' if len(high_confidence_results) >= 2 else 'medium',
                'evidence_ratio': bias_support / (linguistic_support + bias_support),
                'interpretation': 'Multiple validations suggest dataset biases rather than linguistic differences'
            }
        else:
            overall_conclusion = {
                'conclusion': 'mixed_evidence',
                'confidence': 'medium',
                'evidence_ratio': 0.5,
                'interpretation': 'Evidence is mixed, suggesting both linguistic differences and dataset biases contribute'
            }
        
        return {
            'overall_conclusion': overall_conclusion,
            'high_confidence_validations': len(high_confidence_results),
            'supporting_evidence': linguistic_support,
            'contradicting_evidence': bias_support,
            'detailed_results': {
                'high_confidence': high_confidence_results,
                'medium_confidence': medium_confidence_results,
                'low_confidence': low_confidence_results
            }
        }
    
    def _calculate_validity_score(self, all_results: Dict) -> float:
        """Calculate overall validity score (0-1) for the divergence hypothesis."""
        scores = []
        weights = {
            'topic_matching': 0.25,
            'synthetic': 0.20,
            'same_event': 0.25,
            'style_transfer': 0.15,
            'cross_dataset': 0.15
        }
        
        for validation_type, results in all_results.items():
            conclusion = results.get('conclusion', {})
            confidence = conclusion.get('confidence', 'low')
            conclusion_text = conclusion.get('conclusion', '')
            
            # Score based on support for linguistic hypothesis
            if any(keyword in conclusion_text for keyword in ['robust', 'persist', 'replicate', 'match']):
                base_score = 0.8
            elif 'mixed' in conclusion_text:
                base_score = 0.5
            else:
                base_score = 0.2
            
            # Adjust by confidence
            confidence_multiplier = {'high': 1.0, 'medium': 0.7, 'low': 0.4}.get(confidence, 0.4)
            final_score = base_score * confidence_multiplier
            
            weight = weights.get(validation_type, 0.1)
            scores.append(final_score * weight)
        
        return sum(scores) / sum(weights.values()) if scores else 0.0
    
    def _generate_recommendations(self, overall_conclusion: Dict) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        conclusion = overall_conclusion['overall_conclusion']['conclusion']
        confidence = overall_conclusion['overall_conclusion']['confidence']
        
        if conclusion == 'linguistic_differences_supported':
            if confidence == 'high':
                recommendations.extend([
                    "✅ Strong evidence supports genuine linguistic differences",
                    "✅ Safe to proceed with DualContextGCN approach",
                    "✅ Consider extending to other domains and languages",
                    "✅ Publish findings with confidence in theoretical foundation"
                ])
            else:
                recommendations.extend([
                    "⚠️ Moderate evidence for linguistic differences",
                    "⚠️ Conduct additional validation with larger datasets",
                    "⚠️ Consider hybrid approach addressing both linguistic and content factors"
                ])
        
        elif conclusion == 'dataset_bias_likely':
            recommendations.extend([
                "❌ Evidence suggests dataset biases rather than linguistic differences",
                "❌ Reconsider theoretical assumptions",
                "❌ Focus on dataset construction and bias mitigation",
                "❌ Consider alternative approaches less dependent on word associations"
            ])
        
        else:  # mixed evidence
            recommendations.extend([
                "⚠️ Mixed evidence suggests both linguistic differences and biases",
                "⚠️ Implement hybrid model accounting for both factors",
                "⚠️ Enhance dataset diversity and quality",
                "⚠️ Conduct longitudinal studies to separate factors"
            ])
        
        # Technical recommendations
        if overall_conclusion.get('supporting_evidence', 0) > 0:
            recommendations.append("🔧 Implement bias-aware training procedures")
            recommendations.append("🔧 Add dataset source indicators as features")
        
        return recommendations
    
    def _generate_validation_report(self, all_results: Dict, overall_conclusion: Dict):
        """Generate comprehensive validation report."""
        print("\n" + "="*80)
        print("🔬 COMPREHENSIVE VALIDATION REPORT")
        print("="*80)
        
        print(f"\n📊 OVERALL CONCLUSION:")
        print(f"Conclusion: {overall_conclusion['overall_conclusion']['conclusion']}")
        print(f"Confidence: {overall_conclusion['overall_conclusion']['confidence']}")
        print(f"Validity Score: {self._calculate_validity_score(all_results):.3f}")
        print(f"Interpretation: {overall_conclusion['overall_conclusion']['interpretation']}")
        
        print(f"\n📈 EVIDENCE SUMMARY:")
        print(f"Supporting Evidence: {overall_conclusion.get('supporting_evidence', 0)}")
        print(f"Contradicting Evidence: {overall_conclusion.get('contradicting_evidence', 0)}")
        print(f"High Confidence Validations: {overall_conclusion.get('high_confidence_validations', 0)}")
        
        print(f"\n🔍 INDIVIDUAL VALIDATION RESULTS:")
        for validation_type, results in all_results.items():
            conclusion = results.get('conclusion', {})
            confidence = conclusion.get('confidence', 'unknown')
            interpretation = conclusion.get('interpretation', 'No interpretation available')
            
            status_emoji = {
                'high': '✅',
                'medium': '⚠️',
                'low': '❌',
                'unknown': '❓'
            }.get(confidence, '❓')
            
            print(f"\n{status_emoji} {validation_type.upper().replace('_', ' ')}")
            print(f"   Confidence: {confidence}")
            print(f"   Result: {interpretation}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        recommendations = self._generate_recommendations(overall_conclusion)
        for rec in recommendations:
            print(f"   {rec}")
        
        print(f"\n" + "="*80)


# Example usage and testing functions
def run_comprehensive_validation_example():
    """
    Example of how to run the comprehensive validation suite.
    """
    # Initialize validation suite
    validator = StrongerValidation()
    
    # Load your existing data
    print("Loading data for validation...")
    
    # Use the existing global variables from your code
    global news_o, news_1, x_test, y_test
    
    # Sample data for computational efficiency
    real_sample = news_o[:500] if len(news_o) > 500 else news_o
    fake_sample = news_1[:500] if len(news_1) > 500 else news_1
    
    print(f"Using {len(real_sample)} real and {len(fake_sample)} fake articles")
    
    # Define test words (can be customized)
    test_words = [
        'officials', 'sources', 'reported', 'confirmed', 'according',
        'president', 'administration', 'government', 'statement',
        'investigation', 'authorities', 'experts', 'study'
    ]
    
    # Run comprehensive validation
    results = validator.comprehensive_validation_suite(
        real_texts=real_sample,
        fake_texts=fake_sample,
        test_words=test_words
    )
    
    return results, validator

def create_additional_datasets_for_validation():
    """
    Create additional datasets for cross-dataset validation.
    This is a placeholder - replace with actual additional datasets if available.
    """
    # This would be replaced with actual additional datasets
    # For now, we'll simulate by creating splits of existing data
    
    global news_o, news_1
    
    # Create artificial "datasets" by splitting existing data
    split_point_real = len(news_o) // 2
    split_point_fake = len(news_1) // 2
    
    additional_datasets = {
        'dataset_A': {
            'real': news_o[:split_point_real],
            'fake': news_1[:split_point_fake]
        },
        'dataset_B': {
            'real': news_o[split_point_real:],
            'fake': news_1[split_point_fake:]
        }
    }
    
    return additional_datasets

def test_specific_validation(validation_type: str = 'topic_matching'):
    """
    Test a specific validation method in detail.
    """
    validator = StrongerValidation()
    
    # Load data
    global news_o, news_1
    real_sample = news_o[:300] if len(news_o) > 300 else news_o
    fake_sample = news_1[:300] if len(news_1) > 300 else news_1
    
    test_words = ['officials', 'sources', 'reported', 'confirmed', 'president']
    
    if validation_type == 'topic_matching':
        result = validator.topic_matching_validation(real_sample, fake_sample, test_words)
    elif validation_type == 'synthetic':
        result = validator.synthetic_validation(real_sample, fake_sample, test_words)
    elif validation_type == 'same_event':
        result = validator.same_event_validation(real_sample, fake_sample, test_words)
    elif validation_type == 'style_transfer':
        result = validator.style_transfer_validation(real_sample, fake_sample, test_words)
    else:
        print(f"Unknown validation type: {validation_type}")
        return None
    
    return result

def visualize_validation_results(results: Dict):
    """
    Create visualizations for validation results.
    """
    if not results or 'individual_validations' not in results:
        print("No results to visualize")
        return
    
    individual_results = results['individual_validations']
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Validation Results Summary', fontsize=16, fontweight='bold')
    
    # 1. Confidence levels across validations
    ax1 = axes[0, 0]
    validation_names = []
    confidence_scores = []
    confidence_colors = []
    
    for val_type, val_results in individual_results.items():
        validation_names.append(val_type.replace('_', ' ').title())
        conclusion = val_results.get('conclusion', {})
        confidence = conclusion.get('confidence', 'low')
        
        confidence_score = {'high': 3, 'medium': 2, 'low': 1}.get(confidence, 1)
        confidence_scores.append(confidence_score)
        
        color = {'high': 'green', 'medium': 'orange', 'low': 'red'}.get(confidence, 'gray')
        confidence_colors.append(color)
    
    bars1 = ax1.bar(validation_names, confidence_scores, color=confidence_colors, alpha=0.7)
    ax1.set_ylabel('Confidence Level')
    ax1.set_title('Confidence Levels by Validation Type')
    ax1.set_ylim(0, 3.5)
    ax1.set_yticks([1, 2, 3])
    ax1.set_yticklabels(['Low', 'Medium', 'High'])
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 2. Overall validity score
    ax2 = axes[0, 1]
    validity_score = results.get('validity_score', 0)
    ax2.pie([validity_score, 1-validity_score], 
           labels=['Supporting Evidence', 'Contradicting Evidence'],
           colors=['lightgreen', 'lightcoral'],
           autopct='%1.1f%%',
           startangle=90)
    ax2.set_title(f'Overall Validity Score: {validity_score:.3f}')
    
    # 3. Evidence comparison
    ax3 = axes[1, 0]
    overall_conclusion = results.get('overall_conclusion', {})
    supporting = overall_conclusion.get('supporting_evidence', 0)
    contradicting = overall_conclusion.get('contradicting_evidence', 0)
    
    ax3.bar(['Supporting\nLinguistic\nDifferences', 'Supporting\nDataset\nBias'], 
           [supporting, contradicting],
           color=['lightgreen', 'lightcoral'])
    ax3.set_ylabel('Evidence Weight')
    ax3.set_title('Evidence Comparison')
    
    # 4. Recommendations summary
    ax4 = axes[1, 1]
    recommendations = results.get('recommendations', [])
    
    # Count recommendation types
    positive_recs = sum(1 for rec in recommendations if '✅' in rec)
    warning_recs = sum(1 for rec in recommendations if '⚠️' in rec)
    negative_recs = sum(1 for rec in recommendations if '❌' in rec)
    technical_recs = sum(1 for rec in recommendations if '🔧' in rec)
    
    rec_counts = [positive_recs, warning_recs, negative_recs, technical_recs]
    rec_labels = ['Positive', 'Warning', 'Negative', 'Technical']
    rec_colors = ['green', 'orange', 'red', 'blue']
    
    ax4.bar(rec_labels, rec_counts, color=rec_colors, alpha=0.7)
    ax4.set_ylabel('Number of Recommendations')
    ax4.set_title('Recommendation Types')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed summary
    print(f"\n{'='*60}")
    print(f"VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Overall Conclusion: {overall_conclusion.get('overall_conclusion', {}).get('conclusion', 'Unknown')}")
    print(f"Validity Score: {validity_score:.3f}")
    print(f"Supporting Evidence: {supporting}")
    print(f"Contradicting Evidence: {contradicting}")
    print(f"High Confidence Validations: {overall_conclusion.get('high_confidence_validations', 0)}")

# Main execution
if __name__ == "__main__":
    print("🔬 Starting Stronger Validation Suite...")
    
    # Run comprehensive validation
    try:
        results, validator = run_comprehensive_validation_example()
        
        # Visualize results
        visualize_validation_results(results)
        
        # Save results
        with open('validation_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print("📁 Results saved to 'validation_results.pkl'")
        
    except Exception as e:
        print(f"❌ Error running validation: {e}")
        print("🔧 Running individual validation test instead...")
        
        # Test individual validation
        result = test_specific_validation('topic_matching')
        if result:
            print("✅ Individual validation completed successfully")
        else:
            print("❌ Individual validation failed")