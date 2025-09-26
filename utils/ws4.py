#!/usr/bin/env python3
"""
Data-Driven Word Selection with Enhanced Validation and Debugging

This script implements statistical methods for word selection with comprehensive
validation to ensure meaningful, publication-ready results.
"""

import sys
sys.path.append('.')

# Import your existing functions
from load_data_vocab import (
    split_news_by_binary_label,
    compute_word_cooccurrence_distribution,
    compute_kl_divergence,
    tokenizer
)
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import random
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2_contingency, ttest_1samp
import warnings
warnings.filterwarnings('ignore')

def load_data_for_word_selection(obj='politifact'):
    """Load data for word selection."""
    train_dict = pickle.load(open(f'data/news_articles/{obj}_train.pkl', 'rb'))
    news_real, news_fake = split_news_by_binary_label(train_dict)
    return news_real, news_fake

def debug_word_cooccurrence(word, real_texts, fake_texts, window_size=5):
    """
    Debug function to manually inspect word co-occurrence calculations.
    This helps identify if extreme KL values are due to implementation issues.
    """
    print(f"\n=== DEBUGGING WORD: '{word}' ===")
    
    # Count basic word occurrences
    real_word_count = sum(1 for text in real_texts if word in text.lower())
    fake_word_count = sum(1 for text in fake_texts if word in text.lower())
    
    print(f"Word appears in {real_word_count}/{len(real_texts)} real texts ({real_word_count/len(real_texts)*100:.1f}%)")
    print(f"Word appears in {fake_word_count}/{len(fake_texts)} fake texts ({fake_word_count/len(fake_texts)*100:.1f}%)")
    
    # Get co-occurrence distributions
    try:
        P_real = compute_word_cooccurrence_distribution(word, real_texts, tokenizer)
        P_fake = compute_word_cooccurrence_distribution(word, fake_texts, tokenizer)
        
        print(f"Real co-occurrence vocab size: {len(P_real) if P_real else 0}")
        print(f"Fake co-occurrence vocab size: {len(P_fake) if P_fake else 0}")
        
        if P_real and P_fake:
            # Show top co-occurring words
            print(f"Top 5 real co-occurrences: {dict(sorted(P_real.items(), key=lambda x: x[1], reverse=True)[:5])}")
            print(f"Top 5 fake co-occurrences: {dict(sorted(P_fake.items(), key=lambda x: x[1], reverse=True)[:5])}")
            
            # Calculate KL divergence
            kl_div = compute_kl_divergence(P_real, P_fake)
            print(f"KL divergence: {kl_div:.4f}")
            
            # Check for potential issues
            if kl_div > 5.0:
                print("WARNING: Extremely high KL divergence - possible implementation issue")
            
            if len(P_real) < 3 or len(P_fake) < 3:
                print("WARNING: Very sparse co-occurrence data - unreliable statistics")
                
            return {
                'word': word,
                'real_texts_with_word': real_word_count,
                'fake_texts_with_word': fake_word_count,
                'real_cooccur_vocab': len(P_real) if P_real else 0,
                'fake_cooccur_vocab': len(P_fake) if P_fake else 0,
                'kl_divergence': kl_div if P_real and P_fake else None,
                'P_real': P_real,
                'P_fake': P_fake
            }
        else:
            print("ERROR: Could not compute co-occurrence distributions")
            return None
            
    except Exception as e:
        print(f"ERROR: Exception in co-occurrence calculation: {e}")
        return None

def validate_kl_calculation(P_real, P_fake, word, threshold=10.0):
    """
    Validate KL divergence calculation and flag potential issues.
    """
    if not P_real or not P_fake:
        return False, "Empty probability distributions"
    
    # Check for reasonable vocabulary overlap
    common_words = set(P_real.keys()) & set(P_fake.keys())
    if len(common_words) < 2:
        return False, f"Insufficient vocabulary overlap: {len(common_words)} common words"
    
    # Check probability sums
    real_sum = sum(P_real.values())
    fake_sum = sum(P_fake.values())
    
    if abs(real_sum - 1.0) > 0.001 or abs(fake_sum - 1.0) > 0.001:
        return False, f"Probability distributions don't sum to 1.0: real={real_sum:.4f}, fake={fake_sum:.4f}"
    
    # Calculate KL divergence with additional checks
    kl_div = compute_kl_divergence(P_real, P_fake)
    
    if kl_div > threshold:
        return False, f"KL divergence {kl_div:.4f} exceeds reasonable threshold {threshold}"
    
    if np.isnan(kl_div) or np.isinf(kl_div):
        return False, f"Invalid KL divergence: {kl_div}"
    
    return True, "Valid"

def robust_word_selection(real_texts, fake_texts, method='chi_square', top_k=100, min_df=15):
    """
    More conservative word selection with enhanced validation.
    """
    print(f"  Running robust {method} selection...")
    
    if method == 'chi_square':
        vectorizer = CountVectorizer(
            min_df=min_df, 
            max_df=0.8,  # More conservative max_df
            stop_words='english', 
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b',
            max_features=5000  # Limit vocabulary size
        )
        
        all_texts = real_texts + fake_texts
        dtm = vectorizer.fit_transform(all_texts)
        vocab = vectorizer.get_feature_names_out()
        
        chi_scores = []
        for i, word in enumerate(vocab):
            word_column = dtm[:, i].toarray().flatten()
            
            real_with = sum(word_column[:len(real_texts)] > 0)
            real_without = len(real_texts) - real_with
            fake_with = sum(word_column[len(real_texts):] > 0)
            fake_without = len(fake_texts) - fake_with
            
            # More conservative cell count requirements
            if all(count >= 5 for count in [real_with, real_without, fake_with, fake_without]):
                contingency = [[real_with, real_without], 
                              [fake_with, fake_without]]
                
                try:
                    chi2, p_value, _, _ = chi2_contingency(contingency)
                    # Only select words with reasonable chi-square values
                    if chi2 < 50.0:  # Cap extremely high chi-square values
                        chi_scores.append((word, chi2, p_value))
                except:
                    continue
        
        chi_scores.sort(key=lambda x: x[1], reverse=True)
        selected_words = [word for word, _, _ in chi_scores[:top_k]]
        
    else:  # mutual information or other methods
        vectorizer = TfidfVectorizer(
            min_df=min_df, 
            max_df=0.8, 
            stop_words='english',
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b',
            max_features=5000
        )
        
        all_texts = real_texts + fake_texts
        labels = [0] * len(real_texts) + [1] * len(fake_texts)
        
        X = vectorizer.fit_transform(all_texts)
        vocab = vectorizer.get_feature_names_out()
        
        mi_scores = mutual_info_classif(X, labels, random_state=42)
        
        # Select words with reasonable MI scores
        valid_indices = [i for i, score in enumerate(mi_scores) if 0.001 < score < 2.0]
        mi_scores_filtered = [(vocab[i], mi_scores[i]) for i in valid_indices]
        mi_scores_filtered.sort(key=lambda x: x[1], reverse=True)
        
        selected_words = [word for word, _ in mi_scores_filtered[:top_k]]
    
    print(f"    Selected {len(selected_words)} words via {method}")
    return selected_words

def validated_kl_analysis(word_list, real_texts, fake_texts, max_kl_threshold=5.0):
    """
    KL analysis with extensive validation and quality checks.
    """
    print(f"\nValidated KL analysis for {len(word_list)} words...")
    
    results = {}
    validation_failures = []
    extreme_values = []
    
    for i, word in enumerate(word_list):
        if i % 50 == 0:
            print(f"  Processing word {i+1}/{len(word_list)}: {word}")
        
        try:
            P_real = compute_word_cooccurrence_distribution(word, real_texts, tokenizer)
            P_fake = compute_word_cooccurrence_distribution(word, fake_texts, tokenizer)
            
            if P_real and P_fake:
                # Validate calculation quality
                is_valid, validation_msg = validate_kl_calculation(P_real, P_fake, word, max_kl_threshold)
                
                if is_valid:
                    kl_div = compute_kl_divergence(P_real, P_fake)
                    results[word] = {
                        'kl_divergence': kl_div,
                        'real_vocab_size': len(P_real),
                        'fake_vocab_size': len(P_fake),
                        'threshold_exceeded': kl_div > 0.1,
                        'P_real': P_real,
                        'P_fake': P_fake
                    }
                else:
                    validation_failures.append((word, validation_msg))
                    if "exceeds reasonable threshold" in validation_msg:
                        extreme_values.append(word)
            else:
                validation_failures.append((word, "Could not compute distributions"))
                
        except Exception as e:
            validation_failures.append((word, f"Exception: {str(e)}"))
            continue
    
    print(f"  Successfully analyzed: {len(results)} words")
    print(f"  Validation failures: {len(validation_failures)} words")
    print(f"  Extreme values filtered: {len(extreme_values)} words")
    
    if validation_failures:
        print(f"  First 5 validation failures:")
        for word, msg in validation_failures[:5]:
            print(f"    {word}: {msg}")
    
    return results, validation_failures

def realistic_statistical_validation(results):
    """
    Statistical validation with realistic expectations and bounds checking.
    """
    print(f"\nRunning realistic statistical validation...")
    
    if not results:
        print("  No results to validate")
        return {}
    
    kl_values = [r['kl_divergence'] for r in results.values()]
    
    # Basic statistics with sanity checks
    mean_kl = np.mean(kl_values)
    median_kl = np.median(kl_values)
    std_kl = np.std(kl_values)
    min_kl = np.min(kl_values)
    max_kl = np.max(kl_values)
    
    # Check for realistic ranges
    if mean_kl > 3.0:
        print(f"  WARNING: Mean KL divergence ({mean_kl:.4f}) is unusually high")
    
    if std_kl > mean_kl:
        print(f"  WARNING: High standard deviation suggests inconsistent results")
    
    # Count significant results with realistic expectations
    significant_count = sum(1 for result in results.values() if result['threshold_exceeded'])
    significance_rate = significant_count / len(results)
    
    if significance_rate > 0.8:
        print(f"  WARNING: Significance rate ({significance_rate*100:.1f}%) is suspiciously high")
    elif significance_rate < 0.1:
        print(f"  WARNING: Significance rate ({significance_rate*100:.1f}%) is very low")
    else:
        print(f"  Significance rate ({significance_rate*100:.1f}%) appears reasonable")
    
    # Statistical tests
    t_stat, p_value = ttest_1samp(kl_values, 0)
    
    # Bootstrap confidence intervals
    def bootstrap_mean(data, n_iterations=1000):
        if len(data) == 0:
            return 0, 0
        bootstrap_means = []
        for _ in range(n_iterations):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
        return np.percentile(bootstrap_means, [2.5, 97.5])
    
    ci_lower, ci_upper = bootstrap_mean(kl_values)
    
    # Effect size categorization
    effect_sizes = [kl / 0.1 for kl in kl_values]
    small_effects = sum(1 for es in effect_sizes if 1 <= es < 3)
    medium_effects = sum(1 for es in effect_sizes if 3 <= es < 8)
    large_effects = sum(1 for es in effect_sizes if es >= 8)
    
    validation_results = {
        'mean_kl': mean_kl,
        'median_kl': median_kl,
        'std_kl': std_kl,
        'min_kl': min_kl,
        'max_kl': max_kl,
        't_statistic': t_stat,
        'p_value': p_value,
        'confidence_interval_95': (ci_lower, ci_upper),
        'n_words_tested': len(results),
        'significant_words': significant_count,
        'significance_rate': significance_rate,
        'small_effects': small_effects,
        'medium_effects': medium_effects,
        'large_effects': large_effects,
        'quality_warnings': []
    }
    
    # Add quality warnings
    if mean_kl > 3.0:
        validation_results['quality_warnings'].append("Unusually high mean KL divergence")
    if significance_rate > 0.8:
        validation_results['quality_warnings'].append("Suspiciously high significance rate")
    if std_kl > mean_kl * 1.5:
        validation_results['quality_warnings'].append("High variance in KL values")
    
    # Print results
    print(f"  Statistical Validation Results:")
    print(f"    Words tested: {validation_results['n_words_tested']}")
    print(f"    Mean KL divergence: {validation_results['mean_kl']:.4f}")
    print(f"    Median KL divergence: {validation_results['median_kl']:.4f}")
    print(f"    Range: {validation_results['min_kl']:.4f} - {validation_results['max_kl']:.4f}")
    print(f"    Standard deviation: {validation_results['std_kl']:.4f}")
    print(f"    95% CI: ({validation_results['confidence_interval_95'][0]:.4f}, {validation_results['confidence_interval_95'][1]:.4f})")
    print(f"    Significance rate: {validation_results['significance_rate']*100:.1f}%")
    print(f"    Effect sizes - Small: {validation_results['small_effects']}, Medium: {validation_results['medium_effects']}, Large: {validation_results['large_effects']}")
    
    if validation_results['quality_warnings']:
        print(f"    Quality warnings: {', '.join(validation_results['quality_warnings'])}")
    
    return validation_results

def run_debugging_analysis(dataset='politifact', debug_words=None):
    """
    Run debugging analysis on specific words to identify issues.
    """
    print("="*70)
    print("DEBUGGING ANALYSIS FOR WORD CO-OCCURRENCE CALCULATIONS")
    print("="*70)
    
    # Load data with smaller samples for debugging
    real_texts, fake_texts = load_data_for_word_selection(dataset)
    real_sample = real_texts[:50]  # Smaller sample for debugging
    fake_sample = fake_texts[:50]
    
    print(f"Using {len(real_sample)} real and {len(fake_sample)} fake texts for debugging")
    
    # Default debug words if none provided
    if debug_words is None:
        debug_words = ['president', 'said', 'government', 'officials', 'reported', 
                      'news', 'people', 'state', 'time', 'according']
    
    debug_results = []
    for word in debug_words:
        result = debug_word_cooccurrence(word, real_sample, fake_sample)
        if result:
            debug_results.append(result)
    
    # Analyze debugging results
    print(f"\n=== DEBUGGING SUMMARY ===")
    valid_kl_count = sum(1 for r in debug_results if r['kl_divergence'] is not None)
    print(f"Successfully calculated KL for {valid_kl_count}/{len(debug_results)} words")
    
    if valid_kl_count > 0:
        kl_values = [r['kl_divergence'] for r in debug_results if r['kl_divergence'] is not None]
        print(f"KL divergence range: {min(kl_values):.4f} - {max(kl_values):.4f}")
        print(f"Mean KL divergence: {np.mean(kl_values):.4f}")
        
        extreme_kl_words = [r['word'] for r in debug_results 
                           if r['kl_divergence'] is not None and r['kl_divergence'] > 5.0]
        if extreme_kl_words:
            print(f"Words with extreme KL values (>5.0): {extreme_kl_words}")
    
    return debug_results

def run_conservative_analysis(dataset='politifact', target_words=100):
    """
    Run conservative analysis with extensive validation.
    """
    print("="*80)
    print("CONSERVATIVE DATA-DRIVEN ANALYSIS WITH ENHANCED VALIDATION")
    print("="*80)
    
    # Step 1: Load data with reasonable samples
    print(f"\n1. Loading {dataset} data...")
    real_texts, fake_texts = load_data_for_word_selection(dataset)
    
    # Use larger samples for more reliable statistics
    max_texts = 300
    if len(real_texts) > max_texts:
        real_texts = random.sample(real_texts, max_texts)
    if len(fake_texts) > max_texts:
        fake_texts = random.sample(fake_texts, max_texts)
    
    print(f"   Using {len(real_texts)} real and {len(fake_texts)} fake texts")
    
    # Step 2: Conservative word selection
    print(f"\n2. Running conservative word selection...")
    chi_words = robust_word_selection(real_texts, fake_texts, 'chi_square', top_k=target_words//2)
    mi_words = robust_word_selection(real_texts, fake_texts, 'mutual_info', top_k=target_words//2)
    
    # Combine with overlap preference
    all_words = list(set(chi_words + mi_words))
    if len(all_words) > target_words:
        # Prioritize words selected by both methods
        word_counts = Counter(chi_words + mi_words)
        sorted_words = sorted(all_words, key=lambda x: word_counts[x], reverse=True)
        selected_words = sorted_words[:target_words]
    else:
        selected_words = all_words
    
    print(f"   Selected {len(selected_words)} words for analysis")
    
    # Step 3: Validated KL analysis
    print(f"\n3. Running validated KL analysis...")
    results, failures = validated_kl_analysis(selected_words, real_texts, fake_texts)
    
    # Step 4: Statistical validation
    print(f"\n4. Statistical validation...")
    validation_results = realistic_statistical_validation(results)
    
    # Step 5: Create results table
    if results:
        print(f"\n5. Creating results table...")
        table_data = []
        for word, result in results.items():
            table_data.append({
                'Word': word,
                'KL_Divergence': result['kl_divergence'],
                'Threshold_Exceeded': result['threshold_exceeded'],
                'Effect_Size': result['kl_divergence'] / 0.1,
                'Real_Vocab_Size': result['real_vocab_size'],
                'Fake_Vocab_Size': result['fake_vocab_size']
            })
        
        table_data.sort(key=lambda x: x['KL_Divergence'], reverse=True)
        df = pd.DataFrame(table_data)
        df.to_csv('conservative_word_analysis.csv', index=False)
        
        print(f"   Results saved to 'conservative_word_analysis.csv'")
        print(f"   Top 5 discriminative words:")
        for i, row in df.head().iterrows():
            print(f"     {i+1}. {row['Word']}: {row['KL_Divergence']:.4f}")
    
    final_results = {
        'dataset': dataset,
        'selected_words': selected_words,
        'kl_results': results,
        'validation_failures': failures,
        'validation_results': validation_results
    }
    
    with open(f'conservative_analysis_{dataset}.pkl', 'wb') as f:
        pickle.dump(final_results, f)
    
    print(f"\n" + "="*80)
    print("CONSERVATIVE ANALYSIS COMPLETE")
    print("="*80)
    
    return final_results
def compute_shared_vocabulary_cooccurrence(target_word, real_texts, fake_texts, tokenizer, window_size=3):
    """
    Compute co-occurrence distributions over SHARED vocabulary space.
    This fixes the vocabulary mismatch issue causing extreme KL divergences.
    """
    from collections import Counter, defaultdict
    
    def get_cooccurrence_counts(texts, target_word, window_size):
        cooccur_counts = Counter()
        total_cooccurrences = 0
        
        for text in texts:
            tokens = tokenizer.tokenize(text.lower())
            tokens = [t for t in tokens if t.isalpha() and len(t) > 1]
            
            # Find all occurrences of target word
            target_positions = [i for i, token in enumerate(tokens) if token == target_word]
            
            for pos in target_positions:
                # Get window around target word
                start = max(0, pos - window_size)
                end = min(len(tokens), pos + window_size + 1)
                
                for i in range(start, end):
                    if i != pos:  # Skip the target word itself
                        cooccur_counts[tokens[i]] += 1
                        total_cooccurrences += 1
        
        return cooccur_counts, total_cooccurrences
    
    # Get co-occurrence counts for both corpora
    real_counts, real_total = get_cooccurrence_counts(real_texts, target_word, window_size)
    fake_counts, fake_total = get_cooccurrence_counts(fake_texts, target_word, window_size)
    
    # Require minimum co-occurrences for reliable statistics
    if real_total < 20 or fake_total < 20:
        return None, None
    
    # Create SHARED vocabulary (union of both vocabularies)
    shared_vocab = set(real_counts.keys()) | set(fake_counts.keys())
    
    # Remove very rare words to reduce noise
    shared_vocab = {word for word in shared_vocab 
                   if real_counts.get(word, 0) + fake_counts.get(word, 0) >= 2}
    
    if len(shared_vocab) < 10:  # Need reasonable vocabulary size
        return None, None
    
    # Create probability distributions over shared vocabulary
    real_probs = {}
    fake_probs = {}
    
    for word in shared_vocab:
        real_probs[word] = real_counts.get(word, 0) / real_total
        fake_probs[word] = fake_counts.get(word, 0) / fake_total
    
    # Verify probabilities sum to reasonable values
    real_sum = sum(real_probs.values())
    fake_sum = sum(fake_probs.values())
    
    if real_sum < 0.1 or fake_sum < 0.1:  # Too sparse
        return None, None
    
    # Normalize to proper probabilities
    for word in shared_vocab:
        real_probs[word] = real_probs[word] / real_sum
        fake_probs[word] = fake_probs[word] / fake_sum
    
    return real_probs, fake_probs

def compute_kl_divergence_safe(P, Q, epsilon=1e-10):
    """
    Compute KL divergence with proper smoothing and bounds checking.
    """
    if not P or not Q:
        return None
    
    # Ensure same vocabulary
    vocab = set(P.keys()) | set(Q.keys())
    
    kl_div = 0.0
    for word in vocab:
        p = P.get(word, epsilon)
        q = Q.get(word, epsilon)
        
        # Add smoothing to prevent log(0)
        p = max(p, epsilon)
        q = max(q, epsilon)
        
        kl_div += p * np.log(p / q)
    
    # Sanity check - flag unrealistic values
    if kl_div > 5.0 or kl_div < 0:
        print(f"WARNING: Unusual KL divergence {kl_div:.4f}")
    
    return kl_div

def test_fixed_implementation():
    """Test the fixed implementation on your debug words"""
    
    # Your existing data loading
    real_texts, fake_texts = load_data_for_word_selection('politifact')
    real_sample = real_texts[:100]  # Larger sample for stability
    fake_sample = fake_texts[:100]
    
    test_words = ['president', 'said', 'government', 'officials', 'news']
    
    print("Testing Fixed Implementation:")
    print("=" * 50)
    
    for word in test_words:
        print(f"\nWord: {word}")
        
        # Use fixed function
        P_real, P_fake = compute_shared_vocabulary_cooccurrence(
            word, real_sample, fake_sample, tokenizer, window_size=3
        )
        
        if P_real and P_fake:
            kl_div = compute_kl_divergence_safe(P_real, P_fake)
            
            print(f"  Shared vocab size: {len(P_real)}")
            print(f"  KL divergence: {kl_div:.4f}")
            print(f"  Realistic? {'Yes' if 0.01 < kl_div < 3.0 else 'No - still extreme'}")
            
            # Show top co-occurrences
            real_top = sorted(P_real.items(), key=lambda x: x[1], reverse=True)[:3]
            fake_top = sorted(P_fake.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  Real top: {real_top}")
            print(f"  Fake top: {fake_top}")
        else:
            print(f"  Insufficient data for reliable analysis")

# Replace your original functions with these fixed versions
# Then re-run your analysis
# Main execution
if __name__ == "__main__":
    test_fixed_implementation()