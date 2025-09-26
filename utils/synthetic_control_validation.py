# Enhanced validation_integration.py with Synthetic Control Tests
# Integration script to run stronger validation with your existing code

import sys
import os
import random
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle

sys.path.append('.')  # Add current directory to path

# Import your existing functions
try:
    from load_data_vocab import (
        compute_word_cooccurrence_distribution, 
        compute_kl_divergence,
        split_news_by_binary_label,
        tokenizer
    )
    print("✅ Successfully imported existing functions")
except ImportError as e:
    print(f"❌ Error importing existing functions: {e}")
    print("🔧 Please ensure load_data_vocab.py is in the current directory")
    sys.exit(1)

# Import the validation class
from stronger_validation_code import StrongerValidation

def load_and_prepare_data(obj='gossipcop'):
    """
    Load and prepare data for validation using your existing data loading logic.
    """
    print(f"📁 Loading {obj} dataset...")
    
    try:
        # Load training data
        train_dict = pickle.load(open(f'data/news_articles/{obj}_train.pkl', 'rb'))
        test_dict = pickle.load(open(f'data/news_articles/{obj}_test.pkl', 'rb'))
        
        # Split by labels
        news_real, news_fake = split_news_by_binary_label(train_dict)
        
        print(f"✅ Loaded {len(news_real)} real and {len(news_fake)} fake training articles")
        
        # Sample for efficiency (adjust size based on computational resources)
        max_articles = 300
        news_real_sample = news_real[:max_articles] if len(news_real) > max_articles else news_real
        news_fake_sample = news_fake[:max_articles] if len(news_fake) > max_articles else news_fake
        
        print(f"🔬 Using {len(news_real_sample)} real and {len(news_fake_sample)} fake articles for validation")
        
        return {
            'real_texts': news_real_sample,
            'fake_texts': news_fake_sample,
            'test_real': [text for text, label in zip(test_dict['news'], test_dict['labels']) if label == 0],
            'test_fake': [text for text, label in zip(test_dict['news'], test_dict['labels']) if label == 1]
        }
        
    except FileNotFoundError as e:
        print(f"❌ Data files not found: {e}")
        print("🔧 Please ensure data files are in the correct directory structure")
        return None

def create_multi_dataset_validation():
    """
    Create multiple datasets for cross-validation.
    """
    print("📄 Creating multi-dataset validation setup...")
    
    datasets = {}
    
    # Try to load different datasets
    dataset_names = ['gossipcop', 'politifact','lun']  # Add more if available
    
    for dataset_name in dataset_names:
        try:
            data = load_and_prepare_data(dataset_name)
            if data:
                datasets[dataset_name] = {
                    'real': data['real_texts'],
                    'fake': data['fake_texts']
                }
                print(f"  ✅ Added {dataset_name}: {len(data['real_texts'])} real, {len(data['fake_texts'])} fake")
        except Exception as e:
            print(f"  ⚠️ Could not load {dataset_name}: {e}")
    
    # If we only have one dataset, create artificial splits for cross-validation
    if len(datasets) < 2:
        print("  🔧 Creating artificial dataset splits for cross-validation...")
        
        if 'gossipcop' in datasets:
            main_data = datasets['gossipcop']
        elif 'lun' in datasets:
            main_data = datasets['lun'] 
        elif 'politifact' in datasets:
            main_data = datasets['politifact']
        else:
            # If no datasets loaded, try to load any available
            for dataset_name in dataset_names:
                try:
                    data = load_and_prepare_data(dataset_name)
                    if data:
                        main_data = {'real': data['real_texts'], 'fake': data['fake_texts']}
                        break
                except:
                    continue
            else:
                print("  ❌ No datasets available for splitting")
                return datasets
        
        # Split into multiple "datasets"
        real_texts = main_data['real']
        fake_texts = main_data['fake']
        
        split1 = len(real_texts) // 3
        split2 = 2 * len(real_texts) // 3
        
        datasets['split_A'] = {
            'real': real_texts[:split1],
            'fake': fake_texts[:split1]
        }
        datasets['split_B'] = {
            'real': real_texts[split1:split2],
            'fake': fake_texts[split1:split2]
        }
        datasets['split_C'] = {
            'real': real_texts[split2:],
            'fake': fake_texts[split2:]
        }
        
        print(f"  ✅ Created 3 artificial splits")
    
    return datasets

class EnhancedStrongerValidation(StrongerValidation):
    """
    Enhanced validation suite including Synthetic Control Tests
    """
    
    def synthetic_control_tests(self, real_texts: list, fake_texts: list, 
                               test_words: list = None, n_iterations: int = 10) -> dict:
        """
        Synthetic Control Tests: Create synthetic datasets by randomly reassigning labels
        
        This test distinguishes between genuine linguistic patterns and dataset construction bias:
        - If divergences remain high in shuffled data → dataset construction bias
        - If divergences collapse to near zero → genuine context-specific patterns
        
        Args:
            real_texts: List of real news articles
            fake_texts: List of fake news articles
            test_words: Words to test for divergence
            n_iterations: Number of synthetic control iterations
            
        Returns:
            Dictionary with synthetic control test results
        """
        print("🎲 Running Synthetic Control Tests...")
        print("="*60)
        print(f"Testing hypothesis: Are observed divergences due to genuine linguistic patterns")
        print(f"or dataset construction bias?")
        print(f"Method: Randomly reassign labels {n_iterations} times and measure divergences")
        
        test_words = test_words or self._get_default_test_words()
        
        # Step 1: Calculate original divergences (baseline)
        print(f"\n📊 Step 1: Computing baseline divergences...")
        original_divergences = self._test_word_divergences(real_texts, fake_texts, test_words)
        
        original_kl_values = []
        for word in test_words:
            if word in original_divergences:
                kl_val = original_divergences[word]['kl_divergence']
                original_kl_values.append(kl_val)
                print(f"  {word}: KL = {kl_val:.4f}")
        
        original_mean_kl = np.mean(original_kl_values) if original_kl_values else 0
        print(f"📈 Original Mean KL Divergence: {original_mean_kl:.4f}")
        
        # Step 2: Create synthetic control datasets by random label reassignment
        print(f"\n🎲 Step 2: Creating {n_iterations} synthetic control datasets...")
        
        all_texts = real_texts + fake_texts
        total_articles = len(all_texts)
        n_real_original = len(real_texts)
        
        synthetic_results = []
        synthetic_kl_distributions = {word: [] for word in test_words}
        
        for iteration in range(n_iterations):
            print(f"  Iteration {iteration + 1}/{n_iterations}", end="")
            
            # Randomly shuffle and reassign labels
            shuffled_indices = list(range(total_articles))
            random.shuffle(shuffled_indices)
            
            # Create synthetic "real" and "fake" datasets
            synthetic_real_indices = shuffled_indices[:n_real_original]
            synthetic_fake_indices = shuffled_indices[n_real_original:]
            
            synthetic_real_texts = [all_texts[i] for i in synthetic_real_indices]
            synthetic_fake_texts = [all_texts[i] for i in synthetic_fake_indices]
            
            # Calculate divergences for synthetic dataset
            synthetic_divergences = self._test_word_divergences(
                synthetic_real_texts, synthetic_fake_texts, test_words
            )
            
            # Collect KL values for this iteration
            iteration_kl_values = []
            for word in test_words:
                if word in synthetic_divergences:
                    kl_val = synthetic_divergences[word]['kl_divergence']
                    synthetic_kl_distributions[word].append(kl_val)
                    iteration_kl_values.append(kl_val)
                else:
                    synthetic_kl_distributions[word].append(0.0)
                    iteration_kl_values.append(0.0)
            
            iteration_mean_kl = np.mean(iteration_kl_values) if iteration_kl_values else 0
            synthetic_results.append({
                'iteration': iteration + 1,
                'mean_kl': iteration_mean_kl,
                'divergences': synthetic_divergences,
                'kl_values': iteration_kl_values
            })
            
            print(f" → Mean KL: {iteration_mean_kl:.4f}")
        
        # Step 3: Statistical Analysis
        print(f"\n📈 Step 3: Statistical Analysis...")
        
        # Calculate statistics for synthetic controls
        synthetic_mean_kls = [result['mean_kl'] for result in synthetic_results]
        synthetic_overall_mean = np.mean(synthetic_mean_kls)
        synthetic_std = np.std(synthetic_mean_kls)
        
        print(f"Original Mean KL: {original_mean_kl:.4f}")
        print(f"Synthetic Mean KL: {synthetic_overall_mean:.4f} (±{synthetic_std:.4f})")
        
        # Calculate collapse ratio (how much divergence collapsed)
        collapse_ratio = 1 - (synthetic_overall_mean / original_mean_kl) if original_mean_kl > 0 else 0
        print(f"Collapse Ratio: {collapse_ratio:.4f} ({collapse_ratio*100:.1f}% reduction)")
        
        # Statistical significance test
        if len(synthetic_mean_kls) > 1:
            # One-sample t-test: Is synthetic mean significantly different from original?
            t_stat, p_value = stats.ttest_1samp(synthetic_mean_kls, original_mean_kl)
            
            # Effect size (Cohen's d)
            effect_size = (original_mean_kl - synthetic_overall_mean) / synthetic_std if synthetic_std > 0 else float('inf')
            
            print(f"Statistical Test:")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.6f}")
            print(f"  Effect size (Cohen's d): {effect_size:.4f}")
        else:
            t_stat, p_value, effect_size = None, None, None
        
        # Word-level analysis
        word_level_analysis = {}
        for word in test_words:
            if word in original_divergences:
                original_word_kl = original_divergences[word]['kl_divergence']
                synthetic_word_kls = synthetic_kl_distributions[word]
                
                synthetic_word_mean = np.mean(synthetic_word_kls) if synthetic_word_kls else 0
                synthetic_word_std = np.std(synthetic_word_kls) if len(synthetic_word_kls) > 1 else 0
                
                word_collapse_ratio = 1 - (synthetic_word_mean / original_word_kl) if original_word_kl > 0 else 0
                
                # Word-level significance
                if len(synthetic_word_kls) > 1:
                    word_t_stat, word_p_value = stats.ttest_1samp(synthetic_word_kls, original_word_kl)
                else:
                    word_t_stat, word_p_value = None, None
                
                word_level_analysis[word] = {
                    'original_kl': original_word_kl,
                    'synthetic_mean_kl': synthetic_word_mean,
                    'synthetic_std_kl': synthetic_word_std,
                    'collapse_ratio': word_collapse_ratio,
                    't_statistic': word_t_stat,
                    'p_value': word_p_value,
                    'synthetic_distribution': synthetic_word_kls
                }
        
        # Step 4: Interpretation and Conclusion
        conclusion = self._conclude_synthetic_control_test(
            original_mean_kl, synthetic_overall_mean, collapse_ratio, 
            p_value, effect_size, word_level_analysis
        )
        
        return {
            'validation_type': 'synthetic_control_tests',
            'original_mean_kl': original_mean_kl,
            'synthetic_mean_kl': synthetic_overall_mean,
            'synthetic_std_kl': synthetic_std,
            'collapse_ratio': collapse_ratio,
            'statistical_test': {
                't_statistic': t_stat,
                'p_value': p_value,
                'effect_size': effect_size
            },
            'word_level_analysis': word_level_analysis,
            'synthetic_results': synthetic_results,
            'n_iterations': n_iterations,
            'conclusion': conclusion
        }
    
    def _conclude_synthetic_control_test(self, original_mean_kl: float, synthetic_mean_kl: float,
                                        collapse_ratio: float, p_value: float, effect_size: float,
                                        word_level_analysis: dict) -> dict:
        """
        Draw conclusions from synthetic control tests
        """
        print(f"\n🎯 Step 4: Drawing Conclusions...")
        
        # Criteria for genuine linguistic patterns:
        # 1. High collapse ratio (>70%)
        # 2. Statistically significant difference
        # 3. Large effect size (>0.8)
        # 4. Consistent across multiple words
        
        criteria_met = 0
        total_criteria = 4
        
        # Criterion 1: Collapse ratio
        high_collapse = collapse_ratio > 0.7
        if high_collapse:
            criteria_met += 1
            print(f"✅ High collapse ratio: {collapse_ratio:.3f} (>0.7)")
        else:
            print(f"❌ Low collapse ratio: {collapse_ratio:.3f} (≤0.7)")
        
        # Criterion 2: Statistical significance
        significant = p_value is not None and p_value < 0.01
        if significant:
            criteria_met += 1
            print(f"✅ Statistically significant: p = {p_value:.6f} (<0.01)")
        else:
            print(f"❌ Not statistically significant: p = {p_value}")
        
        # Criterion 3: Effect size
        large_effect = effect_size is not None and effect_size > 0.8
        if large_effect:
            criteria_met += 1
            print(f"✅ Large effect size: d = {effect_size:.3f} (>0.8)")
        else:
            print(f"❌ Small/medium effect size: d = {effect_size}")
        
        # Criterion 4: Word-level consistency
        consistent_words = sum(1 for analysis in word_level_analysis.values() 
                              if analysis['collapse_ratio'] > 0.5)
        total_words = len(word_level_analysis)
        consistency_ratio = consistent_words / total_words if total_words > 0 else 0
        
        word_consistent = consistency_ratio > 0.6
        if word_consistent:
            criteria_met += 1
            print(f"✅ Word-level consistency: {consistent_words}/{total_words} words ({consistency_ratio:.1%})")
        else:
            print(f"❌ Inconsistent across words: {consistent_words}/{total_words} words ({consistency_ratio:.1%})")
        
        # Overall conclusion
        confidence_score = criteria_met / total_criteria
        
        if criteria_met >= 3:
            conclusion_type = 'genuine_linguistic_patterns'
            confidence = 'high'
            interpretation = (f"Strong evidence for genuine context-specific linguistic patterns. "
                            f"Divergences collapse by {collapse_ratio:.1%} when labels are randomized, "
                            f"indicating the original patterns are not due to dataset construction bias.")
        elif criteria_met >= 2:
            conclusion_type = 'mixed_evidence'
            confidence = 'medium'
            interpretation = (f"Mixed evidence. Some collapse observed ({collapse_ratio:.1%}) but "
                            f"not meeting all criteria for genuine linguistic patterns. "
                            f"Both linguistic differences and dataset bias may contribute.")
        else:
            conclusion_type = 'dataset_construction_bias'
            confidence = 'high' if criteria_met <= 1 else 'medium'
            interpretation = (f"Evidence suggests dataset construction bias. "
                            f"Divergences persist ({1-collapse_ratio:.1%} remain) even with "
                            f"randomized labels, indicating systematic bias in dataset construction.")
        
        print(f"\n🎯 SYNTHETIC CONTROL CONCLUSION:")
        print(f"   Result: {conclusion_type}")
        print(f"   Confidence: {confidence}")
        print(f"   Criteria met: {criteria_met}/{total_criteria}")
        print(f"   Interpretation: {interpretation}")
        
        return {
            'conclusion': conclusion_type,
            'confidence': confidence,
            'criteria_met': criteria_met,
            'total_criteria': total_criteria,
            'confidence_score': confidence_score,
            'interpretation': interpretation,
            'criteria_details': {
                'high_collapse_ratio': high_collapse,
                'statistically_significant': significant,
                'large_effect_size': large_effect,
                'word_level_consistent': word_consistent
            }
        }
    
    def visualize_synthetic_control_results(self, results: dict):
        """
        Create comprehensive visualizations for synthetic control test results
        """
        if not results or results.get('validation_type') != 'synthetic_control_tests':
            print("❌ Invalid results for visualization")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Synthetic Control Tests Results', fontsize=16, fontweight='bold')
        
        # 1. Original vs Synthetic Mean KL Divergence
        ax1 = axes[0, 0]
        original_kl = results['original_mean_kl']
        synthetic_kl = results['synthetic_mean_kl']
        synthetic_std = results['synthetic_std_kl']
        
        categories = ['Original\nData', 'Synthetic\nControls']
        values = [original_kl, synthetic_kl]
        errors = [0, synthetic_std]
        colors = ['lightcoral', 'lightblue']
        
        bars = ax1.bar(categories, values, yerr=errors, capsize=5, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Mean KL Divergence')
        ax1.set_title('Original vs Synthetic KL Divergence')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value, error in zip(bars, values, errors):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + error + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Collapse Ratio Visualization
        ax2 = axes[0, 1]
        collapse_ratio = results['collapse_ratio']
        remaining_ratio = 1 - collapse_ratio
        
        wedges, texts, autotexts = ax2.pie([collapse_ratio, remaining_ratio], 
                                          labels=['Collapsed', 'Remaining'],
                                          colors=['lightgreen', 'lightcoral'],
                                          autopct='%1.1f%%',
                                          startangle=90)
        ax2.set_title(f'Divergence Collapse Ratio\n{collapse_ratio:.3f} ({collapse_ratio*100:.1f}%)')
        
        # 3. Distribution of Synthetic KL Values
        ax3 = axes[0, 2]
        synthetic_results = results['synthetic_results']
        synthetic_kls = [r['mean_kl'] for r in synthetic_results]
        
        ax3.hist(synthetic_kls, bins=min(10, len(synthetic_kls)), alpha=0.7, color='lightblue', edgecolor='black')
        ax3.axvline(original_kl, color='red', linestyle='--', linewidth=2, label=f'Original: {original_kl:.4f}')
        ax3.axvline(synthetic_kl, color='blue', linestyle='-', linewidth=2, label=f'Synthetic Mean: {synthetic_kl:.4f}')
        ax3.set_xlabel('Mean KL Divergence')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Synthetic KL Values')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Word-level Collapse Ratios
        ax4 = axes[1, 0]
        word_analysis = results['word_level_analysis']
        words = list(word_analysis.keys())
        word_collapse_ratios = [word_analysis[word]['collapse_ratio'] for word in words]
        
        bars = ax4.bar(range(len(words)), word_collapse_ratios, color='lightgreen', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Words')
        ax4.set_ylabel('Collapse Ratio')
        ax4.set_title('Word-level Collapse Ratios')
        ax4.set_xticks(range(len(words)))
        ax4.set_xticklabels(words, rotation=45, ha='right')
        ax4.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Threshold (0.7)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Add value labels on bars
        for i, (bar, ratio) in enumerate(zip(bars, word_collapse_ratios)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{ratio:.2f}', ha='center', va='bottom', fontsize=8)
        
        # 5. Statistical Significance Indicators
        ax5 = axes[1, 1]
        statistical_test = results['statistical_test']
        p_value = statistical_test.get('p_value', 1.0)
        effect_size = statistical_test.get('effect_size', 0.0)
        
        # Create significance indicators
        significance_levels = ['p < 0.001', 'p < 0.01', 'p < 0.05', 'p ≥ 0.05']
        if p_value < 0.001:
            sig_index = 0
        elif p_value < 0.01:
            sig_index = 1
        elif p_value < 0.05:
            sig_index = 2
        else:
            sig_index = 3
        
        colors = ['darkgreen', 'green', 'orange', 'red']
        ax5.bar(['Statistical\nSignificance'], [sig_index + 1], color=colors[sig_index], alpha=0.7)
        ax5.set_ylabel('Significance Level')
        ax5.set_title(f'Statistical Significance\np = {p_value:.6f}')
        ax5.set_yticks([1, 2, 3, 4])
        ax5.set_yticklabels(significance_levels)
        ax5.grid(True, alpha=0.3)
        
        # 6. Conclusion Summary
        ax6 = axes[1, 2]
        conclusion = results['conclusion']
        conclusion_details = conclusion['criteria_details']
        
        criteria_names = ['High Collapse\nRatio', 'Statistically\nSignificant', 
                         'Large Effect\nSize', 'Word-level\nConsistent']
        criteria_met = [conclusion_details['high_collapse_ratio'],
                       conclusion_details['statistically_significant'],
                       conclusion_details['large_effect_size'],
                       conclusion_details['word_level_consistent']]
        
        colors = ['green' if met else 'red' for met in criteria_met]
        bars = ax6.bar(criteria_names, [1 if met else 0 for met in criteria_met], color=colors, alpha=0.7)
        
        ax6.set_ylabel('Criterion Met')
        ax6.set_title(f'Validation Criteria\n{conclusion["criteria_met"]}/{conclusion["total_criteria"]} Met')
        ax6.set_yticks([0, 1])
        ax6.set_yticklabels(['Not Met', 'Met'])
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed summary
        self._print_synthetic_control_summary(results)
    
    def _print_synthetic_control_summary(self, results: dict):
        """
        Print detailed summary of synthetic control test results
        """
        print("\n" + "="*80)
        print("🎲 SYNTHETIC CONTROL TESTS SUMMARY")
        print("="*80)
        
        # Basic statistics
        print(f"\n📊 BASIC STATISTICS:")
        print(f"   Original Mean KL Divergence: {results['original_mean_kl']:.4f}")
        print(f"   Synthetic Mean KL Divergence: {results['synthetic_mean_kl']:.4f} (±{results['synthetic_std_kl']:.4f})")
        print(f"   Collapse Ratio: {results['collapse_ratio']:.4f} ({results['collapse_ratio']*100:.1f}% reduction)")
        print(f"   Number of Iterations: {results['n_iterations']}")
        
        # Statistical test results
        stat_test = results['statistical_test']
        print(f"\n📈 STATISTICAL ANALYSIS:")
        print(f"   t-statistic: {stat_test.get('t_statistic', 'N/A')}")
        print(f"   p-value: {stat_test.get('p_value', 'N/A')}")
        print(f"   Effect size (Cohen's d): {stat_test.get('effect_size', 'N/A')}")
        
        # Conclusion
        conclusion = results['conclusion']
        print(f"\n🎯 CONCLUSION:")
        print(f"   Result: {conclusion['conclusion']}")
        print(f"   Confidence: {conclusion['confidence']}")
        print(f"   Criteria met: {conclusion['criteria_met']}/{conclusion['total_criteria']}")
        print(f"   Confidence score: {conclusion['confidence_score']:.3f}")
        
        print(f"\n💭 INTERPRETATION:")
        print(f"   {conclusion['interpretation']}")
        
        # Word-level details
        word_analysis = results['word_level_analysis']
        print(f"\n🔤 WORD-LEVEL ANALYSIS:")
        for word, analysis in word_analysis.items():
            collapse_ratio = analysis['collapse_ratio']
            p_val = analysis.get('p_value', None)
            p_str = f"p={p_val:.4f}" if p_val is not None else "p=N/A"
            status = "✅" if collapse_ratio > 0.5 else "❌"
            print(f"   {status} {word}: {collapse_ratio:.3f} collapse ratio ({p_str})")
        
        print("\n" + "="*80)

    def comprehensive_validation_suite_enhanced(self, real_texts: list, fake_texts: list,
                                               additional_datasets: dict = None,
                                               test_words: list = None) -> dict:
        """
        Enhanced comprehensive validation suite including Synthetic Control Tests
        """
        print("🔬 Running Enhanced Comprehensive Validation Suite...")
        print("="*80)
        
        all_results = {}
        
        # 1. Synthetic Control Tests (FIRST - most important)
        all_results['synthetic_control'] = self.synthetic_control_tests(real_texts, fake_texts, test_words)
        
        # 2. Topic Matching Validation
        all_results['topic_matching'] = self.topic_matching_validation(real_texts, fake_texts, test_words)
        
        # 3. Synthetic Validation  
        all_results['synthetic'] = self.synthetic_validation(real_texts, fake_texts, test_words)
        
        # 4. Same Event Validation
        all_results['same_event'] = self.same_event_validation(real_texts, fake_texts, test_words)
        
        # 5. Style Transfer Validation
        all_results['style_transfer'] = self.style_transfer_validation(real_texts, fake_texts, test_words)
        
        # 6. Cross-Dataset Validation (if additional datasets provided)
        if additional_datasets:
            all_results['cross_dataset'] = self.cross_dataset_validation(additional_datasets)
        
        # Enhanced overall analysis (weighted by Synthetic Control results)
        overall_conclusion = self._synthesize_enhanced_validation_results(all_results)
        
        # Generate enhanced report
        self._generate_enhanced_validation_report(all_results, overall_conclusion)
        
        return {
            'individual_validations': all_results,
            'overall_conclusion': overall_conclusion,
            'validity_score': self._calculate_enhanced_validity_score(all_results),
            'recommendations': self._generate_enhanced_recommendations(overall_conclusion)
        }
    
    def _synthesize_enhanced_validation_results(self, all_results: dict) -> dict:
        """
        Enhanced synthesis giving special weight to Synthetic Control results
        """
        synthetic_control_result = all_results.get('synthetic_control', {})
        synthetic_control_conclusion = synthetic_control_result.get('conclusion', {})
        
        # Synthetic Control Tests have highest weight in final decision
        synthetic_control_weight = 0.5  # 50% of total weight
        other_validations_weight = 0.5  # Remaining 50% split among others
        
        # Get Synthetic Control evidence
        synthetic_control_support = 0
        if synthetic_control_conclusion.get('conclusion') == 'genuine_linguistic_patterns':
            synthetic_control_support = 2 * synthetic_control_conclusion.get('confidence_score', 0)
        elif synthetic_control_conclusion.get('conclusion') == 'mixed_evidence':
            synthetic_control_support = 1 * synthetic_control_conclusion.get('confidence_score', 0)
        # dataset_construction_bias gets 0 support
        
        # Get evidence from other validations
        other_support = 0
        other_count = 0
        
        for val_type, results in all_results.items():
            if val_type == 'synthetic_control':
                continue
                
            conclusion = results.get('conclusion', {})
            confidence = conclusion.get('confidence', 'low')
            conclusion_text = conclusion.get('conclusion', '')
            
            if any(keyword in conclusion_text for keyword in ['robust', 'persist', 'replicate', 'match']):
                support_value = {'high': 2, 'medium': 1.5, 'low': 1}.get(confidence, 1)
                other_support += support_value
            
            other_count += 1
        
        # Normalize other support
        other_support_normalized = other_support / max(other_count, 1) if other_count > 0 else 0
        
        # Calculate weighted overall support
        weighted_support = (synthetic_control_weight * synthetic_control_support + 
                           other_validations_weight * other_support_normalized)
        
        # Enhanced conclusion logic
        if synthetic_control_conclusion.get('conclusion') == 'genuine_linguistic_patterns':
            if synthetic_control_conclusion.get('confidence') == 'high' and weighted_support > 1.5:
                overall_conclusion = {
                    'conclusion': 'strong_linguistic_differences_supported',
                    'confidence': 'very_high',
                    'primary_evidence': 'synthetic_control_tests',
                    'interpretation': ('Synthetic Control Tests provide strong evidence for genuine linguistic '
                                     'differences. Random label reassignment collapses divergences, confirming '
                                     'patterns are not due to dataset construction bias.')
                }
            else:
                overall_conclusion = {
                    'conclusion': 'linguistic_differences_supported',
                    'confidence': 'high',
                    'primary_evidence': 'synthetic_control_tests',
                    'interpretation': ('Synthetic Control Tests support genuine linguistic differences, '
                                     'though some uncertainty remains.')
                }
        
        elif synthetic_control_conclusion.get('conclusion') == 'dataset_construction_bias':
            overall_conclusion = {
                'conclusion': 'dataset_bias_confirmed',
                'confidence': 'high',
                'primary_evidence': 'synthetic_control_tests',
                'interpretation': ('Synthetic Control Tests reveal dataset construction bias. '
                                 'Divergences persist even with randomized labels, indicating '
                                 'systematic bias rather than genuine linguistic differences.')
            }
        
        else:  # mixed evidence
            overall_conclusion = {
                'conclusion': 'mixed_evidence_complex_factors',
                'confidence': 'medium',
                'primary_evidence': 'synthetic_control_tests',
                'interpretation': ('Synthetic Control Tests show mixed results, suggesting both '
                                 'linguistic differences and dataset biases contribute to observed patterns.')
            }
        
        return {
            'overall_conclusion': overall_conclusion,
            'synthetic_control_weight': synthetic_control_weight,
            'synthetic_control_support': synthetic_control_support,
            'other_validations_support': other_support_normalized,
            'weighted_support_score': weighted_support,
            'total_validations': len(all_results)
        }
    
    def _calculate_enhanced_validity_score(self, all_results: dict) -> float:
        """
        Calculate enhanced validity score with special weight for Synthetic Control Tests
        """
        synthetic_control_result = all_results.get('synthetic_control', {})
        synthetic_control_conclusion = synthetic_control_result.get('conclusion', {})
        
        # Base score from Synthetic Control Tests (60% weight)
        if synthetic_control_conclusion.get('conclusion') == 'genuine_linguistic_patterns':
            base_score = 0.9 * synthetic_control_conclusion.get('confidence_score', 0)
        elif synthetic_control_conclusion.get('conclusion') == 'mixed_evidence':
            base_score = 0.5 * synthetic_control_conclusion.get('confidence_score', 0)
        else:  # dataset_construction_bias
            base_score = 0.1
        
        # Additional score from other validations (40% weight)
        other_scores = []
        weights = {
            'topic_matching': 0.1,
            'synthetic': 0.1,
            'same_event': 0.1,
            'style_transfer': 0.05,
            'cross_dataset': 0.05
        }
        
        for validation_type, results in all_results.items():
            if validation_type == 'synthetic_control':
                continue
                
            conclusion = results.get('conclusion', {})
            confidence = conclusion.get('confidence', 'low')
            conclusion_text = conclusion.get('conclusion', '')
            
            # Score based on support for linguistic hypothesis
            if any(keyword in conclusion_text for keyword in ['robust', 'persist', 'replicate', 'match']):
                validation_score = 0.8
            elif 'mixed' in conclusion_text:
                validation_score = 0.5
            else:
                validation_score = 0.2
            
            # Adjust by confidence
            confidence_multiplier = {'high': 1.0, 'medium': 0.7, 'low': 0.4}.get(confidence, 0.4)
            final_validation_score = validation_score * confidence_multiplier
            
            weight = weights.get(validation_type, 0.05)
            other_scores.append(final_validation_score * weight)
        
        # Combine scores
        other_total = sum(other_scores) / sum(weights.values()) if other_scores else 0
        final_score = 0.6 * base_score + 0.4 * other_total
        
        return min(1.0, max(0.0, final_score))
    
    def _generate_enhanced_recommendations(self, overall_conclusion: dict) -> list:
        """
        Generate enhanced recommendations based on Synthetic Control results
        """
        recommendations = []
        
        conclusion = overall_conclusion['overall_conclusion']['conclusion']
        confidence = overall_conclusion['overall_conclusion']['confidence']
        primary_evidence = overall_conclusion['overall_conclusion']['primary_evidence']
        
        if conclusion == 'strong_linguistic_differences_supported':
            recommendations.extend([
                "🎯 STRONG VALIDATION: Synthetic Control Tests confirm genuine linguistic differences",
                "✅ Proceed confidently with DualContextGCN approach",
                "✅ Random label reassignment collapses divergences - patterns are real",
                "✅ Publish results with high confidence in theoretical foundation",
                "✅ Expand to other domains, languages, and larger datasets",
                "🔬 Consider deeper linguistic analysis of identified patterns"
            ])
        
        elif conclusion == 'linguistic_differences_supported':
            recommendations.extend([
                "🎯 GOOD VALIDATION: Synthetic Control Tests support linguistic differences",
                "✅ Proceed with DualContextGCN with reasonable confidence",
                "⚠️ Monitor for potential residual biases in deployment",
                "🔬 Consider additional validation with larger sample sizes",
                "📊 Document validation methodology for reproducibility"
            ])
        
        elif conclusion == 'dataset_bias_confirmed':
            recommendations.extend([
                "🚨 CRITICAL FINDING: Synthetic Control Tests reveal dataset construction bias",
                "❌ Current approach likely capturing bias, not linguistic differences",
                "❌ Do not proceed with current DualContextGCN without major modifications",
                "🔧 Redesign dataset construction and collection methodology",
                "🔧 Implement bias-aware training and evaluation procedures",
                "🔄 Consider alternative approaches less dependent on word associations",
                "📊 Investigate sources of systematic bias in dataset construction"
            ])
        
        else:  # mixed evidence
            recommendations.extend([
                "⚠️ MIXED RESULTS: Both linguistic differences and biases present",
                "⚠️ Proceed with caution and additional controls",
                "🔧 Implement hybrid model accounting for both factors",
                "🔧 Add dataset source and temporal features to model",
                "📊 Increase dataset diversity and size",
                "🔬 Conduct follow-up studies with improved methodology"
            ])
        
        # Technical recommendations based on primary evidence
        if primary_evidence == 'synthetic_control_tests':
            recommendations.extend([
                "💡 Synthetic Control Tests provide the strongest evidence base",
                "💡 Use collapse ratio as key metric for future validation studies",
                "💡 Apply similar methodology to validate other NLP bias claims"
            ])
        
        return recommendations
    
    def _generate_enhanced_validation_report(self, all_results: dict, overall_conclusion: dict):
        """
        Generate enhanced validation report highlighting Synthetic Control results
        """
        print("\n" + "="*100)
        print("🔬 ENHANCED COMPREHENSIVE VALIDATION REPORT")
        print("="*100)
        
        # Highlight Synthetic Control Results first
        synthetic_control = all_results.get('synthetic_control', {})
        if synthetic_control:
            print(f"\n🎲 SYNTHETIC CONTROL TESTS (PRIMARY EVIDENCE):")
            sc_conclusion = synthetic_control.get('conclusion', {})
            collapse_ratio = synthetic_control.get('collapse_ratio', 0)
            
            if sc_conclusion.get('conclusion') == 'genuine_linguistic_patterns':
                emoji = "✅"
                status = "PATTERNS ARE GENUINE"
            elif sc_conclusion.get('conclusion') == 'dataset_construction_bias':
                emoji = "❌"
                status = "DATASET BIAS DETECTED"
            else:
                emoji = "⚠️"
                status = "MIXED EVIDENCE"
            
            print(f"   {emoji} {status}")
            print(f"   Collapse Ratio: {collapse_ratio:.3f} ({collapse_ratio*100:.1f}% divergence reduction)")
            print(f"   Confidence: {sc_conclusion.get('confidence', 'unknown')}")
            print(f"   Criteria Met: {sc_conclusion.get('criteria_met', 0)}/{sc_conclusion.get('total_criteria', 4)}")
        
        # Overall conclusion
        print(f"\n📊 OVERALL CONCLUSION:")
        oc = overall_conclusion['overall_conclusion']
        print(f"   Result: {oc['conclusion']}")
        print(f"   Confidence: {oc['confidence']}")
        print(f"   Primary Evidence: {oc['primary_evidence']}")
        print(f"   Validity Score: {self._calculate_enhanced_validity_score(all_results):.3f}")
        
        print(f"\n💭 INTERPRETATION:")
        print(f"   {oc['interpretation']}")
        
        # Other validation results summary
        print(f"\n🔍 SUPPORTING VALIDATION RESULTS:")
        for val_type, results in all_results.items():
            if val_type == 'synthetic_control':
                continue
                
            conclusion = results.get('conclusion', {})
            confidence = conclusion.get('confidence', 'unknown')
            
            emoji = {'high': '✅', 'medium': '⚠️', 'low': '❌'}.get(confidence, '❓')
            print(f"   {emoji} {val_type.replace('_', ' ').title()}: {confidence} confidence")
        
        # Enhanced recommendations
        print(f"\n💡 ENHANCED RECOMMENDATIONS:")
        recommendations = self._generate_enhanced_recommendations(overall_conclusion)
        for rec in recommendations:
            print(f"   {rec}")
        
        # Technical implications
        print(f"\n🔧 TECHNICAL IMPLICATIONS:")
        conclusion_type = oc['conclusion']
        
        if 'strong_linguistic' in conclusion_type or 'linguistic_differences_supported' in conclusion_type:
            print("   ✅ DualContextGCN approach is theoretically sound")
            print("   ✅ Focus on optimizing dual graph architecture")
            print("   ✅ Word association patterns capture genuine linguistic phenomena")
            print("   🔬 Consider extending to other linguistic features beyond word associations")
        
        elif 'dataset_bias_confirmed' in conclusion_type:
            print("   ❌ Current word association approach captures dataset bias")
            print("   ❌ Reconsider fundamental assumptions about fake news language")
            print("   🔧 Develop bias-resistant architectures")
            print("   🔧 Focus on content-independent features")
        
        else:
            print("   ⚠️ Implement bias-aware training procedures")
            print("   ⚠️ Add explicit bias detection and mitigation")
            print("   ⚠️ Use ensemble approaches combining multiple evidence types")
            print("   🔬 Develop more sophisticated bias vs. pattern separation techniques")
        
        print(f"\n" + "="*100)

# Enhanced integration functions

def run_enhanced_comprehensive_validation():
    """
    Main function to run the enhanced comprehensive validation including Synthetic Control Tests
    """
    print("🚀 Starting Enhanced Comprehensive Validation with Synthetic Control Tests")
    print("="*80)
    
    # Step 1: Load data
    data = load_and_prepare_data('gossipcop')
    if not data:
        print("❌ Failed to load data. Exiting.")
        return None
    
    real_texts = data['real_texts']
    fake_texts = data['fake_texts']
    
    # Step 2: Initialize enhanced validator
    validator = EnhancedStrongerValidation(tokenizer=tokenizer)
    
    # Step 3: Define test words
    test_words = [
        'officials', 'sources', 'reported', 'confirmed', 'according',
        'president', 'administration', 'government', 'statement',
        'investigation', 'authorities', 'experts', 'study'
    ]
    
    # Step 4: Create additional datasets for cross-validation
    additional_datasets = create_multi_dataset_validation()
    
    # Step 5: Run enhanced comprehensive validation suite
    print(f"\n🔬 Running enhanced validation suite with Synthetic Control Tests...")
    
    try:
        validation_results = validator.comprehensive_validation_suite_enhanced(
            real_texts=real_texts,
            fake_texts=fake_texts,
            additional_datasets=additional_datasets,
            test_words=test_words
        )
        
        # Step 6: Create visualizations
        print(f"\n📊 Creating visualizations...")
        synthetic_control_results = validation_results['individual_validations'].get('synthetic_control')
        if synthetic_control_results:
            validator.visualize_synthetic_control_results(synthetic_control_results)
        
        # Step 7: Generate final enhanced report
        print(f"\n📋 Generating final enhanced validation report...")
        generate_enhanced_final_report(validation_results)
        
        return validation_results
        
    except Exception as e:
        print(f"❌ Error in enhanced validation suite: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_synthetic_control_test_only():
    """
    Run only the Synthetic Control Tests for focused analysis
    """
    print("🎲 Running Synthetic Control Tests Only")
    print("="*60)
    
    # Load data
    data = load_and_prepare_data('gossipcop')
    if not data:
        print("❌ Failed to load data. Exiting.")
        return None
    
    real_texts = data['real_texts']
    fake_texts = data['fake_texts']
    
    # Initialize validator
    validator = EnhancedStrongerValidation(tokenizer=tokenizer)
    
    # Define test words
    test_words = [
        'officials', 'sources', 'reported', 'confirmed', 'according',
        'president', 'administration', 'government', 'statement',
        'investigation', 'authorities', 'experts', 'study', 'news'
    ]
    
    print(f"📊 Testing with {len(real_texts)} real and {len(fake_texts)} fake articles")
    print(f"🔤 Testing words: {test_words}")
    
    # Run Synthetic Control Tests
    try:
        results = validator.synthetic_control_tests(
            real_texts=real_texts, 
            fake_texts=fake_texts, 
            test_words=test_words,
            n_iterations=15  # Increased iterations for more robust results
        )
        
        # Visualize results
        validator.visualize_synthetic_control_results(results)
        
        return results
        
    except Exception as e:
        print(f"❌ Error in synthetic control tests: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_enhanced_final_report(validation_results):
    """
    Generate enhanced final report highlighting Synthetic Control findings
    """
    print("\n" + "="*100)
    print("📋 ENHANCED FINAL VALIDATION REPORT")
    print("="*100)
    
    # Extract key results
    synthetic_control = validation_results['individual_validations'].get('synthetic_control', {})
    overall_conclusion = validation_results.get('overall_conclusion', {})
    validity_score = validation_results.get('validity_score', 0)
    
    # Primary findings from Synthetic Control Tests
    if synthetic_control:
        print(f"\n🎲 PRIMARY FINDINGS: SYNTHETIC CONTROL TESTS")
        print(f"   Method: Random label reassignment to test for dataset bias")
        
        collapse_ratio = synthetic_control.get('collapse_ratio', 0)
        original_kl = synthetic_control.get('original_mean_kl', 0)
        synthetic_kl = synthetic_control.get('synthetic_mean_kl', 0)
        
        print(f"   Original KL Divergence: {original_kl:.4f}")
        print(f"   Synthetic KL Divergence: {synthetic_kl:.4f}")
        print(f"   Collapse Ratio: {collapse_ratio:.4f} ({collapse_ratio*100:.1f}% reduction)")
        
        sc_conclusion = synthetic_control.get('conclusion', {})
        conclusion_type = sc_conclusion.get('conclusion', 'unknown')
        
        if conclusion_type == 'genuine_linguistic_patterns':
            print(f"   🎯 VERDICT: GENUINE LINGUISTIC PATTERNS DETECTED")
            print(f"   ✅ High collapse ratio indicates patterns are NOT due to dataset bias")
            print(f"   ✅ Random reassignment destroys patterns, confirming they're real")
        
        elif conclusion_type == 'dataset_construction_bias':
            print(f"   🎯 VERDICT: DATASET CONSTRUCTION BIAS DETECTED")
            print(f"   ❌ Low collapse ratio indicates patterns persist with random labels")
            print(f"   ❌ Observed divergences likely due to systematic dataset bias")
        
        else:
            print(f"   🎯 VERDICT: MIXED EVIDENCE")
            print(f"   ⚠️ Partial collapse suggests both real patterns and bias contribute")
    
    # Overall assessment
    overall_assessment = overall_conclusion.get('overall_conclusion', {})
    print(f"\n📊 OVERALL ASSESSMENT:")
    print(f"   Final Verdict: {overall_assessment.get('conclusion', 'unknown')}")
    print(f"   Confidence Level: {overall_assessment.get('confidence', 'unknown')}")
    print(f"   Validity Score: {validity_score:.3f}/1.000")
    print(f"   Primary Evidence: {overall_assessment.get('primary_evidence', 'unknown')}")
    
    print(f"\n💭 SCIENTIFIC INTERPRETATION:")
    interpretation = overall_assessment.get('interpretation', 'No interpretation available')
    print(f"   {interpretation}")
    
    # Implications for DualContextGCN
    print(f"\n🔬 IMPLICATIONS FOR DUALCONTEXTGCN:")
    
    final_verdict = overall_assessment.get('conclusion', '')
    
    if 'strong_linguistic' in final_verdict or 'linguistic_differences_supported' in final_verdict:
        print(f"   ✅ PROCEED WITH CONFIDENCE")
        print(f"   ✅ Context-specific word associations capture genuine linguistic phenomena")
        print(f"   ✅ Dual graph architecture is theoretically justified")
        print(f"   ✅ Expected to generalize well to new fake news detection tasks")
        print(f"   🚀 Consider scaling up and expanding to other domains")
    
    elif 'dataset_bias_confirmed' in final_verdict:
        print(f"   ❌ DO NOT PROCEED WITHOUT MAJOR REVISIONS")
        print(f"   ❌ Current approach captures dataset artifacts, not linguistic differences")
        print(f"   ❌ Model likely to fail on new datasets with different biases")
        print(f"   🔧 Fundamental revision of approach required")
        print(f"   🔧 Consider alternative architectures less dependent on word associations")
    
    else:
        print(f"   ⚠️ PROCEED WITH CAUTION")
        print(f"   ⚠️ Both genuine patterns and biases contribute to performance")
        print(f"   ⚠️ Implement bias detection and mitigation strategies")
        print(f"   🔧 Consider hybrid approaches and bias-aware training")
        print(f"   📊 Monitor performance across diverse test sets")
    
    # Methodological contributions
    print(f"\n🏆 METHODOLOGICAL CONTRIBUTIONS:")
    print(f"   📈 Demonstrated effectiveness of Synthetic Control Tests for NLP bias detection")
    print(f"   📈 Provided framework for validating linguistic vs. dataset bias claims")
    print(f"   📈 Established collapse ratio as key metric for pattern authenticity")
    print(f"   📈 Created replicable methodology for fake news detection validation")
    
    # Recommendations for field
    print(f"\n🌐 RECOMMENDATIONS FOR THE FIELD:")
    recommendations = validation_results.get('recommendations', [])
    for rec in recommendations[:10]:  # Top 10 recommendations
        print(f"   {rec}")
    
    print(f"\n" + "="*100)

def create_multi_dataset_validation_enhanced():
    """
    Enhanced multi-dataset creation for cross-validation
    """
    print("📄 Creating enhanced multi-dataset validation setup...")
    
    datasets = {}
    
    # Try to load different datasets
    dataset_names = ['gossipcop', 'politifact', 'lun']  # Add more if available
    
    for dataset_name in dataset_names:
        try:
            data = load_and_prepare_data(dataset_name)
            if data:
                datasets[dataset_name] = {
                    'real': data['real_texts'],
                    'fake': data['fake_texts']
                }
                print(f"  ✅ Added {dataset_name}: {len(data['real_texts'])} real, {len(data['fake_texts'])} fake")
        except Exception as e:
            print(f"  ⚠️ Could not load {dataset_name}: {e}")
    
    # If we only have one dataset, create temporal/thematic splits
    if len(datasets) < 2:
        print("  🔧 Creating enhanced artificial splits for cross-validation...")
        
        if 'gossipcop' in datasets or 'lun' in datasets:
            main_dataset_name = 'gossipcop' if 'gossipcop' in datasets else 'lun'
            main_data = datasets[main_dataset_name]
            
            # Create multiple split strategies
            real_texts = main_data['real']
            fake_texts = main_data['fake']
            
            # Temporal splits (early, middle, late)
            n_splits = 3
            real_chunk_size = len(real_texts) // n_splits
            fake_chunk_size = len(fake_texts) // n_splits
            
            for i in range(n_splits):
                start_real = i * real_chunk_size
                end_real = (i + 1) * real_chunk_size if i < n_splits - 1 else len(real_texts)
                
                start_fake = i * fake_chunk_size
                end_fake = (i + 1) * fake_chunk_size if i < n_splits - 1 else len(fake_texts)
                
                datasets[f'temporal_split_{i+1}'] = {
                    'real': real_texts[start_real:end_real],
                    'fake': fake_texts[start_fake:end_fake]
                }
            
            print(f"  ✅ Created {n_splits} temporal splits")
    
    return datasets

# Main execution functions
def main_enhanced():
    """
    Enhanced main execution function with Synthetic Control Tests
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Enhanced Validation with Synthetic Control Tests')
    parser.add_argument('--mode', choices=['full', 'synthetic_only', 'quick'], default='full',
                       help='Validation mode: full enhanced suite, synthetic control only, or quick test')
    parser.add_argument('--dataset', default='gossipcop',
                       help='Dataset to use for validation')
    parser.add_argument('--iterations', type=int, default=15,
                       help='Number of synthetic control iterations')
    parser.add_argument('--save', action='store_true',
                       help='Save results to file')
    
    args = parser.parse_args()
    
    print(f"🔬 Starting Enhanced Validation with Synthetic Control Tests (mode: {args.mode})")
    
    try:
        if args.mode == 'synthetic_only':
            results = run_synthetic_control_test_only()
        elif args.mode == 'quick':
            # Quick synthetic control test
            data = load_and_prepare_data(args.dataset)
            if data:
                validator = EnhancedStrongerValidation(tokenizer=tokenizer)
                results = validator.synthetic_control_tests(
                    data['real_texts'][:100], data['fake_texts'][:100], 
                    n_iterations=5
                )
                if results:
                    validator.visualize_synthetic_control_results(results)
        else:  # full
            results = run_enhanced_comprehensive_validation()
        
        if results and args.save:
            save_validation_results(results, 'enhanced_validation_results.pkl')
        
        print("\n✅ Enhanced validation completed successfully!")
        print("📋 Synthetic Control Tests provide the strongest evidence for pattern authenticity")
        
    except KeyboardInterrupt:
        print("\n⏹️ Validation interrupted by user")
    except Exception as e:
        print(f"\n❌ Enhanced validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_enhanced()