# validation_integration.py
# Integration script to run stronger validation with your existing code

import sys
import os
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
    import pickle
    
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

def run_baseline_divergence_test(real_texts, fake_texts, test_words=None):
    """
    Run baseline divergence test using your existing functions.
    """
    print("🔍 Running baseline divergence test...")
    
    if test_words is None:
        test_words = [
            'officials', 'sources', 'reported', 'confirmed', 'according',
            'president', 'administration', 'government', 'statement',
            'investigation', 'authorities', 'experts', 'study', 'news'
        ]
    
    baseline_results = {}
    
    for word in test_words:
        try:
            # Use your existing functions
            P_real = compute_word_cooccurrence_distribution(word, real_texts, tokenizer)
            P_fake = compute_word_cooccurrence_distribution(word, fake_texts, tokenizer)
            
            if P_real and P_fake:
                kl_div = compute_kl_divergence(P_real, P_fake)
                baseline_results[word] = {
                    'kl_divergence': kl_div,
                    'real_vocab_size': len(P_real),
                    'fake_vocab_size': len(P_fake),
                    'threshold_exceeded': kl_div > 0.1
                }
                print(f"  {word}: KL = {kl_div:.4f}")
            
        except Exception as e:
            print(f"  ❌ Error processing '{word}': {e}")
            continue
    
    # Summary statistics
    kl_values = [result['kl_divergence'] for result in baseline_results.values()]
    significant_count = sum(1 for result in baseline_results.values() if result['threshold_exceeded'])
    
    print(f"\n📊 Baseline Results Summary:")
    print(f"  Words tested: {len(baseline_results)}")
    print(f"  Mean KL divergence: {sum(kl_values)/len(kl_values):.4f}")
    print(f"  Words exceeding threshold (0.1): {significant_count}/{len(baseline_results)}")
    print(f"  Percentage significant: {significant_count/len(baseline_results)*100:.1f}%")
    
    return baseline_results

def create_multi_dataset_validation():
    """
    Create multiple datasets for cross-validation.
    """
    print("🔄 Creating multi-dataset validation setup...")
    
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

def run_comprehensive_stronger_validation():
    """
    Main function to run the comprehensive stronger validation.
    """
    print("🚀 Starting Comprehensive Stronger Validation")
    print("="*60)
    
    # Step 1: Load data
    data = load_and_prepare_data('gossipcop')
    if not data:
        print("❌ Failed to load data. Exiting.")
        return None
    
    real_texts = data['real_texts']
    fake_texts = data['fake_texts']
    
    # Step 2: Run baseline test
    test_words = [
        'officials', 'sources', 'reported', 'confirmed', 'according',
        'president', 'administration', 'government', 'statement',
        'investigation', 'authorities', 'experts', 'study'
    ]
    
    baseline_results = run_baseline_divergence_test(real_texts, fake_texts, test_words)
    
    # Step 3: Initialize stronger validation
    validator = StrongerValidation(tokenizer=tokenizer)
    
    # Step 4: Create additional datasets for cross-validation
    additional_datasets = create_multi_dataset_validation()
    
    # Step 5: Run comprehensive validation suite
    print(f"\n🔬 Running comprehensive validation suite...")
    
    try:
        validation_results = validator.comprehensive_validation_suite(
            real_texts=real_texts,
            fake_texts=fake_texts,
            additional_datasets=additional_datasets,
            test_words=test_words
        )
        
        # Step 6: Analyze and compare results
        print(f"\n📈 Comparing baseline vs validation results...")
        comparison = compare_baseline_vs_validation(baseline_results, validation_results)
        
        # Step 7: Generate final report
        generate_final_validation_report(baseline_results, validation_results, comparison)
        
        return {
            'baseline_results': baseline_results,
            'validation_results': validation_results,
            'comparison': comparison
        }
        
    except Exception as e:
        print(f"❌ Error in validation suite: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_baseline_vs_validation(baseline_results, validation_results):
    """
    Compare baseline divergence results with validation results.
    """
    print("🔍 Analyzing baseline vs validation consistency...")
    
    comparison = {
        'baseline_summary': {
            'total_words': len(baseline_results),
            'mean_kl': sum(r['kl_divergence'] for r in baseline_results.values()) / len(baseline_results),
            'significant_words': sum(1 for r in baseline_results.values() if r['threshold_exceeded'])
        },
        'validation_summary': validation_results.get('overall_conclusion', {}),
        'consistency_analysis': {}
    }
    
    # Check if baseline "significant" words remain significant across validations
    baseline_significant_words = [
        word for word, result in baseline_results.items() 
        if result['threshold_exceeded']
    ]
    
    # Analyze validation conclusions
    individual_validations = validation_results.get('individual_validations', {})
    validation_support_count = 0
    
    for val_type, val_results in individual_validations.items():
        conclusion = val_results.get('conclusion', {})
        if any(keyword in conclusion.get('conclusion', '') for keyword in 
              ['robust', 'persist', 'replicate', 'supported']):
            validation_support_count += 1
    
    comparison['consistency_analysis'] = {
        'baseline_significant_words': baseline_significant_words,
        'baseline_significance_rate': len(baseline_significant_words) / len(baseline_results),
        'validation_support_rate': validation_support_count / len(individual_validations),
        'overall_consistency': 'high' if validation_support_count >= len(individual_validations) * 0.6 else 'low'
    }
    
    return comparison

def generate_final_validation_report(baseline_results, validation_results, comparison):
    """
    Generate comprehensive final report.
    """
    print("\n" + "="*80)
    print("📋 FINAL VALIDATION REPORT")
    print("="*80)
    
    # Baseline summary
    baseline_summary = comparison['baseline_summary']
    print(f"\n🔢 BASELINE DIVERGENCE ANALYSIS:")
    print(f"  Words tested: {baseline_summary['total_words']}")
    print(f"  Mean KL divergence: {baseline_summary['mean_kl']:.4f}")
    print(f"  Significant words (>0.1): {baseline_summary['significant_words']}")
    print(f"  Significance rate: {baseline_summary['significant_words']/baseline_summary['total_words']*100:.1f}%")
    
    # Validation summary
    overall_conclusion = validation_results.get('overall_conclusion', {})
    validity_score = validation_results.get('validity_score', 0)
    
    print(f"\n🔬 STRONGER VALIDATION ANALYSIS:")
    print(f"  Overall conclusion: {overall_conclusion.get('overall_conclusion', {}).get('conclusion', 'Unknown')}")
    print(f"  Confidence level: {overall_conclusion.get('overall_conclusion', {}).get('confidence', 'Unknown')}")
    print(f"  Validity score: {validity_score:.3f}")
    print(f"  Supporting evidence: {overall_conclusion.get('supporting_evidence', 0)}")
    print(f"  Contradicting evidence: {overall_conclusion.get('contradicting_evidence', 0)}")
    
    # Consistency analysis
    consistency = comparison['consistency_analysis']
    print(f"\n🎯 CONSISTENCY ANALYSIS:")
    print(f"  Baseline significance rate: {consistency['baseline_significance_rate']*100:.1f}%")
    print(f"  Validation support rate: {consistency['validation_support_rate']*100:.1f}%")
    print(f"  Overall consistency: {consistency['overall_consistency']}")
    
    # Final verdict
    print(f"\n⚖️  FINAL VERDICT:")
    
    if (consistency['baseline_significance_rate'] > 0.5 and 
        consistency['validation_support_rate'] > 0.6 and 
        validity_score > 0.6):
        verdict = "STRONG SUPPORT"
        emoji = "✅"
        interpretation = "Strong evidence for genuine linguistic differences between real and fake news"
    elif (consistency['baseline_significance_rate'] > 0.3 and 
          consistency['validation_support_rate'] > 0.4 and 
          validity_score > 0.4):
        verdict = "MODERATE SUPPORT"
        emoji = "⚠️"
        interpretation = "Moderate evidence with some concerns about dataset biases"
    else:
        verdict = "WEAK SUPPORT"
        emoji = "❌"
        interpretation = "Evidence suggests dataset biases may explain observed divergences"
    
    print(f"  {emoji} {verdict}")
    print(f"  Interpretation: {interpretation}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    recommendations = validation_results.get('recommendations', [])
    for rec in recommendations[:8]:  # Show top 8 recommendations
        print(f"  {rec}")
    
    # Technical implications
    print(f"\n🔧 TECHNICAL IMPLICATIONS:")
    if verdict == "STRONG SUPPORT":
        print("  ✅ Proceed with DualContextGCN approach with confidence")
        print("  ✅ Consider expanding to other domains and languages")
        print("  ✅ Focus on optimizing dual graph architecture")
    elif verdict == "MODERATE SUPPORT":
        print("  ⚠️ Implement bias-aware training procedures")
        print("  ⚠️ Add dataset source and temporal features")
        print("  ⚠️ Consider hybrid approaches")
    else:
        print("  ❌ Reconsider theoretical assumptions")
        print("  ❌ Focus on alternative approaches")
        print("  ❌ Improve dataset construction and diversity")
    
    print("\n" + "="*80)

def run_quick_validation_test():
    """
    Run a quick validation test for debugging and testing.
    """
    print("🏃‍♂️ Running Quick Validation Test...")
    
    # Load minimal data
    data = load_and_prepare_data('gossipcop')
    if not data:
        print("❌ No data available for testing")
        return
    
    # Use small sample
    real_sample = data['real_texts'][:50]
    fake_sample = data['fake_texts'][:50]
    test_words = ['officials', 'sources', 'reported']
    
    print(f"📊 Testing with {len(real_sample)} real, {len(fake_sample)} fake articles")
    print(f"🔤 Testing words: {test_words}")
    
    # Initialize validator
    validator = StrongerValidation(tokenizer=tokenizer)
    
    # Run single validation
    print("\n🎯 Running Topic Matching Validation...")
    topic_result = validator.topic_matching_validation(real_sample, fake_sample, test_words)
    
    print(f"✅ Topic validation completed")
    print(f"Conclusion: {topic_result.get('conclusion', {}).get('conclusion', 'Unknown')}")
    print(f"Confidence: {topic_result.get('conclusion', {}).get('confidence', 'Unknown')}")
    
    return topic_result

def save_validation_results(results, filename='stronger_validation_results.pkl'):
    """
    Save validation results for later analysis.
    """
    import pickle
    
    try:
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"💾 Results saved to {filename}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def main():
    """
    Main execution function.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Stronger Validation for Context-Specific Divergence')
    parser.add_argument('--mode', choices=['full', 'quick'], default='full',
                       help='Validation mode: full or quick test')
    parser.add_argument('--dataset', default='gossipcop',
                       help='Dataset to use for validation')
    parser.add_argument('--save', action='store_true',
                       help='Save results to file')
    
    args = parser.parse_args()
    
    print(f"🔬 Starting Stronger Validation (mode: {args.mode})")
    
    try:
        if args.mode == 'quick':
            results = run_quick_validation_test()
        else:
            results = run_comprehensive_stronger_validation()
        
        if results and args.save:
            save_validation_results(results)
        
        print("\n✅ Validation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Validation interrupted by user")
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()