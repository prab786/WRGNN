#!/usr/bin/env python3
"""
Enhanced Run Validation for Context-Specific Association Divergence with Synthetic Control Tests

This script provides a comprehensive validation suite including the new Synthetic Control Tests
to distinguish between genuine linguistic patterns and dataset construction bias.

The Synthetic Control Tests work by:
1. Randomly reassigning "real" vs "fake" labels to articles
2. Recalculating KL-divergence of word associations
3. If divergences remain high → dataset construction bias
4. If divergences collapse to near zero → genuine linguistic patterns

Usage:
    python run_validation.py --mode full           # Run full enhanced validation suite
    python run_validation.py --mode synthetic      # Run only Synthetic Control Tests  
    python run_validation.py --mode quick          # Run quick test
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Ensure we can import the required modules
sys.path.append('.')

def check_dependencies():
    """Check if all required dependencies are available."""
    required_modules = [
        'numpy', 'pandas', 'torch', 'matplotlib', 'seaborn',
        'scipy', 'sklearn', 'transformers'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Missing required modules: {missing_modules}")
        print("🔧 Please install them using: pip install " + " ".join(missing_modules))
        return False
    
    return True

def check_data_files():
    """Check if required data files exist."""
    required_files = [
        'data/news_articles/gossipcop_train.pkl',
        'data/news_articles/gossipcop_test.pkl'
    ]
    
    # Also check for alternative datasets
    alternative_files = [
        'data/news_articles/lun_train.pkl',
        'data/news_articles/politifact_train.pkl'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    available_alternatives = []
    for file_path in alternative_files:
        if os.path.exists(file_path):
            available_alternatives.append(file_path)
    
    if missing_files and not available_alternatives:
        print(f"❌ Missing required data files: {missing_files}")
        print("🔧 Please ensure data files are in the correct directory structure")
        return False
    elif missing_files:
        print(f"⚠️ Some required files missing: {missing_files}")
        print(f"✅ Found alternatives: {available_alternatives}")
        
    return True

def setup_environment():
    """Setup the environment for validation."""
    print("🔧 Setting up environment for enhanced validation...")
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Check data files
    if not check_data_files():
        print("⚠️ Some data files missing, but continuing with available data...")
    
    # Create output directory
    os.makedirs('validation_outputs', exist_ok=True)
    
    return True

def run_enhanced_stronger_validation():
    """Run the enhanced stronger validation suite with Synthetic Control Tests."""
    
    # Setup
    if not setup_environment():
        print("❌ Environment setup failed")
        return False
    
    print("🚀 Starting Enhanced Validation Suite with Synthetic Control Tests...")
    print("="*80)
    
    try:
        # Import enhanced validation modules
        from synthetic_control_validation import run_enhanced_comprehensive_validation
        
        # Run enhanced validation
        results = run_enhanced_comprehensive_validation()
        
        if results:
            print("✅ Enhanced validation completed successfully!")
            
            # Save results
            import pickle
            timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'validation_outputs/enhanced_validation_results_{timestamp}.pkl'
            
            with open(filename, 'wb') as f:
                pickle.dump(results, f)
            print(f"💾 Results saved to {filename}")
            
            # Print key findings
            print_key_synthetic_control_findings(results)
            
            return True
        else:
            print("❌ Enhanced validation failed to produce results")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔧 Running fallback validation...")
        return run_fallback_validation()
    
    except Exception as e:
        print(f"❌ Enhanced validation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_synthetic_control_only():
    """Run only the Synthetic Control Tests."""
    print("🎲 Running Synthetic Control Tests Only...")
    print("="*60)
    
    # Setup
    if not setup_environment():
        print("❌ Environment setup failed")
        return False
    
    try:
        from synthetic_control_validation import run_synthetic_control_test_only
        
        results = run_synthetic_control_test_only()
        
        if results:
            print("✅ Synthetic Control Tests completed successfully!")
            
            # Save results
            import pickle
            timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'validation_outputs/synthetic_control_results_{timestamp}.pkl'
            
            with open(filename, 'wb') as f:
                pickle.dump(results, f)
            print(f"💾 Results saved to {filename}")
            
            # Print interpretation
            print_synthetic_control_interpretation(results)
            
            return True
        else:
            print("❌ Synthetic Control Tests failed")
            return False
            
    except Exception as e:
        print(f"❌ Synthetic Control Tests error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_fallback_validation():
    """Run a simplified validation if enhanced validation fails."""
    print("🔄 Running Fallback Validation...")
    
    try:
        # Import your existing functions directly
        from load_data_vocab import (
            split_news_by_binary_label, 
            compute_word_cooccurrence_distribution,
            compute_kl_divergence,
            tokenizer
        )
        import pickle
        import random
        import numpy as np
        
        # Load data
        print("📁 Loading data...")
        
        # Try different datasets
        dataset_options = ['lun', 'gossipcop', 'politifact']
        train_dict = None
        obj = None
        
        for dataset in dataset_options:
            try:
                train_dict = pickle.load(open(f'data/news_articles/{dataset}_train.pkl', 'rb'))
                obj = dataset
                print(f"✅ Successfully loaded {dataset} dataset")
                break
            except FileNotFoundError:
                continue
        
        if train_dict is None:
            print("❌ No datasets found")
            return False
            
        news_real, news_fake = split_news_by_binary_label(train_dict)
        
        # Sample data for efficiency
        real_sample = news_real[:200] if len(news_real) > 200 else news_real
        fake_sample = news_fake[:200] if len(news_fake) > 200 else news_fake
        
        print(f"📊 Testing with {len(real_sample)} real and {len(fake_sample)} fake articles")
        
        # Implement basic Synthetic Control Test
        print("\n🎲 Running Basic Synthetic Control Test...")
        
        test_words = [
            "breaking", "election", "shocking", "clearly", "noted",
            "exclusive", "campaign", "producer", "appears", "president",
            "obviously", "secret", "based", "studio", "explained",
            "alleged", "director", "definitely", "performance", "incredible"
        ]
        
        # Step 1: Calculate original divergences
        print("📈 Step 1: Computing original divergences...")
        original_results = {}
        
        for word in test_words:
            try:
                P_real = compute_word_cooccurrence_distribution(word, real_sample, tokenizer)
                P_fake = compute_word_cooccurrence_distribution(word, fake_sample, tokenizer)
                
                if P_real and P_fake:
                    kl_div = compute_kl_divergence(P_real, P_fake)
                    original_results[word] = kl_div
                    print(f"  {word}: KL = {kl_div:.4f}")
                
            except Exception as e:
                print(f"  ❌ Error with '{word}': {e}")
        
        if not original_results:
            print("❌ No original results obtained")
            return False
        
        original_mean_kl = sum(original_results.values()) / len(original_results)
        print(f"📊 Original Mean KL Divergence: {original_mean_kl:.4f}")
        
        # Step 2: Run synthetic control iterations
        print(f"\n🎲 Step 2: Running synthetic control iterations...")
        
        all_texts = real_sample + fake_sample
        n_real_original = len(real_sample)
        n_iterations = 10
        
        synthetic_kl_values = []
        
        for iteration in range(n_iterations):
            print(f"  Iteration {iteration + 1}/{n_iterations}", end="")
            
            # Randomly shuffle and reassign labels
            shuffled_texts = all_texts.copy()
            random.shuffle(shuffled_texts)
            
            synthetic_real = shuffled_texts[:n_real_original]
            synthetic_fake = shuffled_texts[n_real_original:]
            
            # Calculate synthetic divergences
            synthetic_results = {}
            
            for word in test_words:
                try:
                    P_real = compute_word_cooccurrence_distribution(word, synthetic_real, tokenizer)
                    P_fake = compute_word_cooccurrence_distribution(word, synthetic_fake, tokenizer)
                    
                    if P_real and P_fake:
                        kl_div = compute_kl_divergence(P_real, P_fake)
                        synthetic_results[word] = kl_div
                
                except Exception:
                    continue
            
            if synthetic_results:
                iteration_mean_kl = sum(synthetic_results.values()) / len(synthetic_results)
                synthetic_kl_values.append(iteration_mean_kl)
                print(f" → Mean KL: {iteration_mean_kl:.4f}")
            else:
                print(f" → No results")
        
        # Step 3: Analyze results
        if not synthetic_kl_values:
            print("❌ No synthetic control results")
            return False
        
        synthetic_mean_kl = np.mean(synthetic_kl_values)
        synthetic_std_kl = np.std(synthetic_kl_values)
        collapse_ratio = 1 - (synthetic_mean_kl / original_mean_kl) if original_mean_kl > 0 else 0
        
        print(f"\n📈 Step 3: Synthetic Control Analysis:")
        print(f"  Original Mean KL: {original_mean_kl:.4f}")
        print(f"  Synthetic Mean KL: {synthetic_mean_kl:.4f} (±{synthetic_std_kl:.4f})")
        print(f"  Collapse Ratio: {collapse_ratio:.4f} ({collapse_ratio*100:.1f}% reduction)")
        
        # Step 4: Interpretation
        print(f"\n🎯 Step 4: Interpretation:")
        
        if collapse_ratio > 0.7:
            verdict = "GENUINE LINGUISTIC PATTERNS"
            interpretation = ("High collapse ratio indicates genuine linguistic differences. "
                            "Random label reassignment destroys most patterns, confirming "
                            "they are not due to dataset construction bias.")
            confidence = "High"
            emoji = "✅"
        elif collapse_ratio > 0.4:
            verdict = "MIXED EVIDENCE"
            interpretation = ("Moderate collapse ratio suggests both linguistic differences "
                            "and dataset biases contribute to observed patterns.")
            confidence = "Medium"  
            emoji = "⚠️"
        else:
            verdict = "DATASET CONSTRUCTION BIAS"
            interpretation = ("Low collapse ratio indicates patterns persist even with "
                            "randomized labels, suggesting systematic dataset bias "
                            "rather than genuine linguistic differences.")
            confidence = "High"
            emoji = "❌"
        
        print(f"  {emoji} {verdict}")
        print(f"  Confidence: {confidence}")
        print(f"  {interpretation}")
        
        # Statistical test
        if len(synthetic_kl_values) > 1:
            from scipy import stats
            t_stat, p_value = stats.ttest_1samp(synthetic_kl_values, original_mean_kl)
            print(f"  Statistical significance: p = {p_value:.6f}")
            
            if p_value < 0.01:
                print(f"  ✅ Highly significant difference between original and synthetic")
            elif p_value < 0.05:
                print(f"  ✅ Significant difference between original and synthetic")
            else:
                print(f"  ❌ No significant difference between original and synthetic")
        
        # Summary
        print(f"\n📋 FALLBACK VALIDATION SUMMARY:")
        print(f"  Dataset: {obj}")
        print(f"  Words tested: {len(original_results)}")
        print(f"  Synthetic iterations: {len(synthetic_kl_values)}")
        print(f"  Final verdict: {verdict}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_key_synthetic_control_findings(results):
    """Print key findings from Synthetic Control Tests."""
    
    synthetic_control = results.get('individual_validations', {}).get('synthetic_control', {})
    
    if not synthetic_control:
        print("⚠️ No Synthetic Control results found")
        return
    
    print("\n" + "="*60)
    print("🎲 KEY SYNTHETIC CONTROL FINDINGS")
    print("="*60)
    
    collapse_ratio = synthetic_control.get('collapse_ratio', 0)
    original_kl = synthetic_control.get('original_mean_kl', 0)
    synthetic_kl = synthetic_control.get('synthetic_mean_kl', 0)
    
    print(f"📊 Quantitative Results:")
    print(f"   Original Divergence: {original_kl:.4f}")
    print(f"   Synthetic Divergence: {synthetic_kl:.4f}")
    print(f"   Collapse Ratio: {collapse_ratio:.4f} ({collapse_ratio*100:.1f}% reduction)")
    
    conclusion = synthetic_control.get('conclusion', {})
    verdict = conclusion.get('conclusion', 'unknown')
    confidence = conclusion.get('confidence', 'unknown')
    
    print(f"\n🎯 Scientific Conclusion:")
    
    if verdict == 'genuine_linguistic_patterns':
        print(f"   ✅ VERDICT: Genuine linguistic patterns detected")
        print(f"   ✅ High collapse ratio confirms patterns are NOT dataset bias")
        print(f"   ✅ DualContextGCN approach is theoretically sound")
    elif verdict == 'dataset_construction_bias':
        print(f"   ❌ VERDICT: Dataset construction bias detected") 
        print(f"   ❌ Low collapse ratio indicates systematic bias")
        print(f"   ❌ Current approach may not generalize")
    else:
        print(f"   ⚠️ VERDICT: Mixed evidence")
        print(f"   ⚠️ Both patterns and biases contribute")
        
    print(f"   Confidence: {confidence}")
    
    interpretation = conclusion.get('interpretation', '')
    if interpretation:
        print(f"\n💭 Interpretation:")
        print(f"   {interpretation}")

def print_synthetic_control_interpretation(results):
    """Print interpretation of Synthetic Control Test results."""
    
    if not results:
        print("❌ No results to interpret")
        return
    
    print("\n" + "="*50)
    print("🎲 SYNTHETIC CONTROL INTERPRETATION")
    print("="*50)
    
    collapse_ratio = results.get('collapse_ratio', 0)
    conclusion = results.get('conclusion', {})
    
    print(f"🔍 What Synthetic Control Tests Tell Us:")
    print(f"   Method: Randomly reassign 'real' vs 'fake' labels")
    print(f"   Logic: If patterns persist → dataset bias")
    print(f"           If patterns collapse → genuine linguistic differences")
    
    print(f"\n📊 Your Results:")
    print(f"   Collapse Ratio: {collapse_ratio:.3f} ({collapse_ratio*100:.1f}% reduction)")
    
    verdict = conclusion.get('conclusion', 'unknown')
    
    if verdict == 'genuine_linguistic_patterns':
        print(f"\n✅ INTERPRETATION: GENUINE LINGUISTIC PATTERNS")
        print(f"   → Random reassignment destroys most patterns")
        print(f"   → Original divergences capture real linguistic differences")
        print(f"   → NOT caused by how dataset was constructed")
        print(f"   → Context-specific word associations are authentic")
        
        print(f"\n🚀 IMPLICATIONS FOR YOUR RESEARCH:")
        print(f"   ✅ Proceed confidently with DualContextGCN")
        print(f"   ✅ Patterns likely to generalize to new data")
        print(f"   ✅ Strong theoretical foundation confirmed")
        
    elif verdict == 'dataset_construction_bias':
        print(f"\n❌ INTERPRETATION: DATASET CONSTRUCTION BIAS")
        print(f"   → Patterns persist even with random labels")
        print(f"   → Divergences caused by how dataset was built")
        print(f"   → NOT capturing genuine linguistic differences")
        print(f"   → Systematic bias in data collection/labeling")
        
        print(f"\n⚠️ IMPLICATIONS FOR YOUR RESEARCH:")
        print(f"   ❌ Current approach captures bias, not language")
        print(f"   ❌ May not work on differently constructed datasets")
        print(f"   🔧 Need to revise approach or dataset construction")
        
    else:
        print(f"\n⚠️ INTERPRETATION: MIXED EVIDENCE")
        print(f"   → Some patterns collapse, some persist")
        print(f"   → Both genuine differences AND bias contribute")
        print(f"   → Complex interaction of factors")
        
        print(f"\n🔧 IMPLICATIONS FOR YOUR RESEARCH:")
        print(f"   ⚠️ Proceed with caution")
        print(f"   ⚠️ Implement bias-aware training")
        print(f"   🔧 Consider hybrid approaches")

def run_quick_test_enhanced():
    """Run a quick enhanced test with basic Synthetic Control."""
    print("🏃‍♂️ Running Quick Enhanced Test...")
    
    try:
        # Use fallback validation which includes basic synthetic control
        return run_fallback_validation()
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def print_validation_help_enhanced():
    """Print enhanced help information about validation."""
    print("""
🔬 ENHANCED VALIDATION FOR CONTEXT-SPECIFIC ASSOCIATION DIVERGENCE

This enhanced validation suite includes the new SYNTHETIC CONTROL TESTS which provide
the strongest evidence for distinguishing genuine linguistic patterns from dataset bias.

🎲 SYNTHETIC CONTROL TESTS (NEW & MOST IMPORTANT):
Method: Randomly reassign "real" vs "fake" labels to articles and recalculate divergences
Logic: 
- If divergences remain high → dataset construction bias  
- If divergences collapse to near zero → genuine context-specific patterns

Key Metric: COLLAPSE RATIO (% reduction in divergences)
- >70% collapse = Strong evidence for genuine patterns
- 40-70% collapse = Mixed evidence  
- <40% collapse = Strong evidence for dataset bias

OTHER VALIDATION METHODS:
1. 📊 Topic Matching: Controls for topical differences between real/fake datasets
2. 🧪 Synthetic Validation: Tests with artificially created fake news
3. 📅 Same-Event Validation: Compares real/fake coverage of identical events
4. 🎭 Style Transfer: Tests effect of style vs content on divergences
5. 📄 Cross-Dataset: Validates patterns across multiple independent datasets

INTERPRETATION:
- ✅ Strong Support: Evidence for genuine linguistic differences
- ⚠️ Moderate Support: Mixed evidence, some bias concerns  
- ❌ Weak Support: Likely dataset biases rather than linguistic differences

ENHANCED USAGE:
    python run_validation.py --help              # Show this help
    python run_validation.py --mode synthetic    # Run only Synthetic Control Tests
    python run_validation.py --mode full         # Full enhanced validation suite  
    python run_validation.py --mode quick        # Quick test with basic synthetic control
    
OUTPUT:
- Comprehensive validation report with Synthetic Control findings
- Collapse ratio analysis and confidence scores
- Clear recommendations for proceeding with research
- Saved results in validation_outputs/

🎯 SCIENTIFIC IMPACT:
The Synthetic Control Tests provide the most direct test of whether your Context-Specific 
Association Divergence captures genuine linguistic phenomena or dataset construction artifacts.
This is crucial for establishing the validity of your DualContextGCN approach.
""")

def main():
    """Enhanced main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run Enhanced Validation with Synthetic Control Tests for Context-Specific Association Divergence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_validation.py --mode synthetic     # Run Synthetic Control Tests only
  python run_validation.py --mode full          # Complete enhanced validation
  python run_validation.py --mode quick         # Quick test with basic synthetic control
  python run_validation.py --help              # Show detailed help
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['synthetic', 'full', 'quick', 'help'], 
        default='synthetic',
        help='Validation mode to run'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        default=15,
        help='Number of synthetic control iterations (default: 15)'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'help':
        print_validation_help_enhanced()
        return
    
    # Set verbosity
    if not args.verbose:
        import warnings
        warnings.filterwarnings('ignore')
    
    print("🔬 ENHANCED VALIDATION SUITE WITH SYNTHETIC CONTROL TESTS")
    print("="*70)
    print("🎲 NEW: Synthetic Control Tests - The gold standard for bias detection")
    print("="*70)
    
    try:
        if args.mode == 'synthetic':
            print("🎲 Running Synthetic Control Tests Only...")
            success = run_synthetic_control_only()
        elif args.mode == 'quick':
            print("🏃‍♂️ Running Quick Enhanced Validation...")
            success = run_quick_test_enhanced()
        else:  # full
            print("🚀 Running Full Enhanced Validation Suite...")
            success = run_enhanced_stronger_validation()
        
        if success:
            print("\n✅ Enhanced validation completed successfully!")
            print("📁 Check validation_outputs/ for detailed results")
            print("\n🎯 KEY MESSAGE:")
            print("The Synthetic Control Tests provide the strongest evidence")
            print("for whether your patterns are genuine or dataset bias.")
        else:
            print("\n❌ Enhanced validation encountered issues")
            print("🔧 Try running with --mode quick for basic testing")
        
    except KeyboardInterrupt:
        print("\n⏹️ Validation interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()