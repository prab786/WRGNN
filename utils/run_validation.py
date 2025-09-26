#!/usr/bin/env python3
"""
Run Stronger Validation for Context-Specific Association Divergence

This script provides a comprehensive validation suite to test whether observed
word association divergences represent genuine linguistic differences or dataset biases.

Usage:
    python run_validation.py --mode full     # Run comprehensive validation
    python run_validation.py --mode quick    # Run quick test
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
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"⚠️  Missing data files: {missing_files}")
        print("🔧 Please ensure data files are in the correct directory structure")
        return False
    
    return True

def setup_environment():
    """Setup the environment for validation."""
    print("🔧 Setting up environment...")
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Check data files
    if not check_data_files():
        print("⚠️  Some data files missing, but continuing with available data...")
    
    # Create output directory
    os.makedirs('validation_outputs', exist_ok=True)
    
    return True

def run_stronger_validation():
    """Run the stronger validation suite."""
    
    # Setup
    if not setup_environment():
        print("❌ Environment setup failed")
        return False
    
    print("🚀 Starting Stronger Validation Suite...")
    print("="*60)
    
    try:
        # Import validation modules
        from validation_integration import run_comprehensive_stronger_validation
        
        # Run validation
        results = run_comprehensive_stronger_validation()
        
        if results:
            print("✅ Validation completed successfully!")
            
            # Save results
            import pickle
            with open('validation_outputs/validation_results.pkl', 'wb') as f:
                pickle.dump(results, f)
            print("💾 Results saved to validation_outputs/validation_results.pkl")
            
            return True
        else:
            print("❌ Validation failed to produce results")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("🔧 Running fallback validation...")
        return run_fallback_validation()
    
    except Exception as e:
        print(f"❌ Validation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_fallback_validation():
    """Run a simplified validation if full validation fails."""
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
        
        # Load data
        print("📁 Loading data...")
        obj = 'lun'
        train_dict = pickle.load(open(f'data/news_articles/{obj}_train.pkl', 'rb'))
        news_real, news_fake = split_news_by_binary_label(train_dict)
        
        # Sample data
        real_sample = news_real[:100]
        fake_sample = news_fake[:100]
        
        print(f"📊 Testing with {len(real_sample)} real and {len(fake_sample)} fake articles")
        
        # Test basic divergence
        test_words =  [
    "breaking",
    "election",
    "shocking",
    "clearly",
    "noted",
    "exclusive",
    "campaign",
    "producer",
    "appears",
    "president",
    "obviously",
    "secret",
    "based",
    "studio",
    "explained",
    "alleged",
    "director",
    "definitely",
    "performance",
    "incredible",
    "celebrity",
    "certainly",
    "apparently",
    "actress",
    "claimed",
    "lies",
    "reportedly",
    "sources",
    "truth",
    "amazing",
    "character",
    "might",
    "cover",
    "absolutely",
    "seems",
    "reported",
    "revealed",
    "actor",
    "announced",
    "house",
    "statement",
    "according",
    "movie",
    "confirmed",
    "added",
    "could",
    "told",
    "said"
]
        results = {}
        
        print("🔍 Testing word divergences...")
        for word in test_words:
            try:
                P_real = compute_word_cooccurrence_distribution(word, real_sample, tokenizer)
                P_fake = compute_word_cooccurrence_distribution(word, fake_sample, tokenizer)
                
                if P_real and P_fake:
                    kl_div = compute_kl_divergence(P_real, P_fake)
                    results[word] = kl_div
                    print(f"  {word}: KL = {kl_div:.4f}")
                
            except Exception as e:
                print(f"  ❌ Error with '{word}': {e}")
        
        # Simple analysis
        if results:
            mean_kl = sum(results.values()) / len(results)
            significant_count = sum(1 for kl in results.values() if kl > 0.1)
            
            print(f"\n📈 Fallback Results:")
            print(f"  Mean KL divergence: {mean_kl:.4f}")
            print(f"  Significant words (>0.1): {significant_count}/{len(results)}")
            print(f"  Significance rate: {significant_count/len(results)*100:.1f}%")
            
            # Simple conclusion
            if significant_count / len(results) > 0.5 and mean_kl > 0.2:
                print(f"  🎯 Preliminary conclusion: Evidence supports differential word associations")
            else:
                print(f"  ⚠️  Preliminary conclusion: Weak evidence for differential associations")
            
            return True
        else:
            print("❌ No results obtained")
            return False
            
    except Exception as e:
        print(f"❌ Fallback validation failed: {e}")
        return False

def run_quick_test():
    """Run a quick test to verify everything works."""
    print("🏃‍♂️ Running Quick Test...")
    
    try:
        from validation_integration import run_quick_validation_test
        result = run_quick_validation_test()
        return result is not None
    except:
        return run_fallback_validation()

def print_validation_help():
    """Print help information about validation."""
    print("""
🔬 STRONGER VALIDATION FOR CONTEXT-SPECIFIC ASSOCIATION DIVERGENCE

This validation suite tests whether observed word association divergences between
real and fake news represent genuine linguistic differences or dataset biases.

VALIDATION METHODS:
1. 📊 Topic Matching: Controls for topical differences between real/fake datasets
2. 🧪 Synthetic Validation: Tests with artificially created fake news
3. 📅 Same-Event Validation: Compares real/fake coverage of identical events
4. 🎭 Style Transfer: Tests effect of style vs content on divergences
5. 🔄 Cross-Dataset: Validates patterns across multiple independent datasets

INTERPRETATION:
- ✅ Strong Support: Evidence for genuine linguistic differences
- ⚠️  Moderate Support: Mixed evidence, some bias concerns
- ❌ Weak Support: Likely dataset biases rather than linguistic differences

USAGE:
    python run_validation.py --help           # Show this help
    python run_validation.py --mode quick     # Quick test (5-10 minutes)
    python run_validation.py --mode full      # Full validation (30-60 minutes)
    
OUTPUT:
- Comprehensive validation report
- Confidence scores and recommendations
- Saved results in validation_outputs/
""")

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run Stronger Validation for Context-Specific Association Divergence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_validation.py --mode quick     # Quick test
  python run_validation.py --mode full      # Complete validation
  python run_validation.py --help           # Show detailed help
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['quick', 'full', 'help'], 
        default='full',
        help='Validation mode to run'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'help':
        print_validation_help()
        return
    
    # Set verbosity
    if not args.verbose:
        import warnings
        warnings.filterwarnings('ignore')
    
    print("🔬 STRONGER VALIDATION SUITE")
    print("="*50)
    
    try:
        if args.mode == 'quick':
            print("🏃‍♂️ Running Quick Validation Test...")
            success = run_quick_test()
        else:  # full
            print("🚀 Running Comprehensive Validation...")
            success = run_stronger_validation()
        
        if success:
            print("\n✅ Validation completed successfully!")
            print("📁 Check validation_outputs/ for detailed results")
        else:
            print("\n❌ Validation encountered issues")
            print("🔧 Try running with --mode quick for basic testing")
        
    except KeyboardInterrupt:
        print("\n⏹️  Validation interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()