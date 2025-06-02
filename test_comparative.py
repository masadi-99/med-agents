"""
Test script for Enhanced Comparative Reasoning Framework
"""

import dspy
from config import OPENAI_API_KEY
from medical_reasoning import (
    EnhancedComparativeReasoningSolver,
    ComparativeReasoningSolver,  # Keep original for comparison
    visualize_enhanced_option_trees,
    visualize_level_divergences,
    visualize_structured_resolutions,
    visualize_enhanced_comparative_summary,
    # Legacy visualization functions for comparison
    visualize_option_trees,
    visualize_claim_matches,
    visualize_conflicts,
    visualize_conflict_resolutions,
    visualize_comparative_summary
)

def test_enhanced_comparative_reasoning():
    """Test the enhanced comparative reasoning solver."""
    
    # Configure DSPy with OpenAI
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        cache=False,
        temperature=0.1
    )
    dspy.configure(lm=lm)
    
    # Create enhanced comparative solver instance
    enhanced_solver = EnhancedComparativeReasoningSolver()
    
    # Test question - Pregnant woman with ASD
    question = """A 22-year-old woman from a rural area who recently discovered she was pregnant is referred for a cardiology consultation due to cyanosis, dyspnea, and a cardiac murmur revealed at the initial prenatal visit. She is gravida 1, para 0 with an estimated gestational age of 19 weeks. She says that the murmur was found in her childhood, and the doctor at that time placed her under observation only. However, she has been lost to follow-up and has not had proper follow up in years. Currently, she complains of dizziness and occasional dyspnea on exertion which has gradually increased during her pregnancy. Prior to her pregnancy, she did not have any symptoms. The vital signs are as follows: blood pressure 125/60 mm Hg, heart rate 81/min, respiratory rate 13/min, and temperature 36.7°C (98.0°F). Her examination is significant for acrocyanosis and a fixed splitting of S2 and grade 3/6 midsystolic murmur best heard over the left upper sternal border. Which of the following physiological pregnancy changes is causing the change in this patient's condition?"""
    
    options = {
        'A': 'Increase in heart rate',
        'B': 'Decrease in systemic vascular resistance',
        'C': 'Increase in cardiac output',
        'D': 'Increase in blood volume'
    }
    
    print("Enhanced Comparative Reasoning Framework Test")
    print("=" * 80)
    print(f"Question: {question[:200]}...\n")
    
    print("Options:")
    for key, value in options.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # Get enhanced comparative analysis
        enhanced_result = enhanced_solver(question=question, options=options)
        
        # Enhanced Visualizations
        print("\n" + "="*80)
        print("ENHANCED COMPARATIVE REASONING RESULTS")
        print("="*80)
        
        visualize_enhanced_comparative_summary(enhanced_result)
        
        visualize_enhanced_option_trees(enhanced_result['option_trees'], enhanced_result['option_analyses'])
        
        visualize_level_divergences(enhanced_result['level_divergences'])
        
        visualize_structured_resolutions(enhanced_result['divergence_resolutions'], enhanced_result['level_scores'])
        
        return enhanced_result
        
    except Exception as e:
        print(f"❌ Error during enhanced comparative reasoning: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def compare_frameworks():
    """Compare the original and enhanced comparative frameworks."""
    
    # Configure DSPy
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        cache=False,
        temperature=0.1
    )
    dspy.configure(lm=lm)
    
    # Test question
    question = """A 22-year-old woman from a rural area who recently discovered she was pregnant is referred for a cardiology consultation due to cyanosis, dyspnea, and a cardiac murmur revealed at the initial prenatal visit. She is gravida 1, para 0 with an estimated gestational age of 19 weeks. She says that the murmur was found in her childhood, and the doctor at that time placed her under observation only. However, she has been lost to follow-up and has not had proper follow up in years. Currently, she complains of dizziness and occasional dyspnea on exertion which has gradually increased during her pregnancy. Prior to her pregnancy, she did not have any symptoms. The vital signs are as follows: blood pressure 125/60 mm Hg, heart rate 81/min, respiratory rate 13/min, and temperature 36.7°C (98.0°F). Her examination is significant for acrocyanosis and a fixed splitting of S2 and grade 3/6 midsystolic murmur best heard over the left upper sternal border. Which of the following physiological pregnancy changes is causing the change in this patient's condition?"""
    
    options = {
        'A': 'Increase in heart rate',
        'B': 'Decrease in systemic vascular resistance',
        'C': 'Increase in cardiac output',
        'D': 'Increase in blood volume'
    }
    
    print("Framework Comparison Test")
    print("=" * 80)
    print(f"Question: {question[:150]}...")
    print(f"Options: {list(options.keys())}")
    print("\n" + "="*80)
    
    try:
        # Test original framework
        print("🔄 Testing Original Comparative Framework...")
        original_solver = ComparativeReasoningSolver()
        original_result = original_solver(question=question, options=options)
        
        print("\n📊 ORIGINAL FRAMEWORK RESULTS:")
        print("-" * 50)
        print(f"Answer: {original_result['answer']}")
        print(f"Confidence: {original_result['confidence']:.2f}")
        print(f"Method: {original_result['reasoning_method']}")
        print(f"Conflicts Found: {len(original_result.get('conflicts', []))}")
        
        # Test enhanced framework
        print(f"\n🔄 Testing Enhanced Comparative Framework...")
        enhanced_solver = EnhancedComparativeReasoningSolver()
        enhanced_result = enhanced_solver(question=question, options=options)
        
        print("\n📊 ENHANCED FRAMEWORK RESULTS:")
        print("-" * 50)
        print(f"Answer: {enhanced_result['answer']}")
        print(f"Confidence: {enhanced_result['confidence']:.2f}")
        print(f"Method: {enhanced_result['reasoning_method']}")
        print(f"Level Divergences: {sum(len(divs) for divs in enhanced_result['level_divergences'].values())}")
        print(f"Pairwise Comparisons: {len(enhanced_result['claim_comparisons'])}")
        
        # Comparison summary
        print(f"\n🔍 FRAMEWORK COMPARISON:")
        print("-" * 50)
        print(f"Both frameworks selected: {original_result['answer'] == enhanced_result['answer']}")
        print(f"Confidence difference: {abs(original_result['confidence'] - enhanced_result['confidence']):.2f}")
        
        # Feature comparison
        print(f"\n🆚 FEATURE COMPARISON:")
        print("-" * 50)
        print("Original Framework:")
        print("  ✓ Global claim matching")
        print("  ✓ Conflict identification")
        print("  ✓ Binary conflict resolution")
        
        print("\nEnhanced Framework:")
        print("  ✓ Pairwise claim matching")
        print("  ✓ Level-based divergence analysis")
        print("  ✓ Structured divergence judgment")
        print("  ✓ Level-weighted scoring")
        print("  ✓ Enhanced visualization")
        
        return {
            'original': original_result,
            'enhanced': enhanced_result,
            'comparison': {
                'same_answer': original_result['answer'] == enhanced_result['answer'],
                'confidence_diff': abs(original_result['confidence'] - enhanced_result['confidence'])
            }
        }
        
    except Exception as e:
        print(f"❌ Error during framework comparison: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def demo_enhanced_features():
    """Demonstrate the specific enhanced features."""
    
    print("\n🎯 ENHANCED COMPARATIVE REASONING FEATURES DEMO")
    print("=" * 70)
    
    print("\n🔍 Key Enhancements from Alternative Model:")
    print("-" * 50)
    print("1. ✨ Pairwise Claim Matching")
    print("   • More precise claim-to-claim comparison")
    print("   • Similarity classification (IDENTICAL/SIMILAR/RELATED/CONFLICTING/UNRELATED)")
    print("   • Level-aware matching (only compare same hierarchy levels)")
    
    print("\n2. 📊 Level-Based Divergence Analysis")
    print("   • Structured analysis by reasoning hierarchy")
    print("   • Level-specific divergence categorization")
    print("   • Better understanding of where reasoning diverges")
    
    print("\n3. ⚖️ Structured Divergence Judgment")
    print("   • Level-weighted scoring system")
    print("   • Explicit divergence impact assessment")
    print("   • More nuanced conflict resolution")
    
    print("\n4. 🎨 Enhanced Visualization")
    print("   • Level-based tree display with status icons")
    print("   • Hierarchical divergence organization")
    print("   • Level score breakdown")
    print("   • Structured resolution analysis")
    
    print("\n5. 🏗️ Better Data Structures")
    print("   • ClaimComparison and DivergencePoint dataclasses")
    print("   • ClaimSimilarity enum for precise categorization")
    print("   • Level-aware divergence tracking")
    
    print("\n🆚 Improvements Over Original:")
    print("-" * 50)
    print("• More systematic claim comparison")
    print("• Better hierarchy level consideration")
    print("• Enhanced conflict categorization")
    print("• Cleaner data organization")
    print("• More detailed analysis output")

if __name__ == "__main__":
    print("Testing Enhanced Comparative Reasoning Framework")
    print("=" * 80)
    
    # Demo the enhanced features
    demo_enhanced_features()
    
    # Test the enhanced framework
    print(f"\n🧪 Running Enhanced Framework Test...")
    enhanced_result = test_enhanced_comparative_reasoning()
    
    # Compare frameworks if both work
    if enhanced_result:
        print(f"\n🔬 Running Framework Comparison...")
        comparison = compare_frameworks()
        
        if comparison:
            print(f"\n✅ Both frameworks completed successfully!")
            print(f"Check results above for detailed analysis.")
    
    print(f"\n🎉 Testing complete!") 