"""
Test script for Optimized Comparative Reasoning Framework v3.0
"""

import dspy
from config import OPENAI_API_KEY
from medical_reasoning import (
    OptimizedComparativeReasoningSolver,
    visualize_enhanced_option_trees,
    visualize_claim_comparisons,
    visualize_level_divergences,
    visualize_divergence_details,
    visualize_structured_resolutions,
    visualize_enhanced_comparative_summary,
    visualize_complete_analysis,
    visualize_optimization_stats,
    visualize_batch_relationships,
    visualize_optimized_comparative_summary
)

def test_optimized_comparative_reasoning(detailed=True):
    """Test the optimized comparative reasoning solver with beautiful visualizations."""
    
    # Configure DSPy with OpenAI
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        cache=False,
        temperature=0.1
    )
    dspy.configure(lm=lm)
    
    # Create optimized comparative solver instance
    solver = OptimizedComparativeReasoningSolver()
    
    # Test question: Heart failure with atrial septal defect
    question = """
    A 35-year-old patient with a known large atrial septal defect (ASD) presents with increasing shortness of breath and fatigue over the past 6 months. Physical examination reveals elevated jugular venous pressure, a systolic murmur, and mild peripheral edema. Echocardiography shows dilated right heart chambers with preserved left ventricular function. What is the most likely acute physiological change responsible for the patient's recent clinical deterioration?
    """
    
    options = {
        "A": "Increase in heart rate",
        "B": "Increase in systemic vascular resistance", 
        "C": "Increase in cardiac output",
        "D": "Increase in blood volume",
        "E": "Increase in myocardial contractility"
    }
    
    try:
        # Get optimized comparative analysis
        result = solver(question=question, options=options)
        
        if detailed:
            # Complete analysis with all visualizations
            visualize_complete_analysis(result)
        else:
            # Standard visualizations
            print("\n" + "="*80)
            print("OPTIMIZED COMPARATIVE REASONING ANALYSIS")
            print("="*80)
            
            visualize_optimized_comparative_summary(result)
            visualize_enhanced_option_trees(result['option_trees'], result['option_analyses'])
            visualize_batch_relationships(result.get('level_relationships', {}))
            visualize_level_divergences(result['level_divergences'])
            
            # Show optimization stats
            visualize_optimization_stats(result)
        
        return result
        
    except Exception as e:
        print(f"❌ Error during optimized comparative reasoning: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_efficiency_comparison():
    """Show efficiency metrics of the optimized framework."""
    print("⚡ EFFICIENCY ANALYSIS TEST")
    print("=" * 60)
    
    # Configure DSPy
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        cache=False,
        temperature=0.1
    )
    dspy.configure(lm=lm)
    
    # Simple test question
    question = "A patient presents with chest pain and elevated troponin. What is the most likely diagnosis?"
    options = {
        "A": "Myocardial infarction",
        "B": "Pulmonary embolism",
        "C": "Aortic dissection"
    }
    
    print(f"\n🧪 Testing with {len(options)} options...")
    
    try:
        # Test optimized solver
        print("\n🚀 Testing Optimized Solver...")
        optimized_solver = OptimizedComparativeReasoningSolver()
        optimized_result = optimized_solver(question=question, options=options)
        
        # Show optimization metrics
        print("\n📊 EFFICIENCY ANALYSIS:")
        print("─" * 50)
        
        opt_stats = optimized_result.get('optimization_stats', {})
        opt_claims = sum(len(claims) for claims in optimized_result['option_trees'].values())
        
        print(f"🚀 Optimized Framework Results:")
        print(f"   Method: {optimized_result['reasoning_method']}")
        print(f"   Answer: {optimized_result['answer']} (Confidence: {optimized_result['confidence']:.2f})")
        print(f"   Claims Generated: {opt_claims}")
        print(f"   Call Reduction: {opt_stats.get('estimated_call_reduction', 'N/A')}")
        print(f"   Levels Analyzed: {opt_stats.get('levels_analyzed', 'N/A')}")
        print(f"   Batch Operations: {opt_stats.get('batch_operations_count', 'N/A')}")
        
        # Calculate theoretical efficiency metrics
        num_options = len(options)
        estimated_old_calls = num_options * (num_options - 1) // 2 * 5  # Theoretical pairwise across 5 levels
        estimated_new_calls = 2 + opt_stats.get('levels_analyzed', 0)
        efficiency = ((estimated_old_calls - estimated_new_calls) / estimated_old_calls * 100) if estimated_old_calls > 0 else 0
        
        print(f"\n📈 THEORETICAL EFFICIENCY METRICS:")
        print(f"   Estimated Traditional Calls: ~{estimated_old_calls}")
        print(f"   Optimized Framework Calls: ~{estimated_new_calls}")
        print(f"   Theoretical Efficiency Gain: {efficiency:.1f}% reduction")
        print(f"\n✨ Key Optimizations:")
        print(f"   • Batch level analysis instead of pairwise comparisons")
        print(f"   • Reduced LLM calls from O(n²×levels) to O(levels)")
        print(f"   • Smart fallback mechanisms for robustness")
        
        return {
            'optimized': optimized_result,
            'efficiency_gain': efficiency
        }
        
    except Exception as e:
        print(f"❌ Efficiency analysis error: {str(e)}")
        return None

def demo_enhanced_features():
    """Demonstrate the key features of the optimized framework."""
    print("🚀 OPTIMIZED COMPARATIVE REASONING FRAMEWORK v3.0")
    print("=" * 60)
    print()
    print("🔑 Key Optimizations:")
    print("  ⚡ Batch Level Analysis - Process all claims at each level together")
    print("  🎯 Reduced LLM Calls - From O(n²×levels) to O(levels)")
    print("  📊 Batch Divergence Judgment - Judge multiple conflicts simultaneously")
    print("  🧠 Smart Fallback - Graceful degradation when batch analysis fails")
    print("  💾 Memory Efficient - Lower token usage and faster processing")
    print()
    print("🏗️ Optimized Architecture:")
    print("  1️⃣ Option-Specific Analysis - Generate separate trees (unchanged)")
    print("  2️⃣ Batch Level Analysis - Analyze entire levels simultaneously")
    print("  3️⃣ Batch Divergence Judgment - Resolve multiple conflicts at once")
    print("  4️⃣ Final Answer Selection - Comprehensive decision making")
    print()
    print("📈 Efficiency Improvements:")
    print("  • ~90% reduction in LLM calls for 5-option questions")
    print("  • Faster processing with batch operations")
    print("  • Maintained accuracy with optimized workflows")
    print("  • Better token efficiency and cost reduction")
    print("  • Scalable to larger option sets")

def simple_test():
    """Run a simple test with minimal output."""
    print("🧪 SIMPLE TEST - Optimized Comparative Reasoning")
    print("-" * 50)
    
    # Configure DSPy
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        cache=False,
        temperature=0.1
    )
    dspy.configure(lm=lm)
    
    solver = OptimizedComparativeReasoningSolver()
    
    question = "A patient with shortness of breath has elevated heart rate. What is the most likely cause?"
    options = {
        "A": "Heart failure",
        "B": "Anxiety", 
        "C": "Hyperthyroidism",
        "D": "Anemia"
    }
    
    try:
        result = solver(question=question, options=options)
        print(f"✅ Answer: {result['answer']}")
        print(f"📊 Confidence: {result['confidence']:.2f}")
        print(f"🔬 Method: {result['reasoning_method']}")
        print(f"📈 Claims Generated: {sum(len(claims) for claims in result['option_trees'].values())}")
        print(f"⚔️ Divergences Found: {sum(len(divs) for divs in result['level_divergences'].values())}")
        print(f"⚡ Efficiency: {result['optimization_stats']['estimated_call_reduction']}")
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_visualization_demo():
    """Demo just the visualization capabilities with a quick test."""
    print("🎨 VISUALIZATION DEMO - Optimized Framework")
    print("-" * 50)
    
    # Configure DSPy
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        cache=False,
        temperature=0.1
    )
    dspy.configure(lm=lm)
    
    solver = OptimizedComparativeReasoningSolver()
    
    # Simple medical question for quick demo
    question = "A patient presents with chest pain and elevated troponin. What is the most likely diagnosis?"
    options = {
        "A": "Myocardial infarction",
        "B": "Pulmonary embolism",
        "C": "Aortic dissection"
    }
    
    try:
        result = solver(question=question, options=options)
        
        print("\n🎨 DEMONSTRATING OPTIMIZED VISUALIZATIONS:")
        print("=" * 60)
        
        # Show just the key visualizations
        visualize_optimized_comparative_summary(result)
        visualize_batch_relationships(result.get('level_relationships', {}))
        visualize_optimization_stats(result)
        
        return True
    except Exception as e:
        print(f"❌ Visualization demo error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Optimized Medical Comparative Reasoning Framework v3.0")
    print("=" * 80)
    
    # Demo the optimized features
    demo_enhanced_features()
    
    # Ask user for preference
    print(f"\n🧪 Choose test mode:")
    print("1. Simple Test (quick validation)")
    print("2. Visualization Demo (show new features)")
    print("3. Efficiency Analysis (show optimization metrics)")
    print("4. Standard Test (comprehensive but focused)")
    print("5. Complete Analysis (full detailed output)")
    
    # For automatic execution, use complete analysis
    mode = "5"  # Change this to test different modes
    
    if mode == "1":
        print(f"\n🧪 Running Simple Test...")
        if simple_test():
            print(f"\n✅ Simple test completed successfully!")
        else:
            print(f"\n❌ Simple test failed.")
    
    elif mode == "2":
        print(f"\n🧪 Running Visualization Demo...")
        if test_visualization_demo():
            print(f"\n✅ Visualization demo completed successfully!")
        else:
            print(f"\n❌ Visualization demo failed.")
    
    elif mode == "3":
        print(f"\n🧪 Running Efficiency Analysis...")
        result = test_efficiency_comparison()
        if result:
            print(f"\n✅ Efficiency analysis completed successfully!")
            print(f"Efficiency Gain: {result['efficiency_gain']:.1f}% reduction in LLM calls")
        else:
            print(f"\n❌ Efficiency analysis failed.")
    
    elif mode == "4":
        print(f"\n🧪 Running Standard Test...")
        result = test_optimized_comparative_reasoning(detailed=False)
        if result:
            print(f"\n✅ Standard test completed successfully!")
            print(f"Selected Answer: {result['answer']} with confidence {result['confidence']:.2f}")
        else:
            print(f"\n❌ Standard test failed.")
    
    elif mode == "5":
        print(f"\n🧪 Running Complete Analysis...")
        result = test_optimized_comparative_reasoning(detailed=True)
        if result:
            print(f"\n✅ Complete analysis finished successfully!")
            print(f"Final Answer: {result['answer']} (Confidence: {result['confidence']:.2f})")
        else:
            print(f"\n❌ Complete analysis failed.")
    
    print(f"\n🎉 Testing complete!") 