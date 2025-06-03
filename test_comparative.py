"""
Test script for Enhanced Comparative Reasoning Framework v3.0
"""

import dspy
from config import OPENAI_API_KEY
from medical_reasoning import (
    EnhancedComparativeReasoningSolver,
    visualize_enhanced_option_trees,
    visualize_level_divergences,
    visualize_structured_resolutions,
    visualize_enhanced_comparative_summary
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
    solver = EnhancedComparativeReasoningSolver()
    
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
        # Get enhanced comparative analysis
        result = solver(question=question, options=options)
        
        # Enhanced Visualizations
        print("\n" + "="*80)
        print("ENHANCED COMPARATIVE REASONING ANALYSIS")
        print("="*80)
        
        visualize_enhanced_comparative_summary(result)
        
        visualize_enhanced_option_trees(result['option_trees'], result['option_analyses'])
        
        visualize_level_divergences(result['level_divergences'])
        
        visualize_structured_resolutions(result['divergence_resolutions'], result['level_scores'])
        
        return result
        
    except Exception as e:
        print(f"❌ Error during enhanced comparative reasoning: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def demo_enhanced_features():
    """Demonstrate the key features of the enhanced framework."""
    print("🎯 ENHANCED COMPARATIVE REASONING FRAMEWORK v3.0")
    print("=" * 60)
    print()
    print("🔑 Key Features:")
    print("  ✨ Pairwise Claim Matching - More precise than global matching")
    print("  📊 Level-Based Divergence Analysis - Structured by reasoning hierarchy")
    print("  ⚖️ Weighted Scoring System - Level-aware importance weighting")
    print("  🎨 Enhanced Visualization - Clear structured display")
    print("  🧠 Structured Judgment - Context-aware divergence resolution")
    print()
    print("🏗️ Framework Architecture:")
    print("  1️⃣ Option-Specific Analysis - Generate separate trees for each option")
    print("  2️⃣ Hierarchical Claim Decomposition - 5-level structured reasoning")
    print("  3️⃣ Pairwise Claim Matching - Compare claims across options")
    print("  4️⃣ Level-Based Divergence Detection - Find conflicts by hierarchy")
    print("  5️⃣ Structured Judgment - Resolve divergences with level weighting")
    print("  6️⃣ Final Answer Selection - Comprehensive decision making")
    print()
    print("📈 Improvements over Original:")
    print("  • Better claim matching accuracy")
    print("  • Hierarchy-aware conflict resolution")
    print("  • Weighted scoring by reasoning level")
    print("  • More transparent decision process")
    print("  • Reduced complexity and better focus")

def simple_test():
    """Run a simple test with minimal output."""
    print("🧪 SIMPLE TEST - Enhanced Comparative Reasoning")
    print("-" * 50)
    
    # Configure DSPy
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        cache=False,
        temperature=0.1
    )
    dspy.configure(lm=lm)
    
    solver = EnhancedComparativeReasoningSolver()
    
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
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Enhanced Medical Comparative Reasoning Framework v3.0")
    print("=" * 80)
    
    # Demo the enhanced features
    demo_enhanced_features()
    
    # Run a simple test first
    print(f"\n🧪 Running Simple Test...")
    if simple_test():
        print(f"\n🧪 Running Comprehensive Test...")
        result = test_enhanced_comparative_reasoning()
        
        if result:
            print(f"\n✅ Framework completed successfully!")
            print(f"Selected Answer: {result['answer']} with confidence {result['confidence']:.2f}")
            print(f"Check detailed analysis above.")
        else:
            print(f"\n❌ Framework encountered an error.")
    else:
        print(f"\n❌ Simple test failed. Check configuration.")
    
    print(f"\n🎉 Testing complete!") 