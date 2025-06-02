"""
Test script for Enhanced Medical Verifiable Reasoning Framework v2.0
"""

import dspy
from config import OPENAI_API_KEY
from medical_reasoning import (
    EnhancedMedicalMCQSolver, 
    visualize_claim_dependencies, 
    visualize_clinical_prioritization,
    visualize_clinical_contextualization,
    visualize_reasoning_hierarchy,
    visualize_causal_pathway,
    visualize_reasoning_gaps,
    example_better_claims
)
from collections import Counter

def test_enhanced_solver():
    """Test the enhanced medical MCQ solver with detailed output."""
    
    # Configure DSPy with OpenAI
    lm = dspy.LM(
        model="openai/gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        cache=False,
        temperature=0.1
    )
    dspy.configure(lm=lm)
    
    # Create solver instance
    solver = EnhancedMedicalMCQSolver()
    
    # Test question - Pregnant woman with ASD (original question)
    question = """A 22-year-old woman from a rural area who recently discovered she was pregnant is referred for a cardiology consultation due to cyanosis, dyspnea, and a cardiac murmur revealed at the initial prenatal visit. She is gravida 1, para 0 with an estimated gestational age of 19 weeks. She says that the murmur was found in her childhood, and the doctor at that time placed her under observation only. However, she has been lost to follow-up and has not had proper follow up in years. Currently, she complains of dizziness and occasional dyspnea on exertion which has gradually increased during her pregnancy. Prior to her pregnancy, she did not have any symptoms. The vital signs are as follows: blood pressure 125/60 mm Hg, heart rate 81/min, respiratory rate 13/min, and temperature 36.7°C (98.0°F). Her examination is significant for acrocyanosis and a fixed splitting of S2 and grade 3/6 midsystolic murmur best heard over the left upper sternal border. Which of the following physiological pregnancy changes is causing the change in this patient's condition?"""
    
    options = {
        'A': 'Increase in heart rate',
        'B': 'Decrease in systemic vascular resistance',
        'C': 'Increase in cardiac output',
        'D': 'Increase in blood volume'
    }
    
    print("Enhanced Medical Verifiable Reasoning Framework v2.0 Test")
    print("=" * 70)
    print(f"Question: {question}\n")
    
    print("Options:")
    for key, value in options.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # Get solution
        result = solver(question=question, options=options)
        print(result)
        # Display key findings
        print("🔍 KEY FINDINGS:")
        print("-" * 30)
        print(f"Selected Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.0%}")
        
        if 'primary_mechanism_claim' in result:
            print(f"Primary Mechanism Claim: {result['primary_mechanism_claim']}")
        
        print()
        
        # Clinical Analysis
        print("🏥 CLINICAL ANALYSIS:")
        print("-" * 30)
        clinical = result['clinical_analysis']
        print(f"Clinical Context: {clinical['clinical_context']}")
        print(f"Patient Presentation: {clinical['patient_presentation']}")
        print()
        
        # Critical Claims Analysis
        print("🎯 CRITICAL CLAIMS (Directly Explain Primary Symptoms):")
        print("-" * 60)
        critical_claims = result.get('critical_claims', [])
        for claim_id in critical_claims:
            claim = next((c for c in result['verified_claims'] if c['claim_id'] == claim_id), None)
            if claim:
                print(f"  • {claim_id}: {claim['statement']}")
                print(f"    Truth: {claim.get('truth_status', 'N/A')} | Clinical Relevance: {claim.get('clinical_relevance', 'N/A')} | Evidence: Grade {claim.get('evidence_quality', 'N/A')}")
                print()
        
        # Clinical Prioritization
        if 'clinical_prioritization' in result:
            visualize_clinical_prioritization(result['clinical_prioritization'], result['verified_claims'])
        
        # Reasoning Hierarchy
        if 'claim_hierarchy_explanation' in result:
            visualize_reasoning_hierarchy(result['verified_claims'], result['claim_hierarchy_explanation'])
        
        # Causal Pathway
        if 'causal_pathway' in result:
            visualize_causal_pathway(result['causal_pathway'])
        
        # Reasoning Gaps
        if 'reasoning_gaps' in result:
            visualize_reasoning_gaps(result['reasoning_gaps'])
        
        # Clinical Contextualization
        if 'clinical_contextualization' in result:
            visualize_clinical_contextualization(result['clinical_contextualization'])
        
        # Pitfalls and Alternatives
        if 'pitfalls_and_alternatives' in result:
            print("\n⚠️ PITFALLS & ALTERNATIVE EXPLANATIONS:")
            print("-" * 50)
            for pitfall in result['pitfalls_and_alternatives']:
                print(f"  • {pitfall}")
            print()
        
        # Full Reasoning Chain
        print("🧠 REASONING CHAIN:")
        print("-" * 30)
        for i, step in enumerate(result['reasoning_chain'], 1):
            print(f"  {i}. {step}")
        print()
        
        # Claims Summary Statistics
        verified_claims = result['verified_claims']
        
        # Truth Status Distribution
        truth_counts = Counter(claim.get('truth_status', 'UNKNOWN') for claim in verified_claims)
        print("📊 TRUTH STATUS DISTRIBUTION:")
        print("-" * 40)
        for status, count in truth_counts.items():
            print(f"  {status}: {count}")
        print()
        
        # Clinical Relevance Distribution
        relevance_counts = Counter(claim.get('clinical_relevance', 'UNKNOWN') for claim in verified_claims)
        print("📊 CLINICAL RELEVANCE DISTRIBUTION:")
        print("-" * 45)
        for relevance, count in relevance_counts.items():
            print(f"  {relevance}: {count}")
        print()
        
        # Evidence Quality Distribution
        evidence_counts = Counter(claim.get('evidence_quality', 'F') for claim in verified_claims)
        print("📊 EVIDENCE QUALITY DISTRIBUTION:")
        print("-" * 40)
        for grade, count in evidence_counts.items():
            print(f"  Grade {grade}: {count}")
        print()
        
        # Claim Dependencies Visualization
        visualize_claim_dependencies(result['claims'], result['claim_dependencies'])
        
        # Detailed Claims Analysis
        print("\n📋 DETAILED CLAIMS ANALYSIS:")
        print("-" * 50)
        for claim in verified_claims:
            print(f"\n{claim['claim_id']} ({claim.get('claim_type', 'UNKNOWN')})")
            print(f"  Statement: {claim['statement']}")
            print(f"  Truth Status: {claim.get('truth_status', 'N/A')}")
            print(f"  Clinical Relevance: {claim.get('clinical_relevance', 'N/A')}")
            print(f"  Evidence Quality: Grade {claim.get('evidence_quality', 'N/A')}")
            print(f"  Supports Option: {claim.get('supports_option', 'None')}")
            if claim.get('verification_explanation'):
                print(f"  Explanation: {claim['verification_explanation']}")
        
        # Show example of better claim structure
        #example_better_claims()
        
        return result
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_enhanced_solver() 