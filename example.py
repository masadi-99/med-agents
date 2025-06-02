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
    example_better_claims
)
from collections import Counter

def test_enhanced_solver():
    """Test the enhanced medical MCQ solver with detailed output."""
    
    # Configure DSPy with OpenAI
    lm = dspy.LM(
        model="openai/gpt-4",
        api_key=OPENAI_API_KEY,
        cache=False,
        temperature=0.1
    )
    dspy.configure(lm=lm)
    
    # Create solver instance
    solver = EnhancedMedicalMCQSolver()
    
    # Test question - Heart failure with ASD
    question = """
    A 45-year-old woman with a known atrial septal defect (ASD) presents to the emergency department with increasing shortness of breath, fatigue, and lower extremity edema over the past 2 weeks. She reports that she has been feeling progressively worse despite compliance with her medications. On examination, her blood pressure is 95/60 mmHg, heart rate is 110 bpm, and oxygen saturation is 88% on room air. She has jugular venous distension, bilateral lower extremity edema, and hepatomegaly. Echocardiography shows right heart enlargement and an estimated pulmonary artery pressure of 70 mmHg. Laboratory results show a brain natriuretic peptide (BNP) level of 800 pg/mL.
    
    What is the most likely physiological mechanism responsible for her acute clinical deterioration?
    """
    
    options = {
        'A': 'Decreased left ventricular preload due to right-to-left shunting',
        'B': 'Increased pulmonary vascular resistance leading to right heart failure', 
        'C': 'Decreased myocardial contractility due to volume overload',
        'D': 'Increase in blood volume due to sodium and water retention',
        'E': 'Decreased systemic vascular resistance causing hypotension'
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
        example_better_claims()
        
        return result
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_enhanced_solver() 