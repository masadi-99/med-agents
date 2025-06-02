"""
Example usage of the Enhanced Medical Verifiable Reasoning Framework v2.0
"""

import dspy
from medical_reasoning import EnhancedMedicalMCQSolver, visualize_claim_dependencies, example_better_claims

def setup_dspy():
    """Configure DSPy with your API key."""
    try:
        from config import OPENAI_API_KEY
        api_key = OPENAI_API_KEY
    except ImportError:
        print("Error: config.py file not found!")
        print("Please create a config.py file with your OPENAI_API_KEY")
        return None
    
    lm = dspy.LM('openai/gpt-4o-mini', 
                 api_key=api_key,
                 cache=False,  # Disable caching for better reproducibility
                 temperature=0,
                 seed=42,
                 top_p=0)
    
    dspy.configure(lm=lm)
    return lm

def test_enhanced_solver():
    """Test the enhanced medical MCQ solver."""
    
    print("="*70)
    print("Testing Enhanced Medical Verifiable Reasoning Framework v2.0")
    print("="*70)
    
    # The problematic question from the example
    question = """A 22-year-old woman from a rural area who recently discovered she was pregnant is referred for a cardiology consultation due to cyanosis, dyspnea, and a cardiac murmur revealed at the initial prenatal visit. She is gravida 1, para 0 with an estimated gestational age of 19 weeks. She says that the murmur was found in her childhood, and the doctor at that time placed her under observation only. However, she has been lost to follow-up and has not had proper follow up in years. Currently, she complains of dizziness and occasional dyspnea on exertion which has gradually increased during her pregnancy. Prior to her pregnancy, she did not have any symptoms. The vital signs are as follows: blood pressure 125/60 mm Hg, heart rate 81/min, respiratory rate 13/min, and temperature 36.7°C (98.0°F). Her examination is significant for acrocyanosis and a fixed splitting of S2 and grade 3/6 midsystolic murmur best heard over the left upper sternal border. Which of the following physiological pregnancy changes is causing the change in this patient's condition?"""
    
    options = {
        'A': 'Increase in heart rate',
        'B': 'Decrease in systemic vascular resistance',
        'C': 'Increase in cardiac output',
        'D': 'Increase in blood volume'
    }
    
    # Initialize enhanced solver
    solver = EnhancedMedicalMCQSolver()
    
    print("\nQuestion:")
    print(question[:200] + "...")
    print("\nOptions:")
    for k, v in options.items():
        print(f"  {k}: {v}")
    
    print("\nProcessing with enhanced framework...")
    
    try:
        result = solver(question=question, options=options)
        
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        
        print(f"\nSelected Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.2f}")
        
        # Show clinical analysis
        print("\nClinical Analysis:")
        print(f"  Context: {result['clinical_analysis']['clinical_context']}")
        print(f"  Key findings: {result['clinical_analysis']['patient_presentation']}")
        
        # Show critical claims
        print("\nCritical Claims:")
        critical_claims = [
            c for c in result['verified_claims'] 
            if c['claim_id'] in result.get('critical_claims', [])
        ]
        
        for claim in critical_claims:
            print(f"\n  Claim {claim['claim_id']} ({claim['claim_type']}):")
            print(f"    Statement: {claim['statement']}")
            print(f"    Context: {claim['context']}")
            print(f"    Verification: {claim['verification_status']}")
            if claim.get('depends_on'):
                print(f"    Depends on: {claim['depends_on']}")
        
        # Show reasoning chain
        print("\nReasoning Chain:")
        for i, step in enumerate(result['reasoning_chain'][:5]):
            print(f"  {i+1}. {step}")
        
        # Show claim dependency structure
        print("\nClaim Dependencies:")
        for claim_id, deps in result['claim_dependencies'].items():
            if deps:
                print(f"  {claim_id} → {deps}")
        
        # Statistics
        print("\nVerification Statistics:")
        status_counts = {}
        type_counts = {}
        
        for claim in result['verified_claims']:
            status = claim.get('verification_status', 'UNKNOWN')
            status_counts[status] = status_counts.get(status, 0) + 1
            
            claim_type = claim.get('claim_type', 'UNKNOWN')
            type_counts[claim_type] = type_counts.get(claim_type, 0) + 1
        
        print(f"  Status distribution: {status_counts}")
        print(f"  Type distribution: {type_counts}")
        
        # Show if context mismatches were detected
        context_mismatches = [
            c for c in result['verified_claims']
            if c.get('verification_status') == 'CONTEXT_MISMATCH'
        ]
        
        if context_mismatches:
            print("\nContext Mismatches Detected:")
            for claim in context_mismatches:
                print(f"  - {claim['statement']}")
                print(f"    Expected context: {claim['context']}")
        
        # Show dependency visualization if there are dependencies
        if any(result['claim_dependencies'].values()):
            visualize_claim_dependencies(result['claims'], result['claim_dependencies'])
        
    except Exception as e:
        print(f"\nError: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("Test completed")
    print("="*70)

if __name__ == "__main__":
    # Setup DSPy
    lm = setup_dspy()
    
    if lm is None:
        print("Failed to setup DSPy. Exiting.")
        exit(1)
    
    # Run the enhanced test
    test_enhanced_solver()
    
    # Show example of better claim structure
    example_better_claims() 