"""
Medical Verifiable Reasoning Framework v2.0 for MedQA/MEDMCQA
=============================================================

Enhanced framework with improved claim structure, explicit dependencies,
and better handling of medical context and assumptions.
"""

import dspy
from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass
import json
from enum import Enum

# ============= Enhanced Data Structures =============

class ClaimType(Enum):
    FACT = "FACT"  # Verifiable medical fact
    INFERENCE = "INFERENCE"  # Clinical reasoning/interpretation
    DEFINITION = "DEFINITION"  # Medical term definition
    ASSUMPTION = "ASSUMPTION"  # Underlying assumption
    CONDITION = "CONDITION"  # Conditional statement

class VerificationMethod(Enum):
    TEXTBOOK = "TEXTBOOK"  # Medical textbooks
    GUIDELINE = "GUIDELINE"  # Clinical guidelines
    RESEARCH = "RESEARCH"  # Research papers
    PHYSIOLOGY = "PHYSIOLOGY"  # Physiological principles
    CLINICAL_REASONING = "CLINICAL_REASONING"  # Expert reasoning

@dataclass
class EnhancedMedicalClaim:
    claim_id: str
    claim_type: ClaimType
    statement: str
    context: str  # Explicit context/conditions
    assumptions: List[str]  # Explicit assumptions
    depends_on: List[str]  # IDs of claims this depends on
    verification_method: VerificationMethod
    supports_option: Optional[str]
    contradicts_options: List[str]
    confidence: str  # HIGH, MODERATE, LOW

# ============= Normalization Functions =============

def normalize_verification_status(status: str) -> str:
    """Normalize verification status to ensure consistent format."""
    if not status:
        return "UNVERIFIED"
    
    normalized = status.upper().strip().rstrip('.').rstrip('!')
    
    status_map = {
        'VERIFIED': 'VERIFIED',
        'VERIFIED_WITH_CONTEXT': 'VERIFIED_WITH_CONTEXT',
        'PARTIALLY_VERIFIED': 'PARTIALLY_VERIFIED',
        'UNVERIFIED': 'UNVERIFIED',
        'CONTRADICTED': 'CONTRADICTED',
        'CONTEXT_MISMATCH': 'CONTEXT_MISMATCH'
    }
    
    return status_map.get(normalized, 'UNVERIFIED')

def normalize_claim_type(type_str: str) -> str:
    """Normalize claim type."""
    if not type_str:
        return ClaimType.FACT.value
    
    normalized = type_str.upper().strip()
    
    type_map = {
        'FACT': ClaimType.FACT.value,
        'INFERENCE': ClaimType.INFERENCE.value,
        'DEFINITION': ClaimType.DEFINITION.value,
        'ASSUMPTION': ClaimType.ASSUMPTION.value,
        'CONDITION': ClaimType.CONDITION.value
    }
    
    return type_map.get(normalized, ClaimType.FACT.value)

# ============= Enhanced DSPy Signatures =============

class EnhancedMedicalAnalyzer(dspy.Signature):
    """Analyzes medical case with explicit context recognition."""
    
    question: str = dspy.InputField()
    options: Dict[str, str] = dspy.InputField()
    
    patient_presentation: Dict[str, str] = dspy.OutputField(
        desc="Key patient details: age, symptoms, exam findings, test results"
    )
    clinical_context: str = dspy.OutputField(
        desc="Clinical scenario type (e.g., acute presentation, chronic disease exacerbation, medication side effect, surgical complication, metabolic disorder, infectious process, autoimmune condition, malignancy, trauma, genetic disorder)"
    )
    pathophysiology_considerations: List[str] = dspy.OutputField(
        desc="Relevant pathophysiological processes to consider"
    )
    differential_diagnoses: List[str] = dspy.OutputField(
        desc="Possible diagnoses based on presentation"
    )

class EnhancedClaimDecomposer(dspy.Signature):
    """Decomposes reasoning into structured, verifiable claims."""
    
    question: str = dspy.InputField()
    options: Dict[str, str] = dspy.InputField()
    clinical_analysis: Dict[str, Any] = dspy.InputField()
    
    claims: List[Dict] = dspy.OutputField(
        desc="""List of structured claims with format:
        {
            claim_id: str,
            claim_type: FACT/INFERENCE/DEFINITION/ASSUMPTION/CONDITION,
            statement: str (explicit and specific),
            context: str (conditions under which claim is true),
            assumptions: List[str] (explicit assumptions),
            depends_on: List[str] (IDs of prerequisite claims),
            verification_method: TEXTBOOK/GUIDELINE/RESEARCH/PHYSIOLOGY/CLINICAL_REASONING,
            supports_option: str (A/B/C/D/E or null),
            contradicts_options: str (comma-separated),
            confidence: HIGH/MODERATE/LOW
        }"""
    )
    
    term_definitions: Dict[str, str] = dspy.OutputField(
        desc="Key medical terms used and their definitions"
    )

class ContextAwareVerifier(dspy.Signature):
    """Verifies claims with explicit context checking."""
    
    claim: Dict[str, Any] = dspy.InputField()
    dependent_claims: List[Dict[str, Any]] = dspy.InputField(
        desc="Claims that this claim depends on"
    )
    clinical_context: str = dspy.InputField()
    term_definitions: Dict[str, str] = dspy.InputField()
    
    verification_status: str = dspy.OutputField(
        desc="VERIFIED, VERIFIED_WITH_CONTEXT, PARTIALLY_VERIFIED, UNVERIFIED, CONTRADICTED, CONTEXT_MISMATCH"
    )
    evidence_quality: str = dspy.OutputField(desc="A, B, C, D, or F")
    verification_explanation: str = dspy.OutputField(
        desc="Detailed explanation of verification including context considerations"
    )
    context_validity: str = dspy.OutputField(
        desc="Whether the claim's context matches the clinical scenario"
    )

class DependencyAwareSelector(dspy.Signature):
    """Selects answer considering claim dependencies and context."""
    
    question: str = dspy.InputField()
    options: Dict[str, str] = dspy.InputField()
    verified_claims: List[Dict[str, Any]] = dspy.InputField()
    claim_dependencies: Dict[str, List[str]] = dspy.InputField(
        desc="Map of claim IDs to their dependencies"
    )
    
    answer: str = dspy.OutputField(desc="Selected answer (A/B/C/D/E)")
    confidence_score: float = dspy.OutputField()
    reasoning_chain: List[str] = dspy.OutputField(
        desc="Step-by-step reasoning showing how claims lead to answer"
    )
    critical_claims: List[str] = dspy.OutputField(
        desc="IDs of claims most critical to the answer"
    )

# ============= Enhanced MCQ Solver =============

class EnhancedMedicalMCQSolver(dspy.Module):
    """Enhanced solver with better claim structure and verification."""
    
    def __init__(self):
        super().__init__()
        self.analyzer = dspy.Predict(EnhancedMedicalAnalyzer)
        self.decomposer = dspy.Predict(EnhancedClaimDecomposer)
        self.verifier = dspy.Predict(ContextAwareVerifier)
        self.selector = dspy.Predict(DependencyAwareSelector)
    
    def forward(self, question: str, options: Dict[str, str]) -> Dict:
        # Step 1: Deep clinical analysis
        analysis = self.analyzer(question=question, options=options)
        
        clinical_analysis = {
            'patient_presentation': analysis.patient_presentation,
            'clinical_context': analysis.clinical_context,
            'pathophysiology': analysis.pathophysiology_considerations,
            'differential': analysis.differential_diagnoses
        }
        
        # Step 2: Decompose into structured claims
        decomposition = self.decomposer(
            question=question,
            options=options,
            clinical_analysis=clinical_analysis
        )
        
        # Step 3: Build dependency graph and verify claims
        claim_dict = {c['claim_id']: c for c in decomposition.claims}
        verified_claims = []
        
        # Topological sort to verify in dependency order
        verified_ids = set()
        
        def verify_claim_with_deps(claim_id: str):
            if claim_id in verified_ids:
                return
            
            claim = claim_dict.get(claim_id)
            if not claim:
                return
            
            # First verify dependencies
            for dep_id in claim.get('depends_on', []):
                verify_claim_with_deps(dep_id)
            
            # Get verified dependent claims
            dependent_claims = [
                vc for vc in verified_claims 
                if vc['claim_id'] in claim.get('depends_on', [])
            ]
            
            # Verify this claim
            try:
                verification = self.verifier(
                    claim=claim,
                    dependent_claims=dependent_claims,
                    clinical_context=analysis.clinical_context,
                    term_definitions=decomposition.term_definitions
                )
                
                verified_claim = {
                    **claim,
                    'verification_status': normalize_verification_status(
                        verification.verification_status
                    ),
                    'evidence_quality': verification.evidence_quality,
                    'verification_explanation': verification.verification_explanation,
                    'context_validity': verification.context_validity
                }
                
            except Exception as e:
                verified_claim = {
                    **claim,
                    'verification_status': 'UNVERIFIED',
                    'evidence_quality': 'F',
                    'verification_explanation': f'Verification failed: {str(e)}',
                    'context_validity': 'UNKNOWN'
                }
            
            verified_claims.append(verified_claim)
            verified_ids.add(claim_id)
        
        # Verify all claims
        for claim_id in claim_dict.keys():
            verify_claim_with_deps(claim_id)
        
        # Step 4: Build dependency map
        claim_dependencies = {
            c['claim_id']: c.get('depends_on', []) 
            for c in decomposition.claims
        }
        
        # Step 5: Select answer with dependency awareness
        selection = self.selector(
            question=question,
            options=options,
            verified_claims=verified_claims,
            claim_dependencies=claim_dependencies
        )
        
        return {
            'answer': selection.answer,
            'confidence': selection.confidence_score,
            'clinical_analysis': clinical_analysis,
            'claims': decomposition.claims,
            'verified_claims': verified_claims,
            'term_definitions': decomposition.term_definitions,
            'reasoning_chain': selection.reasoning_chain,
            'critical_claims': selection.critical_claims,
            'claim_dependencies': claim_dependencies
        }

# ============= Specialized Pathophysiology Module =============

class PathophysiologyAnalyzer(dspy.Module):
    """Specialized module for complex pathophysiology questions."""
    
    def __init__(self):
        super().__init__()
        self.analyzer = dspy.Predict(self.PathophysiologySignature)
    
    class PathophysiologySignature(dspy.Signature):
        """Analyzes pathophysiological mechanisms."""
        
        clinical_presentation: str = dspy.InputField()
        question_focus: str = dspy.InputField()
        
        normal_physiology: List[str] = dspy.OutputField(
            desc="Relevant normal physiological processes"
        )
        pathological_changes: List[str] = dspy.OutputField(
            desc="Pathological changes in this condition"
        )
        compensatory_mechanisms: List[str] = dspy.OutputField(
            desc="Body's compensatory responses"
        )
        clinical_implications: List[str] = dspy.OutputField(
            desc="How physiology changes manifest clinically"
        )

# ============= Helper Functions =============

def visualize_claim_dependencies(claims: List[Dict], dependencies: Dict[str, List[str]]):
    """Create a simple text visualization of claim dependencies."""
    print("\nClaim Dependency Graph:")
    print("-" * 50)
    
    # Find root claims (no dependencies)
    root_claims = [
        c['claim_id'] for c in claims 
        if not dependencies.get(c['claim_id'], [])
    ]
    
    print("Root claims (no dependencies):")
    for root in root_claims:
        print(f"  • {root}")
    
    print("\nDependency chains:")
    visited = set()
    
    def print_chain(claim_id: str, indent: int = 0):
        if claim_id in visited:
            return
        visited.add(claim_id)
        
        claim = next((c for c in claims if c['claim_id'] == claim_id), None)
        if claim:
            print("  " * indent + f"└─ {claim_id}: {claim['statement'][:50]}...")
            
            dependents = [
                c['claim_id'] for c in claims 
                if claim_id in dependencies.get(c['claim_id'], [])
            ]
            
            for dep in dependents:
                print_chain(dep, indent + 1)
    
    for root in root_claims:
        print_chain(root)

def example_better_claims():
    """Show example of improved claim structure."""
    
    print("\nExample of Enhanced Claim Structure:")
    print("="*60)
    
    # Example of a well-structured claim
    example_claim = {
        'claim_id': 'C1',
        'claim_type': 'INFERENCE',
        'statement': 'In ASD with Eisenmenger syndrome, decreased systemic vascular resistance leads to increased right-to-left shunting',
        'context': 'Patient has long-standing uncorrected ASD with pulmonary hypertension exceeding systemic pressure',
        'assumptions': [
            'Patient has developed Eisenmenger syndrome',
            'Pulmonary vascular resistance exceeds systemic vascular resistance'
        ],
        'depends_on': ['A1', 'A2'],  # Assumption claims
        'verification_method': 'PHYSIOLOGY',
        'supports_option': 'B',
        'contradicts_options': 'A,C,D',
        'confidence': 'HIGH'
    }
    
    print("Well-Structured Claim Example:")
    print(f"  Type: {example_claim['claim_type']}")
    print(f"  Statement: {example_claim['statement']}")
    print(f"  Context: {example_claim['context']}")
    print(f"  Assumptions: {example_claim['assumptions']}")
    print(f"  Dependencies: {example_claim['depends_on']}")
    
    # Compare with poorly structured claim
    poor_claim = {
        'claim_id': 'P1',
        'statement': 'Increased cardiac output worsens symptoms',
        'context': '',  # Missing context!
        'assumptions': [],  # Hidden assumptions!
        'depends_on': []  # Hidden dependencies!
    }
    
    print("\nPoorly Structured Claim (original style):")
    print(f"  Statement: {poor_claim['statement']}")
    print(f"  Problems: Ambiguous, missing context, hidden assumptions") 