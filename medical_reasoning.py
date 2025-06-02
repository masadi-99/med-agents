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
    TEXTBOOK = "TEXTBOOK"  # Standard medical textbooks
    GUIDELINE = "GUIDELINE"  # Clinical guidelines
    RESEARCH = "RESEARCH"  # Peer-reviewed research papers
    PHYSIOLOGY = "PHYSIOLOGY"  # Basic physiological principles
    CLINICAL_REASONING = "CLINICAL_REASONING"  # Expert clinical reasoning
    PATIENT_HISTORY = "PATIENT_HISTORY"  # Patient-reported history
    PHYSICAL_EXAM = "PHYSICAL_EXAM"  # Direct physical examination findings

class VerificationStatus(Enum):
    VERIFIED = "VERIFIED"
    VERIFIED_WITH_CONTEXT = "VERIFIED_WITH_CONTEXT"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    CONTRADICTED = "CONTRADICTED"

class RelevanceStatus(Enum):
    RELEVANT = "RELEVANT"
    PARTIALLY_RELEVANT = "PARTIALLY_RELEVANT"
    IRRELEVANT = "IRRELEVANT"

# Evidence quality definitions
EVIDENCE_QUALITY_MAP = {
    'A': 'High-quality evidence (clinical guidelines, standard textbooks)',
    'B': 'Moderate-quality evidence (reputable medical sources, consensus statements)',
    'C': 'Expert opinion or clinical reasoning',
    'D': 'Patient-reported or anecdotal evidence',
    'F': 'No clear evidence or unverifiable'
}

# Confidence level definitions
CONFIDENCE_LEVELS = {
    'HIGH': '>90% certainty, strong evidence or consensus',
    'MODERATE': '60-90% certainty, moderate evidence or expert consensus',
    'LOW': '<60% certainty, limited evidence or controversial'
}

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
    truth_status: Optional[str]  # VERIFIED, PARTIALLY_VERIFIED, UNVERIFIED, CONTRADICTED
    relevance_status: str  # RELEVANT, PARTIALLY_RELEVANT, IRRELEVANT

# ============= Normalization Functions =============

def normalize_verification_status(status: str) -> str:
    """Normalize verification status to ensure consistent format."""
    if not status:
        return VerificationStatus.UNVERIFIED.value
    
    normalized = status.upper().strip().rstrip('.').rstrip('!')
    
    # Map to new verification status enum
    status_map = {
        'VERIFIED': VerificationStatus.VERIFIED.value,
        'VERIFIED_WITH_CONTEXT': VerificationStatus.VERIFIED_WITH_CONTEXT.value,
        'PARTIALLY_VERIFIED': VerificationStatus.PARTIALLY_VERIFIED.value,
        'UNVERIFIED': VerificationStatus.UNVERIFIED.value,
        'CONTRADICTED': VerificationStatus.CONTRADICTED.value,
        # Remove CONTEXT_MISMATCH - handle as relevance instead
    }
    
    return status_map.get(normalized, VerificationStatus.UNVERIFIED.value)

def normalize_relevance_status(status: str) -> str:
    """Normalize relevance status to ensure consistent format."""
    if not status:
        return RelevanceStatus.RELEVANT.value
    
    normalized = status.upper().strip().rstrip('.').rstrip('!')
    
    status_map = {
        'RELEVANT': RelevanceStatus.RELEVANT.value,
        'PARTIALLY_RELEVANT': RelevanceStatus.PARTIALLY_RELEVANT.value,
        'IRRELEVANT': RelevanceStatus.IRRELEVANT.value,
    }
    
    return status_map.get(normalized, RelevanceStatus.RELEVANT.value)

def normalize_clinical_relevance(status: str) -> str:
    """Normalize clinical relevance status to ensure consistent format."""
    if not status:
        return RelevanceStatus.RELEVANT.value
    
    normalized = status.upper().strip().rstrip('.').rstrip('!')
    
    status_map = {
        'RELEVANT': RelevanceStatus.RELEVANT.value,
        'PARTIALLY_RELEVANT': RelevanceStatus.PARTIALLY_RELEVANT.value,
        'IRRELEVANT': RelevanceStatus.IRRELEVANT.value,
    }
    
    return status_map.get(normalized, RelevanceStatus.RELEVANT.value)

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
    """Decomposes reasoning into structured, verifiable claims with explicit clinical prioritization.
    
    CLAIM DEPENDENCY GUIDELINES:
    - Start with BASIC FACTS (patient history, symptoms, exam findings)
    - Build PHYSIOLOGICAL FOUNDATION claims that depend on basic facts
    - Create MECHANISM claims that depend on physiological foundations
    - Develop CLINICAL INTERPRETATION claims that depend on mechanisms
    - End with ANSWER JUSTIFICATION claims that depend on clinical interpretations
    
    CLAIM HIERARCHY REQUIREMENTS:
    1. Level 1: Basic Facts (no dependencies)
    2. Level 2: Physiological Context (depends on Level 1)
    3. Level 3: Pathophysiological Mechanisms (depends on Level 2)
    4. Level 4: Clinical Manifestations (depends on Level 3)
    5. Level 5: Answer Justification (depends on Level 4)
    
    Each claim should explicitly build upon previous claims to create a coherent reasoning chain.
    """
    
    question: str = dspy.InputField()
    options: Dict[str, str] = dspy.InputField()
    clinical_analysis: Dict[str, Any] = dspy.InputField()
    
    claims: List[Dict] = dspy.OutputField(
        desc="""List of structured claims with format:
        {
            claim_id: str,
            hierarchy_level: int (1-5),
            claim_type: FACT/INFERENCE/DEFINITION/ASSUMPTION/CONDITION,
            statement: str (explicit and specific),
            context: str (conditions under which claim is true),
            assumptions: List[str] (explicit assumptions),
            depends_on: List[str] (IDs of prerequisite claims),
            verification_method: TEXTBOOK/GUIDELINE/RESEARCH/PHYSIOLOGY/CLINICAL_REASONING/PATIENT_HISTORY/PHYSICAL_EXAM,
            supports_option: str (A/B/C/D/E or null),
            contradicts_options: str (comma-separated),
            confidence: HIGH/MODERATE/LOW,
            expected_clinical_relevance: RELEVANT/PARTIALLY_RELEVANT/IRRELEVANT (initial estimation),
            reasoning_bridge: str (how this claim connects to dependent claims)
        }"""
    )
    
    term_definitions: Dict[str, str] = dspy.OutputField(
        desc="Key medical terms used and their definitions"
    )
    claim_hierarchy_explanation: str = dspy.OutputField(
        desc="Explanation of how claims build upon each other to form a coherent reasoning chain"
    )

class ContextAwareVerifier(dspy.Signature):
    """Verifies claims explicitly separating truth from clinical prioritization.
    
    RELEVANCE ASSESSMENT GUIDANCE:
    When assessing relevance, always prioritize claims that directly explain or significantly 
    contribute to understanding the patient's primary presenting symptoms or clinical deterioration. 
    Claims that only indirectly or partially explain the primary symptoms should be marked as 
    PARTIALLY_RELEVANT. Claims unrelated to the primary symptoms should be marked as IRRELEVANT.
    """
    
    claim: Dict[str, Any] = dspy.InputField()
    dependent_claims: List[Dict[str, Any]] = dspy.InputField(
        desc="Claims that this claim depends on"
    )
    clinical_context: str = dspy.InputField()
    term_definitions: Dict[str, str] = dspy.InputField()
    
    verification_status: str = dspy.OutputField(
        desc="VERIFIED, VERIFIED_WITH_CONTEXT, PARTIALLY_VERIFIED, UNVERIFIED, CONTRADICTED"
    )
    clinical_relevance: str = dspy.OutputField(
        desc="Explicit clinical relevance assessment: RELEVANT, PARTIALLY_RELEVANT, IRRELEVANT"
    )
    evidence_quality: str = dspy.OutputField(desc="A, B, C, D, or F")
    verification_explanation: str = dspy.OutputField(
        desc="Detailed explanation explicitly separating truth verification from clinical prioritization."
    )

class ClinicalPrioritizer(dspy.Signature):
    """Explicitly prioritizes verified claims based on clinical relevance to patient's acute presentation."""
    
    verified_claims: List[Dict[str, Any]] = dspy.InputField()
    clinical_context: str = dspy.InputField()
    patient_presentation: Dict[str, str] = dspy.InputField()
    
    clinical_contextualization: Dict[str, str] = dspy.OutputField(
        desc="Explicit clinical contextualization of each verified claim, clearly stating how each physiological change manifests clinically in this patient scenario."
    )
    clinical_prioritization: List[str] = dspy.OutputField(
        desc="Ranked list of claim IDs based on their clinical relevance to the patient's acute presentation (most relevant first)."
    )
    primary_mechanism_claim: str = dspy.OutputField(
        desc="Claim ID explicitly identified as the primary physiological mechanism responsible for the patient's acute clinical deterioration."
    )
    pitfalls_and_alternatives: List[str] = dspy.OutputField(
        desc="Explicitly stated potential pitfalls or alternative explanations considered during reasoning."
    )

class DependencyAwareSelector(dspy.Signature):
    """Selects answer by constructing and validating a complete reasoning chain from basic facts to conclusion.
    
    REASONING CHAIN CONSTRUCTION:
    1. Identify foundational claims (Level 1: Basic facts)
    2. Trace physiological context (Level 2: Normal/abnormal physiology) 
    3. Establish pathophysiological mechanisms (Level 3: Disease processes)
    4. Connect to clinical manifestations (Level 4: How pathophysiology → symptoms)
    5. Justify answer selection (Level 5: Why this option best explains the chain)
    
    INSTRUCTIONS FOR SELECTOR:
    - Explicitly use the provided primary_mechanism_claim as the main justification for selecting the correct answer.
    - Clearly state how the primary mechanism claim directly explains the patient's acute symptoms.
    - Explicitly state why other claims, although verified, are less clinically relevant.
    
    Ensure each step logically follows from the previous, creating an unbroken chain of reasoning.
    """
    
    question: str = dspy.InputField()
    options: Dict[str, str] = dspy.InputField()
    verified_claims: List[Dict[str, Any]] = dspy.InputField()
    claim_dependencies: Dict[str, List[str]] = dspy.InputField(
        desc="Map of claim IDs to their dependencies"
    )
    clinical_prioritization: List[str] = dspy.InputField(
        desc="Pre-determined clinical prioritization from ClinicalPrioritizer"
    )
    primary_mechanism_claim: str = dspy.InputField(
        desc="Pre-identified primary mechanism claim from ClinicalPrioritizer"
    )
    
    answer: str = dspy.OutputField(desc="Selected answer (A/B/C/D/E)")
    confidence_score: float = dspy.OutputField()
    reasoning_chain_validation: Dict[str, List[str]] = dspy.OutputField(
        desc="Validation of reasoning chain by hierarchy level: {level_1: [claim_ids], level_2: [claim_ids], ...}"
    )
    causal_pathway: List[str] = dspy.OutputField(
        desc="Step-by-step causal pathway from basic facts → physiology → pathophysiology → symptoms → answer"
    )
    reasoning_chain: List[str] = dspy.OutputField(
        desc="Step-by-step reasoning explicitly linking claims to the patient's primary symptoms and clearly stating why the selected option best explains these symptoms."
    )
    critical_claims: List[str] = dspy.OutputField(
        desc="IDs of claims that directly explain or significantly contribute to understanding the patient's primary presenting symptoms."
    )
    reasoning_gaps: List[str] = dspy.OutputField(
        desc="Identified gaps or weak links in the reasoning chain that need attention"
    )

# ============= Enhanced MCQ Solver =============

class EnhancedMedicalMCQSolver(dspy.Module):
    """Enhanced solver with explicit clinical prioritization and contextualization."""
    
    def __init__(self):
        super().__init__()
        self.analyzer = dspy.Predict(EnhancedMedicalAnalyzer)
        self.decomposer = dspy.Predict(EnhancedClaimDecomposer)
        self.verifier = dspy.Predict(ContextAwareVerifier)
        self.prioritizer = dspy.Predict(ClinicalPrioritizer)
        self.selector = dspy.Predict(DependencyAwareSelector)
    
    def forward(self, question: str, options: Dict[str, str]) -> Dict:
        # Step 1: Clinical analysis
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
        
        # Step 3: Verify claims explicitly separating truth and relevance
        claim_dict = {c['claim_id']: c for c in decomposition.claims}
        verified_claims = []
        verified_ids = set()
        
        def verify_claim_with_deps(claim_id: str):
            if claim_id in verified_ids:
                return
            
            claim = claim_dict.get(claim_id)
            if not claim:
                return
            
            # Verify dependencies first
            for dep_id in claim.get('depends_on', []):
                verify_claim_with_deps(dep_id)
            
            dependent_claims = [
                vc for vc in verified_claims 
                if vc['claim_id'] in claim.get('depends_on', [])
            ]
            
            verification = self.verifier(
                claim=claim,
                dependent_claims=dependent_claims,
                clinical_context=analysis.clinical_context,
                term_definitions=decomposition.term_definitions
            )
            
            verified_claim = {
                **claim,
                'verification_status': normalize_verification_status(verification.verification_status),
                'clinical_relevance': normalize_clinical_relevance(verification.clinical_relevance),
                'evidence_quality': verification.evidence_quality,
                'verification_explanation': verification.verification_explanation,
                'truth_status': normalize_verification_status(verification.verification_status),
            }
            
            # Explicitly log warnings if a claim supporting an option is marked as partially relevant or irrelevant
            if verified_claim['clinical_relevance'] != RelevanceStatus.RELEVANT.value and claim.get('supports_option'):
                print(f"⚠️ Warning: Claim {claim['claim_id']} supporting option {claim['supports_option']} marked as {verified_claim['clinical_relevance']}. Review clinical relevance assessment.")
            
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
        
        # Step 4: Explicit Clinical Prioritization (NEW STEP)
        prioritization = self.prioritizer(
            verified_claims=verified_claims,
            clinical_context=analysis.clinical_context,
            patient_presentation=analysis.patient_presentation
        )
        
        # Step 5: Select answer with explicit clinical prioritization and contextualization
        selection = self.selector(
            question=question,
            options=options,
            verified_claims=verified_claims,
            claim_dependencies=claim_dependencies,
            clinical_prioritization=prioritization.clinical_prioritization,
            primary_mechanism_claim=prioritization.primary_mechanism_claim
        )
        
        return {
            'answer': selection.answer,
            'confidence': selection.confidence_score,
            'clinical_analysis': clinical_analysis,
            'claims': decomposition.claims,
            'verified_claims': verified_claims,
            'term_definitions': decomposition.term_definitions,
            'claim_hierarchy_explanation': decomposition.claim_hierarchy_explanation,
            'reasoning_chain_validation': selection.reasoning_chain_validation,
            'causal_pathway': selection.causal_pathway,
            'clinical_contextualization': prioritization.clinical_contextualization,
            'clinical_prioritization': prioritization.clinical_prioritization,
            'primary_mechanism_claim': prioritization.primary_mechanism_claim,
            'reasoning_chain': selection.reasoning_chain,
            'pitfalls_and_alternatives': prioritization.pitfalls_and_alternatives,
            'critical_claims': selection.critical_claims,
            'reasoning_gaps': selection.reasoning_gaps,
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

def visualize_clinical_prioritization(clinical_prioritization: List[str], verified_claims: List[Dict]):
    """Visualize clinical prioritization of claims."""
    print("\nClinical Prioritization (Most to Least Relevant):")
    print("-" * 60)
    
    for i, claim_id in enumerate(clinical_prioritization, 1):
        claim = next((c for c in verified_claims if c['claim_id'] == claim_id), None)
        if claim:
            relevance = claim.get('clinical_relevance', 'UNKNOWN')
            print(f"  {i}. {claim_id} ({relevance}): {claim['statement'][:60]}...")

def visualize_clinical_contextualization(clinical_contextualization: Dict[str, str]):
    """Visualize clinical contextualization of claims."""
    print("\nClinical Contextualization:")
    print("-" * 50)
    
    for claim_id, context in clinical_contextualization.items():
        print(f"  {claim_id}: {context}")

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
        'confidence': 'HIGH',
        'truth_status': 'VERIFIED',
        'clinical_relevance': 'RELEVANT',
        'expected_clinical_relevance': 'RELEVANT'
    }
    
    print("Well-Structured Claim Example:")
    print(f"  Type: {example_claim['claim_type']}")
    print(f"  Statement: {example_claim['statement']}")
    print(f"  Context: {example_claim['context']}")
    print(f"  Truth Status: {example_claim['truth_status']}")
    print(f"  Clinical Relevance: {example_claim['clinical_relevance']}")
    print(f"  Expected Clinical Relevance: {example_claim['expected_clinical_relevance']}")
    print(f"  Assumptions: {example_claim['assumptions']}")
    print(f"  Dependencies: {example_claim['depends_on']}")
    
    # Show evidence quality and confidence definitions
    print("\nEvidence Quality Levels:")
    for grade, description in EVIDENCE_QUALITY_MAP.items():
        print(f"  {grade}: {description}")
    
    print("\nConfidence Levels:")
    for level, description in CONFIDENCE_LEVELS.items():
        print(f"  {level}: {description}")
    
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
    print(f"  Problems: Ambiguous, missing context, hidden assumptions, no truth/relevance separation, no clinical prioritization")

def visualize_reasoning_hierarchy(verified_claims: List[Dict], claim_hierarchy_explanation: str):
    """Visualize the hierarchical structure of reasoning claims."""
    print("\n🏗️ REASONING HIERARCHY:")
    print("-" * 50)
    print(f"Hierarchy Explanation: {claim_hierarchy_explanation}\n")
    
    # Group claims by hierarchy level
    levels = {}
    for claim in verified_claims:
        level = claim.get('hierarchy_level', 0)
        if level not in levels:
            levels[level] = []
        levels[level].append(claim)
    
    # Display each level
    level_names = {
        1: "Level 1: Basic Facts",
        2: "Level 2: Physiological Context", 
        3: "Level 3: Pathophysiological Mechanisms",
        4: "Level 4: Clinical Manifestations",
        5: "Level 5: Answer Justification"
    }
    
    for level in sorted(levels.keys()):
        if level > 0:  # Skip level 0 (unassigned)
            print(f"\n{level_names.get(level, f'Level {level}')}:")
            print("-" * 30)
            for claim in levels[level]:
                print(f"  • {claim['claim_id']}: {claim['statement'][:80]}...")
                if claim.get('reasoning_bridge'):
                    print(f"    Bridge: {claim['reasoning_bridge']}")
                if claim.get('depends_on'):
                    print(f"    Depends on: {claim['depends_on']}")

def visualize_causal_pathway(causal_pathway: List[str]):
    """Visualize the causal pathway from facts to conclusion."""
    print("\n🔗 CAUSAL PATHWAY:")
    print("-" * 40)
    for i, step in enumerate(causal_pathway, 1):
        arrow = " ↓ " if i < len(causal_pathway) else ""
        print(f"  {i}. {step}{arrow}")

def visualize_reasoning_gaps(reasoning_gaps: List[str]):
    """Visualize identified gaps in reasoning."""
    if reasoning_gaps:
        print("\n⚠️ REASONING GAPS IDENTIFIED:")
        print("-" * 45)
        for gap in reasoning_gaps:
            print(f"  • {gap}")
    else:
        print("\n✅ No reasoning gaps identified - chain appears complete") 