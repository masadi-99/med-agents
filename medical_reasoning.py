"""
Medical Verifiable Reasoning Framework v2.0 for MedQA/MEDMCQA
=============================================================

Enhanced framework with improved claim structure, explicit dependencies,
and better handling of medical context and assumptions.
"""

import dspy
from typing import List, Dict, Optional, Set, Any, Tuple
from dataclasses import dataclass
import json
from enum import Enum
from collections import defaultdict

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

# ============= Enhanced Comparative Reasoning Data Structures =============

class ClaimSimilarity(Enum):
    IDENTICAL = "IDENTICAL"  # Same claim
    SIMILAR = "SIMILAR"  # Same concept, different details
    RELATED = "RELATED"  # Related but distinct claims
    CONFLICTING = "CONFLICTING"  # Directly contradictory claims
    UNRELATED = "UNRELATED"  # No meaningful relationship

@dataclass
class ClaimComparison:
    claim1_id: str
    claim2_id: str
    option1: str
    option2: str
    similarity: ClaimSimilarity
    divergence_point: bool
    comparison_notes: str
    level: int  # Hierarchy level where divergence occurs

@dataclass
class DivergencePoint:
    divergence_id: str
    claim_pairs: List[Tuple[str, str]]  # (claim_id_option1, claim_id_option2)
    options: List[str]
    divergence_type: str  # mechanism, assumption, interpretation, factual
    description: str
    level: int  # Hierarchy level
    resolution_needed: bool

# ============= Enhanced Comparative Reasoning Signatures =============

class OptionSpecificAnalyzer(dspy.Signature):
    """Analyzes medical case assuming a specific option is the correct answer."""
    
    question: str = dspy.InputField()
    correct_option: str = dspy.InputField(desc="The option to assume is correct (e.g., 'A: Increase in heart rate')")
    
    patient_presentation: Dict[str, str] = dspy.OutputField(
        desc="Key patient details: age, symptoms, exam findings, test results"
    )
    clinical_context: str = dspy.OutputField(
        desc="Clinical scenario type focused on explaining why the given option is correct"
    )
    pathophysiology_explanation: str = dspy.OutputField(
        desc="Detailed explanation of how this specific option explains the patient's condition"
    )
    supporting_mechanisms: List[str] = dspy.OutputField(
        desc="Physiological mechanisms that support this option being correct"
    )

class PairwiseClaimMatcher(dspy.Signature):
    """Compare two specific claims to determine their relationship and divergence potential."""
    
    claim1: Dict[str, Any] = dspy.InputField()
    claim2: Dict[str, Any] = dspy.InputField()
    option1: str = dspy.InputField()
    option2: str = dspy.InputField()
    
    similarity: str = dspy.OutputField(
        desc="IDENTICAL/SIMILAR/RELATED/CONFLICTING/UNRELATED"
    )
    is_divergence_point: bool = dspy.OutputField(
        desc="Whether this represents a key divergence in reasoning between options"
    )
    divergence_type: str = dspy.OutputField(
        desc="If divergence: mechanism/assumption/interpretation/factual"
    )
    comparison_explanation: str = dspy.OutputField(
        desc="Detailed explanation of the relationship between claims"
    )
    clinical_significance: str = dspy.OutputField(
        desc="Why this comparison matters clinically for determining the correct answer"
    )

class LevelBasedDivergenceAnalyzer(dspy.Signature):
    """Analyze divergences at a specific hierarchy level between option reasoning trees."""
    
    level: int = dspy.InputField(desc="Hierarchy level being analyzed")
    option_pairs: List[Tuple[str, str]] = dspy.InputField(desc="Option pairs to compare")
    level_claims: Dict[str, List[Dict]] = dspy.InputField(
        desc="Claims at this level for each option"
    )
    divergent_comparisons: List[Dict] = dspy.InputField(
        desc="Claim comparisons that show divergence at this level"
    )
    question_context: str = dspy.InputField()
    
    level_divergences: List[Dict] = dspy.OutputField(
        desc="""Divergences identified at this level with format:
        {
            divergence_id: str,
            divergence_type: str,
            description: str,
            claim_pairs: List[Tuple[str, str]],
            options: List[str],
            critical_for_answer: bool
        }"""
    )
    level_summary: str = dspy.OutputField(
        desc="Summary of what diverges at this reasoning level"
    )

class StructuredDivergenceJudge(dspy.Signature):
    """Judge divergences with explicit consideration of hierarchy level and divergence type."""
    
    divergence: Dict = dspy.InputField(desc="Single divergence to judge")
    level: int = dspy.InputField(desc="Hierarchy level of this divergence")
    claim_details: Dict[str, Dict] = dspy.InputField(
        desc="Detailed claim information for each option in divergence"
    )
    clinical_context: str = dspy.InputField()
    patient_presentation: Dict[str, str] = dspy.InputField()
    
    winning_option: str = dspy.OutputField(desc="Option with stronger reasoning at this divergence")
    confidence: float = dspy.OutputField(desc="Confidence in judgment (0-1)")
    level_weight: float = dspy.OutputField(
        desc="Weight/importance of this level for final decision (0-1)"
    )
    divergence_impact: str = dspy.OutputField(
        desc="How this divergence impacts the overall answer selection"
    )
    reasoning: str = dspy.OutputField(
        desc="Detailed explanation considering level and divergence type"
    )
    evidence_quality: str = dspy.OutputField(desc="A, B, C, D, or F")

# ============= Original Comparative Reasoning Solver (for compatibility) =============

class ClaimMatcher(dspy.Signature):
    """Matches corresponding claims across different option reasoning trees."""
    
    option_trees: Dict[str, List[Dict]] = dspy.InputField(
        desc="Dictionary mapping option letters to their claim trees"
    )
    
    matched_claims: List[Dict] = dspy.OutputField(
        desc="""List of claim matches across trees with format:
        {
            claim_group_id: str,
            matched_claims: {option_letter: claim_id},
            claim_statements: {option_letter: statement},
            match_confidence: float,
            match_type: 'IDENTICAL'/'SIMILAR'/'CONFLICTING'
        }"""
    )
    unmatched_claims: Dict[str, List[str]] = dspy.OutputField(
        desc="Claims that couldn't be matched across trees, grouped by option"
    )

class ConflictIdentifier(dspy.Signature):
    """Identifies conflicts between matched claims from different option trees."""
    
    matched_claims: List[Dict] = dspy.InputField()
    clinical_context: str = dspy.InputField()
    
    conflicts: List[Dict] = dspy.OutputField(
        desc="""List of conflicts with format:
        {
            conflict_id: str,
            conflicting_options: List[str],
            conflict_type: 'MECHANISM'/'IMPORTANCE'/'CAUSATION'/'FREQUENCY',
            claim_statements: {option_letter: statement},
            conflict_description: str
        }"""
    )
    critical_conflicts: List[str] = dspy.OutputField(
        desc="Conflict IDs that are most critical for determining the correct answer"
    )

class ConflictJudge(dspy.Signature):
    """Judges between conflicting claims to determine which is more accurate/important."""
    
    conflict: Dict = dspy.InputField(desc="Single conflict to judge")
    clinical_context: str = dspy.InputField()
    patient_presentation: Dict[str, str] = dspy.InputField()
    
    winning_option: str = dspy.OutputField(desc="Option letter with the more accurate/important claim")
    confidence: float = dspy.OutputField(desc="Confidence in the judgment (0-1)")
    reasoning: str = dspy.OutputField(
        desc="Detailed explanation of why this option's claim is more accurate/important"
    )
    evidence_quality: str = dspy.OutputField(desc="A, B, C, D, or F")

class ComparativeReasoningSolver(dspy.Module):
    """Original solver that generates separate reasoning trees for each option and compares them."""
    
    def __init__(self):
        super().__init__()
        self.option_analyzer = dspy.Predict(OptionSpecificAnalyzer)
        self.decomposer = dspy.Predict(EnhancedClaimDecomposer)
        self.verifier = dspy.Predict(ContextAwareVerifier)
        self.prioritizer = dspy.Predict(ClinicalPrioritizer)
        self.matcher = dspy.Predict(ClaimMatcher)
        self.conflict_identifier = dspy.Predict(ConflictIdentifier)
        self.conflict_judge = dspy.Predict(ConflictJudge)
    
    def forward(self, question: str, options: Dict[str, str]) -> Dict:
        print("🔄 Starting comparative reasoning analysis...")
        
        # Step 1: Generate reasoning trees for each option
        option_trees = {}
        option_analyses = {}
        
        for option_letter, option_text in options.items():
            print(f"📊 Analyzing option {option_letter}: {option_text[:50]}...")
            
            # Analyze assuming this option is correct
            correct_option = f"{option_letter}: {option_text}"
            analysis = self.option_analyzer(
                question=question,
                correct_option=correct_option
            )
            
            clinical_analysis = {
                'patient_presentation': analysis.patient_presentation,
                'clinical_context': analysis.clinical_context,
                'pathophysiology_explanation': analysis.pathophysiology_explanation,
                'supporting_mechanisms': analysis.supporting_mechanisms
            }
            
            # Generate claims for this option
            decomposition = self.decomposer(
                question=question,
                options={option_letter: option_text},  # Only show this option
                clinical_analysis=clinical_analysis
            )
            
            # Verify claims
            verified_claims = self._verify_claims(decomposition.claims, analysis.clinical_context, decomposition.term_definitions)
            
            # Store option tree
            option_trees[option_letter] = verified_claims
            option_analyses[option_letter] = clinical_analysis
            
        print("🔍 Matching claims across option trees...")
        
        # Step 2: Match claims across trees
        matching = self.matcher(option_trees=option_trees)
        
        print("⚔️ Identifying conflicts...")
        
        # Step 3: Identify conflicts
        conflict_analysis = self.conflict_identifier(
            matched_claims=matching.matched_claims,
            clinical_context=list(option_analyses.values())[0]['clinical_context']
        )
        
        print(f"🏛️ Judging {len(conflict_analysis.conflicts)} conflicts...")
        
        # Step 4: Judge critical conflicts
        conflict_resolutions = {}
        option_scores = {opt: 0 for opt in options.keys()}
        
        for conflict in conflict_analysis.conflicts:
            if conflict['conflict_id'] in conflict_analysis.critical_conflicts:
                judgment = self.conflict_judge(
                    conflict=conflict,
                    clinical_context=list(option_analyses.values())[0]['clinical_context'],
                    patient_presentation=list(option_analyses.values())[0]['patient_presentation']
                )
                
                conflict_resolutions[conflict['conflict_id']] = judgment
                
                # Clean the winning option (remove quotes if present)
                winning_option = judgment.winning_option.strip('"\'')
                if winning_option in option_scores:
                    option_scores[winning_option] += judgment.confidence
                else:
                    print(f"⚠️ Warning: Unknown option '{winning_option}' from conflict resolution")
        
        # Step 5: Determine final answer
        best_option = max(option_scores, key=option_scores.get) if any(option_scores.values()) else list(options.keys())[0]
        best_score = option_scores[best_option] if best_option in option_scores else 0.0
        
        return {
            'answer': best_option,
            'confidence': best_score,
            'option_trees': option_trees,
            'option_analyses': option_analyses,
            'matched_claims': matching.matched_claims,
            'unmatched_claims': matching.unmatched_claims,
            'conflicts': conflict_analysis.conflicts,
            'critical_conflicts': conflict_analysis.critical_conflicts,
            'conflict_resolutions': conflict_resolutions,
            'option_scores': option_scores,
            'reasoning_method': 'comparative_analysis'
        }
    
    def _verify_claims(self, claims: List[Dict], clinical_context: str, term_definitions: Dict[str, str]) -> List[Dict]:
        """Helper method to verify claims for an option tree."""
        claim_dict = {c['claim_id']: c for c in claims}
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
                clinical_context=clinical_context,
                term_definitions=term_definitions
            )
            
            verified_claim = {
                **claim,
                'verification_status': normalize_verification_status(verification.verification_status),
                'clinical_relevance': normalize_clinical_relevance(verification.clinical_relevance),
                'evidence_quality': verification.evidence_quality,
                'verification_explanation': verification.verification_explanation,
                'truth_status': normalize_verification_status(verification.verification_status),
            }
            
            verified_claims.append(verified_claim)
            verified_ids.add(claim_id)
        
        # Verify all claims
        for claim_id in claim_dict.keys():
            verify_claim_with_deps(claim_id)
        
        return verified_claims

def visualize_option_trees(option_trees: Dict[str, List[Dict]], option_analyses: Dict[str, Dict]):
    """Visualize reasoning trees for each option."""
    print("\n🌳 OPTION-SPECIFIC REASONING TREES:")
    print("=" * 60)
    
    for option_letter, claims in option_trees.items():
        analysis = option_analyses[option_letter]
        print(f"\n📋 OPTION {option_letter} REASONING TREE:")
        print("-" * 50)
        print(f"Clinical Context: {analysis['clinical_context']}")
        print(f"Pathophysiology: {analysis['pathophysiology_explanation'][:100]}...")
        print(f"Supporting Mechanisms: {analysis['supporting_mechanisms']}")
        
        print(f"\nClaims ({len(claims)} total):")
        for claim in claims:
            print(f"  • {claim['claim_id']}: {claim['statement'][:80]}...")
            print(f"    Status: {claim.get('truth_status', 'N/A')} | Evidence: Grade {claim.get('evidence_quality', 'N/A')}")

def visualize_claim_matches(matched_claims: List[Dict], unmatched_claims: Dict[str, List[str]]):
    """Visualize matched and unmatched claims across option trees."""
    print("\n🔗 CLAIM MATCHING ANALYSIS:")
    print("=" * 50)
    
    print(f"\n✅ MATCHED CLAIMS ({len(matched_claims)} groups):")
    print("-" * 40)
    for match in matched_claims:
        print(f"\nGroup {match['claim_group_id']} ({match['match_type']}):")
        print(f"  Confidence: {match['match_confidence']:.2f}")
        for option, claim_id in match['matched_claims'].items():
            statement = match['claim_statements'][option]
            print(f"  {option}: {claim_id} - {statement[:60]}...")
    
    print(f"\n❌ UNMATCHED CLAIMS:")
    print("-" * 40)
    for option, claim_ids in unmatched_claims.items():
        if claim_ids:
            print(f"\nOption {option}: {len(claim_ids)} unique claims")
            for claim_id in claim_ids:
                print(f"  • {claim_id}")

def visualize_conflicts(conflicts: List[Dict], critical_conflicts: List[str]):
    """Visualize conflicts between option trees."""
    print("\n⚔️ CONFLICT ANALYSIS:")
    print("=" * 40)
    
    print(f"\nTotal Conflicts: {len(conflicts)}")
    print(f"Critical Conflicts: {len(critical_conflicts)}")
    
    for conflict in conflicts:
        is_critical = "🔥 CRITICAL" if conflict['conflict_id'] in critical_conflicts else "⚠️  Minor"
        print(f"\n{is_critical} - Conflict {conflict['conflict_id']} ({conflict['conflict_type']}):")
        print(f"  Options: {conflict['conflicting_options']}")
        print(f"  Description: {conflict['conflict_description']}")
        
        print("  Conflicting Claims:")
        for option, statement in conflict['claim_statements'].items():
            print(f"    {option}: {statement[:80]}...")

def visualize_conflict_resolutions(conflict_resolutions: Dict[str, Any], option_scores: Dict[str, float]):
    """Visualize conflict resolution results."""
    print("\n🏛️ CONFLICT RESOLUTIONS:")
    print("=" * 45)
    
    for conflict_id, resolution in conflict_resolutions.items():
        print(f"\nConflict {conflict_id}:")
        print(f"  Winner: Option {resolution.winning_option}")
        print(f"  Confidence: {resolution.confidence:.2f}")
        print(f"  Evidence Quality: Grade {resolution.evidence_quality}")
        print(f"  Reasoning: {resolution.reasoning[:100]}...")
    
    print(f"\n📊 FINAL OPTION SCORES:")
    print("-" * 30)
    sorted_scores = sorted(option_scores.items(), key=lambda x: x[1], reverse=True)
    for option, score in sorted_scores:
        print(f"  Option {option}: {score:.2f}")

def visualize_comparative_summary(result: Dict):
    """Visualize complete comparative reasoning summary."""
    print("\n🎯 COMPARATIVE REASONING SUMMARY:")
    print("=" * 50)
    
    print(f"🏆 Selected Answer: {result['answer']}")
    print(f"📊 Confidence: {result['confidence']:.2f}")
    print(f"🔬 Method: {result['reasoning_method']}")
    
    print(f"\n📈 Analysis Statistics:")
    print(f"  • Option Trees Generated: {len(result['option_trees'])}")
    print(f"  • Matched Claim Groups: {len(result['matched_claims'])}")
    print(f"  • Total Conflicts: {len(result['conflicts'])}")
    print(f"  • Critical Conflicts: {len(result['critical_conflicts'])}")
    print(f"  • Conflicts Resolved: {len(result['conflict_resolutions'])}")
    
    # Show reasoning for why this option won
    if result['conflict_resolutions']:
        print(f"\n🎖️  Why Option {result['answer']} Won:")
        wins = [r for r in result['conflict_resolutions'].values() 
                if r.winning_option.strip('"\'') == result['answer']]
        for resolution in wins:
            print(f"  • {resolution.reasoning[:80]}...")

# ============= Enhanced Comparative Reasoning Solver =============

class EnhancedComparativeReasoningSolver(dspy.Module):
    """Enhanced comparative solver with better claim matching and level-based analysis."""
    
    def __init__(self):
        super().__init__()
        self.option_analyzer = dspy.Predict(OptionSpecificAnalyzer)
        self.decomposer = dspy.Predict(EnhancedClaimDecomposer)
        self.verifier = dspy.Predict(ContextAwareVerifier)
        self.prioritizer = dspy.Predict(ClinicalPrioritizer)
        self.pairwise_matcher = dspy.Predict(PairwiseClaimMatcher)
        self.level_analyzer = dspy.Predict(LevelBasedDivergenceAnalyzer)
        self.structured_judge = dspy.Predict(StructuredDivergenceJudge)
        self.final_selector = dspy.Predict(FinalAnswerSelector)
    
    def forward(self, question: str, options: Dict[str, str]) -> Dict:
        print("🔄 Starting enhanced comparative reasoning analysis...")
        
        # Step 1: Generate reasoning trees for each option
        option_trees = {}
        option_analyses = {}
        
        for option_letter, option_text in options.items():
            print(f"📊 Analyzing option {option_letter}: {option_text[:50]}...")
            
            # Analyze assuming this option is correct
            correct_option = f"{option_letter}: {option_text}"
            analysis = self.option_analyzer(
                question=question,
                correct_option=correct_option
            )
            
            clinical_analysis = {
                'patient_presentation': analysis.patient_presentation,
                'clinical_context': analysis.clinical_context,
                'pathophysiology_explanation': analysis.pathophysiology_explanation,
                'supporting_mechanisms': analysis.supporting_mechanisms
            }
            
            # Generate claims for this option
            decomposition = self.decomposer(
                question=question,
                options={option_letter: option_text},  # Only show this option
                clinical_analysis=clinical_analysis
            )
            
            # Verify claims
            verified_claims = self._verify_claims(decomposition.claims, analysis.clinical_context, decomposition.term_definitions)
            
            # Store option tree
            option_trees[option_letter] = verified_claims
            option_analyses[option_letter] = clinical_analysis
        
        print("🔍 Performing pairwise claim matching...")
        
        # Step 2: Enhanced pairwise claim matching
        claim_comparisons = self._perform_pairwise_matching(option_trees)
        
        print("📊 Analyzing divergences by hierarchy level...")
        
        # Step 3: Level-based divergence analysis
        level_divergences = self._analyze_divergences_by_level(
            option_trees, claim_comparisons, question
        )
        
        print(f"🏛️ Judging {len(level_divergences)} structured divergences...")
        
        # Step 4: Structured divergence judgment
        divergence_resolutions = {}
        option_scores = {opt: 0 for opt in options.keys()}
        level_scores = {opt: defaultdict(float) for opt in options.keys()}
        
        for level, divergences in level_divergences.items():
            for divergence in divergences:
                if divergence.get('critical_for_answer', False):
                    # Get claim details for judgment
                    claim_details = {}
                    for opt in divergence['options']:
                        for claim_pair in divergence['claim_pairs']:
                            if opt == claim_pair[0][0]:  # Extract option from claim_id format
                                claim_id = claim_pair[0] if claim_pair[0].startswith(opt) else claim_pair[1]
                                claim = next((c for c in option_trees[opt] if c['claim_id'] == claim_id), None)
                                if claim:
                                    claim_details[opt] = claim
                    
                    judgment = self.structured_judge(
                        divergence=divergence,
                        level=level,
                        claim_details=claim_details,
                        clinical_context=list(option_analyses.values())[0]['clinical_context'],
                        patient_presentation=list(option_analyses.values())[0]['patient_presentation']
                    )
                    
                    div_id = divergence['divergence_id']
                    divergence_resolutions[div_id] = judgment
                    
                    # Score with level weighting
                    winning_option = judgment.winning_option.strip('"\'')
                    if winning_option in option_scores:
                        weighted_score = judgment.confidence * judgment.level_weight
                        option_scores[winning_option] += weighted_score
                        level_scores[winning_option][level] += weighted_score
                    else:
                        print(f"⚠️ Warning: Unknown option '{winning_option}' from judgment")
        
        # Step 5: Final answer selection
        best_option = max(option_scores, key=option_scores.get) if any(option_scores.values()) else list(options.keys())[0]
        best_score = option_scores[best_option] if best_option in option_scores else 0.0
        
        # Create final selection input
        final_selection = self.final_selector(
            question=question,
            options=options,
            divergence_resolutions=[
                {'divergence': div, 'judgment': res, 'level': self._extract_level_from_div_id(div_id)}
                for div_id, res in divergence_resolutions.items()
                for div in [d for level_divs in level_divergences.values() for d in level_divs if d['divergence_id'] == div_id]
            ],
            option_claim_trees=option_trees
        )
        
        return {
            'answer': best_option,
            'confidence': best_score,
            'option_trees': option_trees,
            'option_analyses': option_analyses,
            'claim_comparisons': claim_comparisons,
            'level_divergences': level_divergences,
            'divergence_resolutions': divergence_resolutions,
            'option_scores': option_scores,
            'level_scores': level_scores,
            'final_selection': final_selection,
            'reasoning_method': 'enhanced_comparative_analysis'
        }
    
    def _perform_pairwise_matching(self, option_trees: Dict[str, List[Dict]]) -> List[ClaimComparison]:
        """Perform enhanced pairwise claim matching."""
        comparisons = []
        option_keys = list(option_trees.keys())
        
        for i in range(len(option_keys)):
            for j in range(i + 1, len(option_keys)):
                opt1, opt2 = option_keys[i], option_keys[j]
                claims1, claims2 = option_trees[opt1], option_trees[opt2]
                
                # Compare claims at same hierarchy levels
                for claim1 in claims1:
                    level1 = claim1.get('hierarchy_level', 1)
                    for claim2 in claims2:
                        level2 = claim2.get('hierarchy_level', 1)
                        
                        # Only compare claims at same level
                        if level1 == level2:
                            comparison = self.pairwise_matcher(
                                claim1=claim1,
                                claim2=claim2,
                                option1=opt1,
                                option2=opt2
                            )
                            
                            comparisons.append(ClaimComparison(
                                claim1_id=claim1['claim_id'],
                                claim2_id=claim2['claim_id'],
                                option1=opt1,
                                option2=opt2,
                                similarity=ClaimSimilarity(comparison.similarity),
                                divergence_point=comparison.is_divergence_point,
                                comparison_notes=comparison.comparison_explanation,
                                level=level1
                            ))
        
        return comparisons
    
    def _analyze_divergences_by_level(self, option_trees: Dict[str, List[Dict]], 
                                    comparisons: List[ClaimComparison], 
                                    question: str) -> Dict[int, List[DivergencePoint]]:
        """Analyze divergences grouped by hierarchy level."""
        level_divergences = defaultdict(list)
        
        # Group comparisons by level
        level_comparisons = defaultdict(list)
        for comp in comparisons:
            if comp.divergence_point:
                level_comparisons[comp.level].append(comp)
        
        # Analyze each level
        for level, comps in level_comparisons.items():
            if not comps:
                continue
                
            # Group claims by level for this analysis
            level_claims = {}
            for opt, claims in option_trees.items():
                level_claims[opt] = [c for c in claims if c.get('hierarchy_level', 1) == level]
            
            # Get option pairs
            option_pairs = list(set((comp.option1, comp.option2) for comp in comps))
            
            # Analyze divergences at this level
            analysis = self.level_analyzer(
                level=level,
                option_pairs=option_pairs,
                level_claims=level_claims,
                divergent_comparisons=[{
                    'claim1_id': comp.claim1_id,
                    'claim2_id': comp.claim2_id,
                    'option1': comp.option1,
                    'option2': comp.option2,
                    'similarity': comp.similarity.value,
                    'notes': comp.comparison_notes
                } for comp in comps],
                question_context=question
            )
            
            # Convert to DivergencePoint objects
            for div_data in analysis.level_divergences:
                divergence = DivergencePoint(
                    divergence_id=div_data['divergence_id'],
                    claim_pairs=div_data['claim_pairs'],
                    options=div_data['options'],
                    divergence_type=div_data['divergence_type'],
                    description=div_data['description'],
                    level=level,
                    resolution_needed=div_data.get('critical_for_answer', True)
                )
                level_divergences[level].append(div_data)  # Store dict for easier access
        
        return dict(level_divergences)
    
    def _extract_level_from_div_id(self, div_id: str) -> int:
        """Extract level from divergence ID."""
        # Assuming format like "L1_DIV_1" 
        try:
            return int(div_id.split('_')[0][1:])
        except:
            return 1
    
    def _verify_claims(self, claims: List[Dict], clinical_context: str, term_definitions: Dict[str, str]) -> List[Dict]:
        """Helper method to verify claims for an option tree."""
        claim_dict = {c['claim_id']: c for c in claims}
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
                clinical_context=clinical_context,
                term_definitions=term_definitions
            )
            
            verified_claim = {
                **claim,
                'verification_status': normalize_verification_status(verification.verification_status),
                'clinical_relevance': normalize_clinical_relevance(verification.clinical_relevance),
                'evidence_quality': verification.evidence_quality,
                'verification_explanation': verification.verification_explanation,
                'truth_status': normalize_verification_status(verification.verification_status),
            }
            
            verified_claims.append(verified_claim)
            verified_ids.add(claim_id)
        
        # Verify all claims
        for claim_id in claim_dict.keys():
            verify_claim_with_deps(claim_id)
        
        return verified_claims

class FinalAnswerSelector(dspy.Signature):
    """Select final answer based on structured divergence resolutions."""
    
    question: str = dspy.InputField()
    options: Dict[str, str] = dspy.InputField()
    divergence_resolutions: List[Dict[str, Any]] = dspy.InputField(
        desc="List of resolved divergence points with level information"
    )
    option_claim_trees: Dict[str, List[Dict]] = dspy.InputField(
        desc="Complete claim trees for each option"
    )
    
    selected_answer: str = dspy.OutputField(desc="A/B/C/D/E")
    confidence_score: float = dspy.OutputField()
    winning_reasoning_path: List[str] = dspy.OutputField(
        desc="The reasoning path that led to the selected answer"
    )
    key_divergences_favoring_answer: List[str] = dspy.OutputField(
        desc="Key divergence points that favored the selected answer"
    )
    level_analysis: Dict[str, str] = dspy.OutputField(
        desc="Analysis of how each hierarchy level contributed to the decision"
    )

# ============= Enhanced Visualization Functions =============

def visualize_enhanced_option_trees(option_trees: Dict[str, List[Dict]], option_analyses: Dict[str, Dict]):
    """Enhanced visualization of option trees with better structure."""
    print("\n🌳 ENHANCED OPTION REASONING TREES:")
    print("=" * 70)
    
    for option_letter, claims in option_trees.items():
        analysis = option_analyses[option_letter]
        print(f"\n📋 OPTION {option_letter} DETAILED TREE:")
        print("-" * 50)
        print(f"Clinical Context: {analysis['clinical_context']}")
        print(f"Pathophysiology: {analysis['pathophysiology_explanation'][:100]}...")
        
        # Group by hierarchy level
        levels = defaultdict(list)
        for claim in claims:
            level = claim.get('hierarchy_level', 1)
            levels[level].append(claim)
        
        level_names = {
            1: "🔸 Level 1: Basic Facts",
            2: "🔹 Level 2: Physiological Context", 
            3: "🔶 Level 3: Pathophysiological Mechanisms",
            4: "🔷 Level 4: Clinical Manifestations",
            5: "⭐ Level 5: Answer Justification"
        }
        
        for level in sorted(levels.keys()):
            if level > 0:
                print(f"\n{level_names.get(level, f'Level {level}')}:")
                for claim in levels[level]:
                    status_icon = "✅" if claim.get('truth_status') == 'VERIFIED' else "⚠️"
                    relevance_icon = "🎯" if claim.get('clinical_relevance') == 'RELEVANT' else "📍"
                    print(f"    {status_icon}{relevance_icon} {claim['claim_id']}: {claim['statement'][:80]}...")
                    if claim.get('depends_on'):
                        print(f"      ← Depends: {claim['depends_on']}")

def visualize_level_divergences(level_divergences: Dict[int, List[Dict]]):
    """Visualize divergences organized by hierarchy level."""
    print("\n📊 LEVEL-BASED DIVERGENCE ANALYSIS:")
    print("=" * 60)
    
    level_names = {
        1: "🔸 Level 1: Basic Facts Divergences",
        2: "🔹 Level 2: Physiological Context Divergences", 
        3: "🔶 Level 3: Mechanism Divergences",
        4: "🔷 Level 4: Clinical Manifestation Divergences",
        5: "⭐ Level 5: Answer Justification Divergences"
    }
    
    for level in sorted(level_divergences.keys()):
        divergences = level_divergences[level]
        if divergences:
            print(f"\n{level_names.get(level, f'Level {level} Divergences')}:")
            print("-" * 50)
            
            for div in divergences:
                critical_icon = "🔥" if div.get('critical_for_answer', False) else "⚠️"
                print(f"\n{critical_icon} {div['divergence_id']} ({div['divergence_type']}):")
                print(f"  Options: {div['options']}")
                print(f"  Description: {div['description']}")
                print(f"  Claim Pairs: {len(div['claim_pairs'])} pairs")

def visualize_structured_resolutions(divergence_resolutions: Dict[str, Any], level_scores: Dict[str, Dict[int, float]]):
    """Visualize structured divergence resolutions with level analysis."""
    print("\n🏛️ STRUCTURED DIVERGENCE RESOLUTIONS:")
    print("=" * 60)
    
    # Group by level
    level_resolutions = defaultdict(list)
    for div_id, resolution in divergence_resolutions.items():
        level = resolution.level_weight if hasattr(resolution, 'level_weight') else 1
        level_resolutions[level].append((div_id, resolution))
    
    for level in sorted(level_resolutions.keys()):
        resolutions = level_resolutions[level]
        if resolutions:
            print(f"\nLevel {level} Resolutions:")
            print("-" * 30)
            
            for div_id, resolution in resolutions:
                print(f"\n{div_id}:")
                print(f"  🏆 Winner: Option {resolution.winning_option}")
                print(f"  📊 Confidence: {resolution.confidence:.2f}")
                print(f"  ⚖️ Level Weight: {resolution.level_weight:.2f}")
                print(f"  📝 Impact: {resolution.divergence_impact}")
                print(f"  💡 Reasoning: {resolution.reasoning[:100]}...")
    
    print(f"\n📈 LEVEL SCORE BREAKDOWN:")
    print("-" * 40)
    for option, scores in level_scores.items():
        if any(scores.values()):
            total = sum(scores.values())
            print(f"\nOption {option} (Total: {total:.2f}):")
            for level, score in sorted(scores.items()):
                if score > 0:
                    print(f"  Level {level}: {score:.2f}")

def visualize_enhanced_comparative_summary(result: Dict):
    """Enhanced visualization of comparative reasoning summary."""
    print("\n🎯 ENHANCED COMPARATIVE REASONING SUMMARY:")
    print("=" * 70)
    
    print(f"🏆 Selected Answer: {result['answer']}")
    print(f"📊 Confidence: {result['confidence']:.2f}")
    print(f"🔬 Method: {result['reasoning_method']}")
    
    # Enhanced statistics
    total_claims = sum(len(claims) for claims in result['option_trees'].values())
    total_comparisons = len(result['claim_comparisons'])
    total_divergences = sum(len(divs) for divs in result['level_divergences'].values())
    
    print(f"\n📈 ENHANCED ANALYSIS STATISTICS:")
    print(f"  • Total Claims Generated: {total_claims}")
    print(f"  • Pairwise Comparisons: {total_comparisons}")
    print(f"  • Divergences by Level: {total_divergences}")
    print(f"  • Critical Resolutions: {len(result['divergence_resolutions'])}")
    
    # Level breakdown
    print(f"\n📊 Divergences by Hierarchy Level:")
    for level, divs in result['level_divergences'].items():
        critical_count = sum(1 for d in divs if d.get('critical_for_answer', False))
        print(f"  Level {level}: {len(divs)} total ({critical_count} critical)")
    
    # Final reasoning path
    if result.get('final_selection') and hasattr(result['final_selection'], 'winning_reasoning_path'):
        print(f"\n🎖️ Winning Reasoning Path:")
        for i, step in enumerate(result['final_selection'].winning_reasoning_path, 1):
            print(f"  {i}. {step}") 