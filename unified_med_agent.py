import dspy
import json
import os
from dotenv import load_dotenv
from typing import Dict, List
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

# Data structures for claims
@dataclass
class MedicalClaim:
    claim_id: str
    claim_type: str  # ANATOMICAL_FACT, DIAGNOSTIC_CRITERIA, etc.
    statement: str
    confidence: str  # HIGH, MODERATE, LOW
    verification_source: str
    depends_on: List[str]

@dataclass
class VerificationResult:
    claim_id: str
    status: str  # VERIFIED, PARTIALLY_VERIFIED, UNVERIFIED, CONTRADICTED
    evidence_quality: str  # A, B, C, D, F
    source: str
    date: str
    notes: str

# Load environment variables
load_dotenv()

# Configure DSPy with fallback to local vLLM
def configure_dspy(model_name="gpt-4o-mini", use_local=False):
    """Configure DSPy with OpenAI or local vLLM server"""
    if use_local or not os.getenv("OPENAI_API_KEY"):
        # Use local vLLM server (DeepSeek R1)
        lm = dspy.LM(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            api_base="http://localhost:8000/v1",
            api_key="EMPTY"
        )
    else:
        # Use OpenAI
        lm = dspy.LM(f'openai/{model_name}', cache=False)
    
    dspy.configure(lm=lm, temperature=0, seed=42, top_p=0)
    return lm

# Core Medical Agent Signatures
class MedAgent_Simple(dspy.Signature):
    """You are a medical expert. Answer the following question based on medical guidelines. Stick to the most recent medical guidelines."""
    question: str = dspy.InputField()
    options: dict[str, str] = dspy.InputField()
    answer: str = dspy.OutputField(desc="The key of the correct option. For example 'B' or 'A'.")

class MedAgent_Teacher(dspy.Signature):
    """You are a medical expert and a professor. You are making educational content for medical students.
    For a medical question, you will return a list of 3 excerpts from the most recent medical guidelines that are essential to answer the question.
    Each returned medical guideline excerpt should be necessary to answer the question."""
    
    question: str = dspy.InputField()
    guidelines: list[str] = dspy.OutputField()

class MedAgent_Student(dspy.Signature):
    """You are a medical expert. 
    Answer the following question based on the provided medical guidelines. Stick to the guidelines. 
    Your answer should be justifiable directly from the guidelines."""
    
    question: str = dspy.InputField()
    options: dict[str, str] = dspy.InputField()
    guidelines: list[str] = dspy.InputField()
    answer: str = dspy.OutputField(desc="The key of the correct option. For example 'B' or 'A'.")

# Advanced Planning and Reasoning Signatures
class MedAgent_Planner(dspy.Signature):
    """You are a medical expert and a professor. You are making educational content for medical students.
    Given a medical question and for each possible option, outline a set of reasoning plan steps that result in that option being chosen.
    Your goal is to test the students for choosing the right set, so each set of plan steps should convincingly result in the respective option.
    Only come up with the outline of the steps, avoid explaining the reasoning for each step."""
    
    question: str = dspy.InputField()
    options: dict[str, str] = dspy.InputField()
    reasoning_steps: dict[str, list] = dspy.OutputField()

class MedAgent_MG_Fetcher(dspy.Signature):
    """You are a medical expert, specifically knowledgable in medical guidelines.
    A question, the correct final answer, and a plan for reasoning that results in the final answer is given to you.
    Your job is, for each plan step, to fetch and write an excerpt from a medical guideline that is needed to carry out that reasoning plan step."""

    question: str = dspy.InputField()
    final_answer: str = dspy.InputField()
    reasoning_plan_steps: list[str] = dspy.InputField()
    guidelines: dict[str, str] = dspy.OutputField(desc="A guideline excerpt needed to carry out each reasoning plan step")

class MedAgent_Cited_Reasoner(dspy.Signature):
    """You are a medical expert. You are given a medical question, the answer, a plan for reasoning, and a guideline excerpt supplementing each step.
    Your job is to follow the reasoning plan and reason step by step, using the information from the guidelines.
    For each reasoning step, you should cite specific parts of the given guidelines. Do not use any additional knowledge beyond the guidelines."""
    
    question: str = dspy.InputField()
    final_answer: str = dspy.InputField()
    reasoning_plan_steps: list[str] = dspy.InputField()
    guidelines: dict[str, str] = dspy.InputField()
    reasoning: str = dspy.OutputField(desc="Step by step reasoning, citing guidelines and sticking to them, until reaching the final answer.")

class MedAgent_Ranker(dspy.Signature):
    """You are a medical expert. Given a medical question and a list of step-by-step reasonings, 
    rank the reasonings from most sound, guideline-grounded, and matching the question information to the least.
    Use the letter of the reasonings in order, for example if A is the most plausible reasoning and D is the least, the answer should be: A B C D"""
    
    question: str = dspy.InputField()
    reasonings: dict[str, str] = dspy.InputField()
    ranked_reasonings: list[str] = dspy.OutputField()

# Enhanced Medical Guideline Ranking Signatures
class MedAgent_Claimer(dspy.Signature):
    """You are a medical expert.
    Given a medical question and the correct answer, generate claims that consider:
    1. The specific clinical scenario and timing (acute vs discharge vs long-term)
    2. All relevant patient factors mentioned in the question
    3. The exact clinical context being asked about
    
    Each claim should be contextually appropriate for the specific question being asked."""
    question: str = dspy.InputField()
    correct_answer: str = dspy.InputField()
    claims: list[str] = dspy.OutputField()

class MedAgent_MG_Score_List(dspy.Signature):
    """You are a medical expert.
    Given a list of medical claims in the context of a medical question, return a Likert score (1-5) for each claim showing how aligned that claim is with current medical guidelines.
    Consider the specific context of the question when scoring."""
    question: str = dspy.InputField()
    claims: list[str] = dspy.InputField()
    scores: list[int] = dspy.OutputField()

class MedAgent_LogicalConsistency(dspy.Signature):
    """You are a medical expert. 
    Analyze a list of medical claims for logical consistency and internal contradictions.
    Return a consistency score (1-5) and identify any contradictory claim pairs."""
    question: str = dspy.InputField()
    claims: list[str] = dspy.InputField()
    consistency_score: int = dspy.OutputField()
    contradictions: list[str] = dspy.OutputField()

class MedAgent_CriticalPath(dspy.Signature):
    """You are a medical expert.
    Given a medical question and reasoning claims, identify which claims are CRITICAL for reaching the conclusion.
    Return indices of claims that are essential vs. supportive."""
    question: str = dspy.InputField()
    claims: list[str] = dspy.InputField()
    critical_indices: list[int] = dspy.OutputField()
    supportive_indices: list[int] = dspy.OutputField()

class MedAgent_ClaimDependency(dspy.Signature):
    """You are a medical expert.
    Analyze the logical flow between claims. Return dependency relationships and identify if any claim invalidates downstream reasoning."""
    question: str = dspy.InputField()
    claims: list[str] = dspy.InputField()
    dependencies: list[str] = dspy.OutputField()  # Format: "claim_i -> claim_j"
    error_propagation: list[int] = dspy.OutputField()  # Indices of claims that invalidate downstream

class MedAgent_ClaimValidator(dspy.Signature):
    """You are a medical expert. Analyze if this reasoning chain has any major logical flaws or contradictions that would invalidate the conclusion.
    Focus on: internal contradictions, impossible physiological sequences, or claims that contradict established medical facts."""
    question: str = dspy.InputField()
    claims: list[str] = dspy.InputField()
    has_fatal_flaw: bool = dspy.OutputField()
    flaw_description: str = dspy.OutputField()

# New Planning and Guideline-Based Signatures
class MedAgent_Planner_2(dspy.Signature):
    """You are a helpful medical expert. Given a question, your job is to lay out a high-level step-by-step plan to answer it. 
    Each step should be high-level and brief.
    The goal of the plan is to specifically answer the question.
    Do not make any assumptions about the right answer. This plan should be general to support any final answer."""

    question: str = dspy.InputField()
    plan: list[str] = dspy.OutputField()

class MedAgent_Planner_3(dspy.Signature):
    """You are a helpful medical expert. Given a question, your job is to lay out a high-level step-by-step plan to answer it. 
    Each step should be high-level and brief.
    The goal of the plan is to specifically answer the question. Focus on the question's purpose and set up the plan to answer the question.
    Don't go beyond what the question is asking.
    Do not make any assumptions about the right answer. This plan should be general to support any final answer."""

    question: str = dspy.InputField()
    options: dict[str, str] = dspy.InputField()
    plan: list[str] = dspy.OutputField()

class MedAgent_Plan_Refiner(dspy.Signature):
    """You are a helpful medical expert. You are given a medical question and a plan to reason through it to reach a final answer. 
    You are also given the correct final answer, your job is to refine the plan in order to be able to reach the correct final answer.
    If the plan is currently good enough to reach that answer, return empty.
    Only do the minimum necessary refining for eacch plan step.
    For each step, return the refined plan step."""

    question: str = dspy.InputField()
    plan: list[str] = dspy.InputField()
    correct_answer: str = dspy.InputField()
    refined_plan: dict[str, str] = dspy.OutputField()

class MedAgent_ContextualClaim(dspy.Signature):
    """Generate medical claims with explicit clinical context.
    Each claim should specify: WHO (patient type), WHEN (clinical timing), WHAT (intervention/finding)"""
    question: str = dspy.InputField()
    answer_option: str = dspy.InputField()
    contextual_claims: list[dict] = dspy.OutputField(desc="List of {'context': str, 'claim': str}")

class MedAgent_GuidelineRetriever(dspy.Signature):
    """For a given medical claim in context, generate the relevant medical guideline or evidence.
    Be specific about which guideline, what it says, and any conditions/contexts where it applies."""
    context: str = dspy.InputField()
    claim: str = dspy.InputField()
    guideline_source: str = dspy.OutputField(desc="e.g., 'AHA/ACC 2023 ACS Guidelines'")
    guideline_text: str = dspy.OutputField(desc="Relevant guideline statement")
    recommendation_class: str = dspy.OutputField(desc="Class I/IIa/IIb/III")
    evidence_level: str = dspy.OutputField(desc="Level A/B/C")

class MedAgent_GuidelineAlignment(dspy.Signature):
    """Evaluate if a medical claim aligns with the retrieved guideline in the given context.
    Consider: Does the guideline support this claim? Are there contraindications? Context match?"""
    context: str = dspy.InputField()
    claim: str = dspy.InputField()
    guideline_text: str = dspy.InputField()
    recommendation_class: str = dspy.InputField()
    alignment_score: int = dspy.OutputField(desc="1-5 score for alignment")
    reasoning: str = dspy.OutputField(desc="Why this score was given")

# Verifiable Medical Reasoning Signatures
class MedicalReasoner(dspy.Signature):
    """Primary medical reasoning agent that provides step-by-step analysis."""
    
    patient_presentation: str = dspy.InputField(
        desc="Patient symptoms, history, and vital signs"
    )
    
    reasoning_steps: List[str] = dspy.OutputField(
        desc="Step-by-step medical reasoning, each step as a complete thought"
    )
    differential_diagnosis: List[str] = dspy.OutputField(
        desc="List of possible diagnoses in order of likelihood"
    )
    primary_assessment: str = dspy.OutputField(
        desc="Primary diagnosis and immediate recommendations"
    )

class ClaimDecomposer(dspy.Signature):
    """Decomposes medical reasoning into atomic, verifiable claims."""
    
    reasoning_text: str = dspy.InputField(
        desc="Medical reasoning text to decompose"
    )
    
    claims: List[Dict[str, str]] = dspy.OutputField(
        desc="List of atomic claims with format: {claim_id, type, statement, confidence, source_type, depends_on}"
    )

class ClaimVerifier(dspy.Signature):
    """Verifies individual medical claims against medical literature."""
    
    claim: Dict[str, str] = dspy.InputField(
        desc="Medical claim to verify"
    )
    medical_context: str = dspy.InputField(
        desc="Additional context about the patient case"
    )
    
    verification_status: str = dspy.OutputField(
        desc="VERIFIED, PARTIALLY_VERIFIED, UNVERIFIED, or CONTRADICTED"
    )
    evidence_quality: str = dspy.OutputField(
        desc="Evidence quality rating: A (Guidelines/Systematic Review), B (RCT), C (Observational), D (Expert Opinion), F (Cannot Verify)"
    )
    source_citation: str = dspy.OutputField(
        desc="Specific citation for verification"
    )
    verification_notes: str = dspy.OutputField(
        desc="Additional notes about verification or contradicting evidence"
    )

class ReasoningSynthesizer(dspy.Signature):
    """Synthesizes verified claims into trustworthy medical assessment."""
    
    original_reasoning: List[str] = dspy.InputField(
        desc="Original reasoning steps"
    )
    verified_claims: List[Dict[str, str]] = dspy.InputField(
        desc="Claims with verification results"
    )
    
    verified_assessment: str = dspy.OutputField(
        desc="Synthesized assessment with confidence indicators and evidence levels"
    )
    confidence_summary: Dict[str, float] = dspy.OutputField(
        desc="Overall confidence scores for different aspects of the assessment"
    )
    key_uncertainties: List[str] = dspy.OutputField(
        desc="Important uncertainties or limitations in the assessment"
    )

class UncertaintyQuantifier(dspy.Signature):
    """Quantifies and communicates uncertainty in medical claims."""
    
    claim: Dict[str, str] = dspy.InputField()
    verification_result: Dict[str, str] = dspy.InputField()
    
    uncertainty_level: float = dspy.OutputField(
        desc="Numerical uncertainty 0.0 (certain) to 1.0 (highly uncertain)"
    )
    uncertainty_explanation: str = dspy.OutputField(
        desc="Human-readable explanation of what creates the uncertainty"
    )
    additional_data_needed: List[str] = dspy.OutputField(
        desc="What additional information would reduce uncertainty"
    )

# Medical MCQ Verifiable Reasoning System
from typing import Optional
from enum import Enum

# ============= Constants and Enums =============
# IMPORTANT: These constants ensure consistent output format across all model responses
# Without normalization, the model might output "verified", "Verified.", "VERIFIED!", etc.
# which makes comparison and downstream processing difficult.

class VerificationStatus(Enum):
    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED" 
    UNVERIFIED = "UNVERIFIED"
    CONTRADICTED = "CONTRADICTED"

class EvidenceQuality(Enum):
    A = "A"  # Guidelines/Systematic Review
    B = "B"  # RCT/Large Studies
    C = "C"  # Observational Studies
    D = "D"  # Expert Opinion
    F = "F"  # Cannot Verify

class ConfidenceLevel(Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"

# Valid values for constraints
VALID_VERIFICATION_STATUS = ["VERIFIED", "PARTIALLY_VERIFIED", "UNVERIFIED", "CONTRADICTED"]
VALID_EVIDENCE_QUALITY = ["A", "B", "C", "D", "F"]
VALID_CONFIDENCE_LEVELS = ["HIGH", "MODERATE", "LOW"]

# ============= Helper Functions =============

def normalize_verification_status(status: str) -> str:
    """Normalize verification status to ensure consistent format."""
    if not status:
        return "UNVERIFIED"
    
    # Remove any extra characters and normalize
    normalized = status.upper().strip().rstrip('.').rstrip('!')
    
    # Map common variations to standard values
    status_map = {
        'VERIFIED': 'VERIFIED',
        'VERIFY': 'VERIFIED',
        'PARTIAL': 'PARTIALLY_VERIFIED',
        'PARTIALLY': 'PARTIALLY_VERIFIED',
        'PARTIALLY_VERIFIED': 'PARTIALLY_VERIFIED',
        'PARTIAL_VERIFIED': 'PARTIALLY_VERIFIED',
        'UNVERIFIED': 'UNVERIFIED',
        'NOT_VERIFIED': 'UNVERIFIED',
        'CANNOT_VERIFY': 'UNVERIFIED',
        'CONTRADICTED': 'CONTRADICTED',
        'CONTRADICT': 'CONTRADICTED',
        'CONTRADICTS': 'CONTRADICTED',
        'FALSE': 'CONTRADICTED'
    }
    
    return status_map.get(normalized, 'UNVERIFIED')

def normalize_evidence_quality(quality: str) -> str:
    """Normalize evidence quality to ensure consistent format."""
    if not quality:
        return "F"
    
    normalized = quality.upper().strip()
    
    # Ensure it's one of the valid values
    if normalized in VALID_EVIDENCE_QUALITY:
        return normalized
    
    # Handle common variations
    if normalized in ['HIGH', 'EXCELLENT']:
        return 'A'
    elif normalized in ['GOOD', 'MODERATE']:
        return 'B'
    elif normalized in ['LOW', 'WEAK']:
        return 'C'
    elif normalized in ['EXPERT', 'OPINION']:
        return 'D'
    else:
        return 'F'

def normalize_confidence_level(confidence: str) -> str:
    """Normalize confidence level to ensure consistent format."""
    if not confidence:
        return "MODERATE"
    
    normalized = confidence.upper().strip()
    
    # Map variations to standard values
    confidence_map = {
        'HIGH': 'HIGH',
        'VERY_HIGH': 'HIGH',
        'STRONG': 'HIGH',
        'MODERATE': 'MODERATE',
        'MEDIUM': 'MODERATE',
        'LOW': 'LOW',
        'WEAK': 'LOW',
        'VERY_LOW': 'LOW'
    }
    
    return confidence_map.get(normalized, 'MODERATE')

def parse_contradicts_options(contradicts_str: str) -> List[str]:
    """Convert comma-separated string to list of options."""
    if not contradicts_str or contradicts_str.lower() == 'none':
        return []
    return [opt.strip() for opt in contradicts_str.split(',') if opt.strip()]

def format_contradicts_options(options_list: List[str]) -> str:
    """Convert list of options to comma-separated string."""
    if not options_list:
        return "none"
    return ",".join(options_list)

# ============= DSPy Signatures for MCQ =============

class MedicalMCQReasoner(dspy.Signature):
    """Analyzes medical MCQ with step-by-step reasoning."""
    
    question: str = dspy.InputField(
        desc="The medical question stem including patient presentation"
    )
    options: Dict[str, str] = dspy.InputField(
        desc="Answer options as dictionary {A: text, B: text, ...}"
    )
    
    initial_analysis: str = dspy.OutputField(
        desc="Initial understanding of what the question is asking"
    )
    key_findings: List[str] = dspy.OutputField(
        desc="Important clinical findings from the question stem"
    )
    reasoning_steps: List[str] = dspy.OutputField(
        desc="Step-by-step reasoning through the problem"
    )
    option_evaluation: Dict[str, str] = dspy.OutputField(
        desc="Evaluation of each option with reasoning"
    )

class MCQClaimDecomposer(dspy.Signature):
    """Decomposes MCQ reasoning into verifiable claims."""
    
    question: str = dspy.InputField()
    options: Dict[str, str] = dspy.InputField()
    reasoning_steps: List[str] = dspy.InputField()
    option_evaluation: Dict[str, str] = dspy.InputField()
    
    claims: List[Dict] = dspy.OutputField(
        desc="List of claims with format: {claim_id: str, type: str, statement: str, confidence: str, supports_option: str, contradicts_options: str (comma-separated)}"
    )

class MCQClaimVerifier(dspy.Signature):
    """Verifies claims specifically for MCQ context."""
    
    claim: Dict[str, str] = dspy.InputField()
    question_context: str = dspy.InputField()
    relevant_options: Dict[str, str] = dspy.InputField()
    
    verification_status: str = dspy.OutputField()
    evidence_quality: str = dspy.OutputField()
    source_citation: str = dspy.OutputField()
    impact_on_options: Dict[str, str] = dspy.OutputField(
        desc="How verification affects each option's likelihood"
    )

class MCQAnswerSelector(dspy.Signature):
    """Selects final answer based on verified claims."""
    
    question: str = dspy.InputField()
    options: Dict[str, str] = dspy.InputField()
    verified_claims: List[Dict[str, str]] = dspy.InputField()
    option_evaluation: Dict[str, str] = dspy.InputField()
    
    answer: str = dspy.OutputField(
        desc="The selected answer key (A, B, C, D, or E)"
    )
    confidence_score: float = dspy.OutputField(
        desc="Confidence in the answer (0.0 to 1.0)"
    )
    answer_justification: str = dspy.OutputField(
        desc="Final justification for the selected answer"
    )

class MCQReasoningSynthesizer(dspy.Signature):
    """Creates verifiable explanation for MCQ answer."""
    
    question: str = dspy.InputField()
    options: Dict[str, str] = dspy.InputField()
    selected_answer: str = dspy.InputField()
    verified_claims: List[Dict[str, str]] = dspy.InputField()
    
    verified_explanation: str = dspy.OutputField(
        desc="Complete explanation with confidence indicators"
    )
    supporting_evidence: List[str] = dspy.OutputField(
        desc="Key verified facts supporting the answer"
    )
    eliminated_options: Dict[str, str] = dspy.OutputField(
        desc="Why each incorrect option was eliminated"
    )

# Medical MCQ Verifiable Reasoning Modules

class VerifiableMedicalMCQSolver(dspy.Module):
    """Complete pipeline for verifiable medical MCQ answering."""
    
    def __init__(self):
        super().__init__()
        self.reasoner = dspy.Predict(MedicalMCQReasoner)
        self.decomposer = dspy.Predict(MCQClaimDecomposer)
        self.verifier = dspy.Predict(MCQClaimVerifier)
        self.selector = dspy.Predict(MCQAnswerSelector)
        self.synthesizer = dspy.Predict(MCQReasoningSynthesizer)
    
    def forward(self, question: str, options: Dict[str, str]) -> Dict:
        # Step 1: Initial reasoning
        reasoning = self.reasoner(question=question, options=options)
        
        # Step 2: Decompose into claims
        claims = self.decomposer(
            question=question,
            options=options,
            reasoning_steps=reasoning.reasoning_steps,
            option_evaluation=reasoning.option_evaluation
        )
        
        # Step 3: Verify each claim
        verified_claims = []
        for claim in claims.claims:
            # Parse contradicts_options if it's a string
            if isinstance(claim.get('contradicts_options'), str):
                claim['contradicts_options_list'] = [opt.strip() for opt in claim['contradicts_options'].split(',') if opt.strip()]
            else:
                claim['contradicts_options_list'] = []
            
            # Normalize confidence level in the claim
            if 'confidence' in claim:
                claim['confidence'] = normalize_confidence_level(claim['confidence'])
                
            verification = self.verifier(
                claim=claim,
                question_context=question,
                relevant_options=options
            )
            
            # Normalize verification outputs
            normalized_status = normalize_verification_status(verification.verification_status)
            normalized_quality = normalize_evidence_quality(verification.evidence_quality)
            
            verified_claim = {
                **claim,
                'verification_status': normalized_status,
                'evidence_quality': normalized_quality,
                'source': verification.source_citation,
                'impact_on_options': verification.impact_on_options
            }
            verified_claims.append(verified_claim)
        
        # Step 4: Select answer based on verified claims
        answer_selection = self.selector(
            question=question,
            options=options,
            verified_claims=verified_claims,
            option_evaluation=reasoning.option_evaluation
        )
        
        # Step 5: Synthesize verifiable explanation
        synthesis = self.synthesizer(
            question=question,
            options=options,
            selected_answer=answer_selection.answer,
            verified_claims=verified_claims
        )
        
        return {
            'answer': answer_selection.answer,
            'confidence': answer_selection.confidence_score,
            'initial_reasoning': reasoning,
            'claims': claims.claims,
            'verified_claims': verified_claims,
            'answer_justification': answer_selection.answer_justification,
            'verified_explanation': synthesis.verified_explanation,
            'supporting_evidence': synthesis.supporting_evidence,
            'eliminated_options': synthesis.eliminated_options
        }
    
    def answer_question(self, question: str, options: dict) -> str:
        """Adapt to standard interface for compatibility"""
        result = self.forward(question, options)
        return result['answer']

class DifferentialDiagnosisModule(dspy.Module):
    """Specialized module for differential diagnosis questions."""
    
    def __init__(self):
        super().__init__()
        self.analyzer = dspy.Predict(self.ClinicalAnalyzer)
        self.ranker = dspy.Predict(self.DiagnosisRanker)
    
    class ClinicalAnalyzer(dspy.Signature):
        """Analyzes clinical presentation systematically."""
        
        question: str = dspy.InputField()
        
        chief_complaint: str = dspy.OutputField()
        key_symptoms: List[str] = dspy.OutputField()
        relevant_history: List[str] = dspy.OutputField()
        physical_findings: List[str] = dspy.OutputField()
        lab_results: List[str] = dspy.OutputField()
    
    class DiagnosisRanker(dspy.Signature):
        """Ranks diagnoses based on clinical findings."""
        
        clinical_findings: Dict[str, List[str]] = dspy.InputField()
        possible_diagnoses: Dict[str, str] = dspy.InputField()
        
        diagnosis_scores: Dict[str, float] = dspy.OutputField(
            desc="Probability scores for each diagnosis option"
        )
        supporting_findings: Dict[str, List[str]] = dspy.OutputField()
        contradicting_findings: Dict[str, List[str]] = dspy.OutputField()
    
    def forward(self, question: str, options: Dict[str, str]) -> Dict:
        # Analyze clinical presentation
        clinical = self.analyzer(question=question)
        
        clinical_findings = {
            'chief_complaint': clinical.chief_complaint,
            'symptoms': clinical.key_symptoms,
            'history': clinical.relevant_history,
            'physical': clinical.physical_findings,
            'labs': clinical.lab_results
        }
        
        # Rank diagnoses
        ranking = self.ranker(
            clinical_findings=clinical_findings,
            possible_diagnoses=options
        )
        
        return ranking
    
    def answer_question(self, question: str, options: dict) -> str:
        """Adapt to standard interface"""
        result = self.forward(question, options)
        # Return option with highest diagnosis score
        best_option = max(result.diagnosis_scores, key=result.diagnosis_scores.get)
        return best_option

class PharmacologyModule(dspy.Module):
    """Specialized module for pharmacology questions."""
    
    def __init__(self):
        super().__init__()
        self.drug_analyzer = dspy.Predict(self.DrugAnalyzer)
        self.interaction_checker = dspy.Predict(self.InteractionChecker)
    
    class DrugAnalyzer(dspy.Signature):
        """Analyzes drug-related questions."""
        
        question: str = dspy.InputField()
        options: Dict[str, str] = dspy.InputField()
        
        drug_class: str = dspy.OutputField()
        mechanism_of_action: str = dspy.OutputField()
        clinical_context: str = dspy.OutputField()
        relevant_side_effects: List[str] = dspy.OutputField()
        contraindications: List[str] = dspy.OutputField()
    
    class InteractionChecker(dspy.Signature):
        """Checks drug interactions and contraindications."""
        
        patient_context: str = dspy.InputField()
        drug_options: Dict[str, str] = dspy.InputField()
        
        interaction_risks: Dict[str, List[str]] = dspy.OutputField()
        contraindication_flags: Dict[str, List[str]] = dspy.OutputField()
        safest_option: str = dspy.OutputField()
    
    def forward(self, question: str, options: Dict[str, str]) -> Dict:
        # Analyze drug aspects
        drug_analysis = self.drug_analyzer(question=question, options=options)
        
        # Check interactions
        interactions = self.interaction_checker(
            patient_context=question,
            drug_options=options
        )
        
        return {
            'drug_analysis': drug_analysis,
            'interactions': interactions,
            'recommended_choice': interactions.safest_option
        }
    
    def answer_question(self, question: str, options: dict) -> str:
        """Adapt to standard interface"""
        result = self.forward(question, options)
        return result['recommended_choice']

# Utility functions
def get_option_letter(options, answer):
    """Convert answer to option letter"""
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    if answer not in options:
        raise ValueError("Answer not found in options.")
    index = options.index(answer)
    return letters[index]

def map_letters_to_options(options):
    """Map option letters to option text"""
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    if len(options) > len(letters):
        raise ValueError("Too many options to assign letters.")
    return {letters[i]: option for i, option in enumerate(options)}

# Parallel Processing Functions
async def make_async_request(session, agent, question, options, request_id):
    """Make an async request to an agent"""
    try:
        # Note: This is a placeholder for async agent calls
        # In practice, you'd need to implement async versions of the agents
        result = agent.answer_question(question, options)
        return {
            "request_id": request_id,
            "question": question,
            "answer": result,
            "success": True
        }
    except Exception as e:
        return {
            "request_id": request_id,
            "question": question,
            "error": str(e),
            "success": False
        }

def make_sync_request(agent, question, options, request_id):
    """Make a synchronous request to an agent"""
    try:
        result = agent.answer_question(question, options)
        return {
            "request_id": request_id,
            "question": question,
            "answer": result,
            "success": True
        }
    except Exception as e:
        return {
            "request_id": request_id,
            "question": question,
            "error": str(e),
            "success": False
        }

def run_parallel_requests(agent, questions_and_options, max_workers=4):
    """Run multiple requests in parallel using threads"""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all requests
        future_to_request = {
            executor.submit(make_sync_request, agent, item[0], item[1], i): i 
            for i, item in enumerate(questions_and_options)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_request):
            results.append(future.result())
    
    return sorted(results, key=lambda x: x["request_id"])

# Simple Medical Agent class
class SimpleMedicalAgent:
    """Basic medical question answering agent"""
    
    def __init__(self, use_chain_of_thought=True):
        if use_chain_of_thought:
            self.predictor = dspy.ChainOfThought(MedAgent_Simple)
        else:
            self.predictor = dspy.Predict(MedAgent_Simple)
    
    def answer_question(self, question: str, options: dict) -> str:
        """Answer a medical question with given options"""
        result = self.predictor(question=question, options=options)
        return result.answer

# Teacher-Student Framework Modules
class MedAgent_Guideline_Simple_Predict_Predict(dspy.Module):
    """Teacher uses Predict, Student uses Predict"""
    def __init__(self):
        super().__init__()
        self.teacher = dspy.Predict(MedAgent_Teacher)
        self.student = dspy.Predict(MedAgent_Student)

    def forward(self, question, options):
        guidelines = self.teacher(question=question)
        answer = self.student(question=question, options=options, guidelines=guidelines)
        return answer

class MedAgent_Guideline_Simple_CoT_Predict(dspy.Module):
    """Teacher uses ChainOfThought, Student uses Predict"""
    def __init__(self):
        super().__init__()
        self.teacher = dspy.ChainOfThought(MedAgent_Teacher)
        self.student = dspy.Predict(MedAgent_Student)

    def forward(self, question, options):
        guidelines = self.teacher(question=question)
        answer = self.student(question=question, options=options, guidelines=guidelines)
        return answer

class MedAgent_Guideline_Simple_Predict_CoT(dspy.Module):
    """Teacher uses Predict, Student uses ChainOfThought"""
    def __init__(self):
        super().__init__()
        self.teacher = dspy.Predict(MedAgent_Teacher)
        self.student = dspy.ChainOfThought(MedAgent_Student)

    def forward(self, question, options):
        guidelines = self.teacher(question=question)
        answer = self.student(question=question, options=options, guidelines=guidelines)
        return answer

class MedAgent_Guideline_Simple_CoT_CoT(dspy.Module):
    """Teacher uses ChainOfThought, Student uses ChainOfThought"""
    def __init__(self):
        super().__init__()
        self.teacher = dspy.ChainOfThought(MedAgent_Teacher)
        self.student = dspy.ChainOfThought(MedAgent_Student)

    def forward(self, question, options):
        guidelines = self.teacher(question=question)
        answer = self.student(question=question, options=options, guidelines=guidelines)
        return answer

# Advanced Planning-based Agent
class AdvancedPlanningAgent(dspy.Module):
    """Advanced agent using planning, guideline fetching, and reasoning"""
    
    def __init__(self):
        super().__init__()
        self.planner = dspy.Predict(MedAgent_Planner)
        self.fetcher = dspy.Predict(MedAgent_MG_Fetcher)
        self.reasoner = dspy.Predict(MedAgent_Cited_Reasoner)
        self.ranker = dspy.Predict(MedAgent_Ranker)
    
    def forward(self, question, options):
        # Step 1: Create reasoning plans for each option
        plans = self.planner(question=question, options=options)
        
        # Step 2: For each option, fetch guidelines and create reasoning
        reasonings = {}
        for option_key, option_text in options.items():
            if option_key in plans.reasoning_steps:
                # Fetch guidelines for this option's reasoning plan
                guidelines = self.fetcher(
                    question=question,
                    final_answer=option_text,
                    reasoning_plan_steps=plans.reasoning_steps[option_key]
                )
                
                # Generate detailed reasoning using guidelines
                reasoning = self.reasoner(
                    question=question,
                    final_answer=option_text,
                    reasoning_plan_steps=plans.reasoning_steps[option_key],
                    guidelines=guidelines.guidelines
                )
                
                reasonings[option_key] = reasoning.reasoning
        
        # Step 3: Rank the reasonings to find the best one
        ranking = self.ranker(question=question, reasonings=reasonings)
        
        # Return the top-ranked option
        if ranking.ranked_reasonings:
            return dspy.Prediction(answer=ranking.ranked_reasonings[0])
        else:
            # Fallback to first option if ranking fails
            return dspy.Prediction(answer=list(options.keys())[0])

# Enhanced Medical Guideline Ranking Modules
class MedAgent_MG_Ranking_Enhanced(dspy.Module):
    """Enhanced medical guideline ranking with claim-based validation"""
    
    def __init__(self):
        super().__init__()
        self.medagent_claimer = dspy.Predict(MedAgent_Claimer)
        self.medagent_mg_score_list = dspy.ChainOfThought(MedAgent_MG_Score_List)
        self.logical_consistency = dspy.ChainOfThought(MedAgent_LogicalConsistency)
        self.critical_path = dspy.Predict(MedAgent_CriticalPath)
        self.claim_dependency = dspy.Predict(MedAgent_ClaimDependency)
    
    def calculate_enhanced_score(self, question, claims, scores):
        """Calculate score using multiple validation methods"""
        
        # 1. Get logical consistency
        consistency_result = self.logical_consistency(question=question, claims=claims)
        consistency_score = consistency_result.consistency_score
        contradictions = consistency_result.contradictions
        
        # 2. Get critical path
        critical_result = self.critical_path(question=question, claims=claims)
        critical_indices = critical_result.critical_indices
        
        # 3. Get error propagation
        dependency_result = self.claim_dependency(question=question, claims=claims)
        error_propagation = dependency_result.error_propagation
        
        # 4. Calculate enhanced score
        enhanced_score = self._compute_final_score(
            scores, consistency_score, contradictions, 
            critical_indices, error_propagation
        )
        
        return enhanced_score, {
            'consistency': consistency_score,
            'contradictions': contradictions,
            'critical_indices': critical_indices,
            'error_propagation': error_propagation
        }
    
    def _compute_final_score(self, scores, consistency_score, contradictions, 
                           critical_indices, error_propagation):
        """Multi-modal scoring with error propagation"""
        
        # Base score calculation
        if not scores:
            return 0
            
        # 1. Check for error propagation (fatal errors)
        if error_propagation:
            min_error_score = min(scores[i] for i in error_propagation if i < len(scores))
            if min_error_score <= 2:  # Fatal error threshold
                return min_error_score * 0.5  # Heavy penalty
        
        # 2. Check for contradictions (severe penalty)
        if contradictions:
            contradiction_penalty = len(contradictions) * 0.3
        else:
            contradiction_penalty = 0
            
        # 3. Weighted scoring (critical claims matter more)
        if critical_indices:
            critical_scores = [scores[i] for i in critical_indices if i < len(scores)]
            supportive_scores = [scores[i] for i, score in enumerate(scores) 
                               if i not in critical_indices]
            
            if critical_scores:
                # Critical claims: 70% weight, use minimum score
                critical_component = min(critical_scores) * 0.7
                # Supportive claims: 30% weight, use average
                supportive_component = (sum(supportive_scores) / len(supportive_scores) if supportive_scores else 5) * 0.3
                weighted_score = critical_component + supportive_component
            else:
                weighted_score = sum(scores) / len(scores)
        else:
            # Fallback to minimum score if no critical path identified
            weighted_score = min(scores) * 0.6 + (sum(scores) / len(scores)) * 0.4
        
        # 4. Apply consistency modifier
        consistency_modifier = (consistency_score - 3) * 0.2  # -0.4 to +0.4
        
        # 5. Final score calculation
        final_score = weighted_score + consistency_modifier - contradiction_penalty
        
        return max(0, min(5, final_score))  # Clamp to [0,5]
    
    def forward(self, question, options):
        scores_dict = {}
        debug_info = {}
        
        for key, option in options.items():
            # Generate claims
            claims = self.medagent_claimer(question=question, correct_answer=option).claims
            
            # Score individual claims
            scores = self.medagent_mg_score_list(question=question, claims=claims).scores
            
            # Calculate enhanced score
            enhanced_score, debug = self.calculate_enhanced_score(question, claims, scores)
            
            scores_dict[key] = enhanced_score
            debug_info[key] = {
                'claims': claims,
                'individual_scores': scores,
                'enhanced_score': enhanced_score,
                **debug
            }
        
        best_answer = max(scores_dict, key=scores_dict.get)
        
        return dspy.Prediction(
            answer=best_answer,
            scores=scores_dict,
            debug_info=debug_info
        )

class MedAgent_MG_Ranking_Fixed(dspy.Module):
    """Fixed version with fatal flaw detection"""
    
    def __init__(self):
        super().__init__()
        self.medagent_claimer = dspy.Predict(MedAgent_Claimer)
        self.medagent_mg_score_list = dspy.ChainOfThought(MedAgent_MG_Score_List)
        self.claim_validator = dspy.ChainOfThought(MedAgent_ClaimValidator)
    
    def forward(self, question, options):
        scores_dict = {}
        debug_info = {}
        
        for key, option in options.items():
            claims = self.medagent_claimer(question=question, correct_answer=option).claims
            individual_scores = self.medagent_mg_score_list(question=question, claims=claims).scores
            
            # Validate reasoning chain
            validation = self.claim_validator(question=question, claims=claims)
            
            if not individual_scores:
                final_score = 0
            elif validation.has_fatal_flaw:
                # Heavy penalty for fatal flaws
                final_score = min(individual_scores) * 0.3
            else:
                # Use weighted combination: 60% minimum, 40% average
                # This prevents averaging away bad claims while still rewarding overall quality
                min_score = min(individual_scores)
                avg_score = sum(individual_scores) / len(individual_scores)
                final_score = min_score * 0.6 + avg_score * 0.4
            
            scores_dict[key] = final_score
            debug_info[key] = {
                'claims': claims,
                'individual_scores': individual_scores,
                'has_fatal_flaw': validation.has_fatal_flaw,
                'flaw_description': validation.flaw_description,
                'final_score': final_score
            }
        
        return dspy.Prediction(
            answer=max(scores_dict, key=scores_dict.get),
            scores=scores_dict,
            debug_info=debug_info
        )

class MedAgent_MG_Ranking_Simple(dspy.Module):
    """Simpler version focusing on key improvements"""
    
    def __init__(self):
        super().__init__()
        self.medagent_claimer = dspy.Predict(MedAgent_Claimer)
        self.medagent_mg_score_list = dspy.ChainOfThought(MedAgent_MG_Score_List)
        self.logical_consistency = dspy.ChainOfThought(MedAgent_LogicalConsistency)
    
    def forward(self, question, options):
        scores_dict = {}
        
        for key, option in options.items():
            claims = self.medagent_claimer(question=question, correct_answer=option).claims
            scores = self.medagent_mg_score_list(question=question, claims=claims).scores
            
            # Check logical consistency
            consistency_result = self.logical_consistency(question=question, claims=claims)
            consistency_score = consistency_result.consistency_score
            
            # Enhanced scoring: minimum score + consistency bonus/penalty
            if scores:
                base_score = min(scores)  # Use minimum instead of average
                consistency_modifier = (consistency_score - 3) * 0.3
                final_score = base_score + consistency_modifier
                
                # Heavy penalty for contradictions
                if consistency_result.contradictions:
                    final_score *= 0.7
                    
                scores_dict[key] = max(0, min(5, final_score))
            else:
                scores_dict[key] = 0
        
        return dspy.Prediction(answer=max(scores_dict, key=scores_dict.get))

class MedAgent_MG_Ranking_Conservative(dspy.Module):
    """Conservative version with simple but effective scoring"""
    
    def __init__(self):
        super().__init__()
        self.medagent_claimer = dspy.Predict(MedAgent_Claimer)
        self.medagent_mg_score_list = dspy.ChainOfThought(MedAgent_MG_Score_List)
    
    def forward(self, question, options):
        scores_dict = {}
        
        for key, option in options.items():
            claims = self.medagent_claimer(question=question, correct_answer=option).claims
            scores = self.medagent_mg_score_list(question=question, claims=claims).scores
            
            if scores:
                # Simple but effective: 70% minimum score + 30% average
                # This heavily weights the weakest link while still considering overall quality
                min_score = min(scores)
                avg_score = sum(scores) / len(scores)
                final_score = min_score * 0.7 + avg_score * 0.3
                scores_dict[key] = final_score
            else:
                scores_dict[key] = 0
        
        return dspy.Prediction(answer=max(scores_dict, key=scores_dict.get))

class MedAgent_GuidelineBased(dspy.Module):
    """Guideline-based medical agent with contextual claims and evidence alignment"""
    
    def __init__(self):
        super().__init__()
        self.contextual_claimer = dspy.Predict(MedAgent_ContextualClaim)
        self.guideline_retriever = dspy.ChainOfThought(MedAgent_GuidelineRetriever)
        self.alignment_evaluator = dspy.ChainOfThought(MedAgent_GuidelineAlignment)
    
    def forward(self, question, options):
        results = {}
        
        for key, option in options.items():
            # 1. Generate contextual claims
            contextual_claims = self.contextual_claimer(
                question=question, 
                answer_option=option
            ).contextual_claims
            
            # 2. For each claim, get guideline and evaluate alignment
            claim_evaluations = []
            for claim_dict in contextual_claims:
                context = claim_dict['context']
                claim = claim_dict['claim']
                
                # Retrieve guideline
                guideline = self.guideline_retriever(context=context, claim=claim)
                
                # Evaluate alignment
                alignment = self.alignment_evaluator(
                    context=context,
                    claim=claim,
                    guideline_text=guideline.guideline_text,
                    recommendation_class=guideline.recommendation_class
                )
                
                claim_evaluations.append({
                    'context': context,
                    'claim': claim,
                    'guideline_source': guideline.guideline_source,
                    'guideline_text': guideline.guideline_text,
                    'recommendation_class': guideline.recommendation_class,
                    'evidence_level': guideline.evidence_level,
                    'alignment_score': alignment.alignment_score,
                    'reasoning': alignment.reasoning
                })
            
            # 3. Calculate overall score
            overall_score = self._calculate_guideline_score(claim_evaluations)
            
            results[key] = {
                'score': overall_score,
                'evaluations': claim_evaluations
            }
        
        best_option = max(results, key=lambda k: results[k]['score'])
        
        return dspy.Prediction(
            answer=best_option,
            results=results
        )
    
    def _calculate_guideline_score(self, evaluations):
        """Weight by guideline strength and evidence quality"""
        weighted_scores = []
        
        for eval in evaluations:
            base_score = eval['alignment_score']
            
            # Weight by recommendation class
            class_weights = {
                'Class I': 1.0,      # Strong recommendation
                'Class IIa': 0.8,    # Reasonable to do
                'Class IIb': 0.6,    # May be reasonable
                'Class III': 0.2     # Not recommended
            }
            
            # Weight by evidence level
            evidence_weights = {
                'Level A': 1.0,      # High-quality evidence
                'Level B': 0.8,      # Moderate-quality evidence  
                'Level C': 0.6       # Limited evidence
            }
            
            class_weight = class_weights.get(eval['recommendation_class'], 0.5)
            evidence_weight = evidence_weights.get(eval['evidence_level'], 0.5)
            
            weighted_score = base_score * class_weight * evidence_weight
            weighted_scores.append(weighted_score)
        
        return sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0

# Simple Medical Reasoning Agent
class SimpleMedicalReasoningAgent(dspy.Module):
    """Simple agent using the MedicalReasoner signature for patient assessment"""
    
    def __init__(self):
        super().__init__()
        self.reasoner = dspy.Predict(MedicalReasoner)
    
    def forward(self, patient_presentation: str):
        """Analyze patient presentation and provide medical reasoning"""
        result = self.reasoner(patient_presentation=patient_presentation)
        return result
    
    def answer_question(self, question: str, options: dict) -> str:
        """Adapt to the standard question-answering interface"""
        # Use the question as patient presentation
        result = self.reasoner(patient_presentation=question)
        
        # Simple heuristic: choose option that best matches primary assessment
        primary = result.primary_assessment.lower()
        best_match = None
        best_score = 0
        
        for key, option in options.items():
            # Count word matches between primary assessment and option
            option_words = set(option.lower().split())
            primary_words = set(primary.split())
            matches = len(option_words.intersection(primary_words))
            
            if matches > best_score:
                best_score = matches
                best_match = key
        
        return best_match if best_match else list(options.keys())[0]

# Claim Decomposing Agent
class ClaimDecomposingAgent(dspy.Module):
    """Agent that decomposes medical reasoning into verifiable claims"""
    
    def __init__(self):
        super().__init__()
        self.reasoner = dspy.Predict(MedicalReasoner)
        self.decomposer = dspy.Predict(ClaimDecomposer)
    
    def forward(self, patient_presentation: str):
        """Analyze patient and decompose reasoning into claims"""
        # Step 1: Generate medical reasoning
        reasoning_result = self.reasoner(patient_presentation=patient_presentation)
        
        # Step 2: Decompose each reasoning step into claims
        all_claims = []
        for step in reasoning_result.reasoning_steps:
            decomposed = self.decomposer(reasoning_text=step)
            all_claims.extend(decomposed.claims)
        
        return {
            'reasoning': reasoning_result,
            'claims': all_claims
        }
    
    def answer_question(self, question: str, options: dict) -> str:
        """Adapt to standard interface using claim-based reasoning"""
        result = self.forward(question)
        
        # Score options based on claim alignment
        option_scores = {}
        for key, option in options.items():
            score = 0
            option_lower = option.lower()
            
            # Score based on claims that mention concepts in the option
            for claim in result['claims']:
                if isinstance(claim, dict) and 'statement' in claim:
                    claim_words = set(claim['statement'].lower().split())
                    option_words = set(option_lower.split())
                    overlap = len(claim_words.intersection(option_words))
                    
                    # Weight by confidence if available
                    confidence_weight = 1.0
                    if 'confidence' in claim:
                        if claim['confidence'].upper() == 'HIGH':
                            confidence_weight = 1.5
                        elif claim['confidence'].upper() == 'LOW':
                            confidence_weight = 0.5
                    
                    score += overlap * confidence_weight
            
            option_scores[key] = score
        
        # Return option with highest score
        best_option = max(option_scores, key=option_scores.get)
        return best_option

# Claim Verifying Agent
class ClaimVerifyingAgent(dspy.Module):
    """Agent that verifies medical claims against literature"""
    
    def __init__(self):
        super().__init__()
        self.reasoner = dspy.Predict(MedicalReasoner)
        self.decomposer = dspy.Predict(ClaimDecomposer)
        self.verifier = dspy.Predict(ClaimVerifier)
    
    def forward(self, patient_presentation: str):
        """Analyze patient, decompose reasoning, and verify claims"""
        # Step 1: Generate medical reasoning
        reasoning_result = self.reasoner(patient_presentation=patient_presentation)
        
        # Step 2: Decompose reasoning into claims
        all_claims = []
        for step in reasoning_result.reasoning_steps:
            decomposed = self.decomposer(reasoning_text=step)
            all_claims.extend(decomposed.claims)
        
        # Step 3: Verify each claim
        verified_claims = []
        for claim in all_claims:
            verification = self.verifier(
                claim=claim,
                medical_context=patient_presentation
            )
            verified_claim = {
                **claim,
                'verification_status': verification.verification_status,
                'evidence_quality': verification.evidence_quality,
                'source_citation': verification.source_citation,
                'verification_notes': verification.verification_notes
            }
            verified_claims.append(verified_claim)
        
        return {
            'reasoning': reasoning_result,
            'claims': all_claims,
            'verified_claims': verified_claims
        }
    
    def answer_question(self, question: str, options: dict) -> str:
        """Answer using verified claims with evidence quality weighting"""
        result = self.forward(question)
        
        option_scores = {}
        for key, option in options.items():
            score = 0
            option_lower = option.lower()
            
            for verified_claim in result['verified_claims']:
                if isinstance(verified_claim, dict) and 'statement' in verified_claim:
                    # Calculate word overlap
                    claim_words = set(verified_claim['statement'].lower().split())
                    option_words = set(option_lower.split())
                    overlap = len(claim_words.intersection(option_words))
                    
                    if overlap > 0:
                        # Weight by verification status
                        status_weight = {
                            'VERIFIED': 2.0,
                            'PARTIALLY_VERIFIED': 1.0,
                            'UNVERIFIED': 0.3,
                            'CONTRADICTED': -1.0
                        }.get(verified_claim.get('verification_status', 'UNVERIFIED'), 0.3)
                        
                        # Weight by evidence quality
                        evidence_weight = {
                            'A': 2.0,  # Guidelines/Systematic Review
                            'B': 1.5,  # RCT
                            'C': 1.0,  # Observational
                            'D': 0.5,  # Expert Opinion
                            'F': 0.1   # Cannot Verify
                        }.get(verified_claim.get('evidence_quality', 'F'), 0.1)
                        
                        # Weight by original confidence
                        confidence_weight = {
                            'HIGH': 1.5,
                            'MODERATE': 1.0,
                            'LOW': 0.5
                        }.get(verified_claim.get('confidence', 'MODERATE'), 1.0)
                        
                        score += overlap * status_weight * evidence_weight * confidence_weight
            
            option_scores[key] = score
        
        # Return option with highest score
        best_option = max(option_scores, key=option_scores.get)
        return best_option

# Complete Verifiable Medical Reasoning Agent
class VerifiableMedicalReasoner(dspy.Module):
    """Main module that orchestrates the complete verifiable reasoning pipeline."""
    
    def __init__(self):
        super().__init__()
        self.reasoner = dspy.Predict(MedicalReasoner)
        self.decomposer = dspy.Predict(ClaimDecomposer)
        self.verifier = dspy.Predict(ClaimVerifier)
        self.synthesizer = dspy.Predict(ReasoningSynthesizer)
    
    def forward(self, patient_presentation: str) -> Dict:
        # Step 1: Generate initial medical reasoning
        reasoning_output = self.reasoner(patient_presentation=patient_presentation)
        
        # Step 2: Decompose each reasoning step into claims
        all_claims = []
        for step in reasoning_output.reasoning_steps:
            decomposed = self.decomposer(reasoning_text=step)
            all_claims.extend(decomposed.claims)
        
        # Step 3: Verify each claim
        verified_claims = []
        for claim in all_claims:
            verification = self.verifier(
                claim=claim,
                medical_context=patient_presentation
            )
            verified_claim = {
                **claim,
                'verification_status': verification.verification_status,
                'evidence_quality': verification.evidence_quality,
                'source_citation': verification.source_citation,
                'verification_notes': verification.verification_notes
            }
            verified_claims.append(verified_claim)
        
        # Step 4: Synthesize verified reasoning
        synthesis = self.synthesizer(
            original_reasoning=reasoning_output.reasoning_steps,
            verified_claims=verified_claims
        )
        
        return {
            'original_reasoning': reasoning_output,
            'decomposed_claims': all_claims,
            'verified_claims': verified_claims,
            'synthesis': synthesis
        }
    
    def answer_question(self, question: str, options: dict) -> str:
        """Answer using complete verifiable reasoning pipeline"""
        result = self.forward(question)
        
        option_scores = {}
        for key, option in options.items():
            score = 0
            option_lower = option.lower()
            
            # Score based on synthesized assessment
            synthesis_words = set(result['synthesis'].verified_assessment.lower().split())
            option_words = set(option_lower.split())
            synthesis_overlap = len(synthesis_words.intersection(option_words))
            
            # Score based on verified claims with sophisticated weighting
            claim_score = 0
            for verified_claim in result['verified_claims']:
                if isinstance(verified_claim, dict) and 'statement' in verified_claim:
                    claim_words = set(verified_claim['statement'].lower().split())
                    overlap = len(claim_words.intersection(option_words))
                    
                    if overlap > 0:
                        # Multi-factor weighting
                        status_weight = {
                            'VERIFIED': 3.0,
                            'PARTIALLY_VERIFIED': 1.5,
                            'UNVERIFIED': 0.2,
                            'CONTRADICTED': -2.0
                        }.get(verified_claim.get('verification_status', 'UNVERIFIED'), 0.2)
                        
                        evidence_weight = {
                            'A': 2.5, 'B': 2.0, 'C': 1.5, 'D': 1.0, 'F': 0.1
                        }.get(verified_claim.get('evidence_quality', 'F'), 0.1)
                        
                        confidence_weight = {
                            'HIGH': 2.0, 'MODERATE': 1.0, 'LOW': 0.5
                        }.get(verified_claim.get('confidence', 'MODERATE'), 1.0)
                        
                        claim_score += overlap * status_weight * evidence_weight * confidence_weight
            
            # Combine synthesis and claim scores
            total_score = synthesis_overlap * 2.0 + claim_score
            
            # Apply confidence penalty if synthesis has low confidence
            if hasattr(result['synthesis'], 'confidence_summary'):
                avg_confidence = sum(result['synthesis'].confidence_summary.values()) / len(result['synthesis'].confidence_summary) if result['synthesis'].confidence_summary else 0.5
                total_score *= (0.5 + avg_confidence)  # Scale by confidence
            
            option_scores[key] = total_score
        
        best_option = max(option_scores, key=option_scores.get)
        return best_option

# Guideline-based Agent Manager
class GuidelineBasedAgentManager:
    """Manager for different teacher-student agent configurations"""
    
    def __init__(self):
        self.agents = {
            'predict_predict': MedAgent_Guideline_Simple_Predict_Predict(),
            'cot_predict': MedAgent_Guideline_Simple_CoT_Predict(),
            'predict_cot': MedAgent_Guideline_Simple_Predict_CoT(),
            'cot_cot': MedAgent_Guideline_Simple_CoT_CoT(),
            'advanced_planning': AdvancedPlanningAgent(),
            'mg_ranking_enhanced': MedAgent_MG_Ranking_Enhanced(),
            'mg_ranking_fixed': MedAgent_MG_Ranking_Fixed(),
            'mg_ranking_simple': MedAgent_MG_Ranking_Simple(),
            'mg_ranking_conservative': MedAgent_MG_Ranking_Conservative(),
            'guideline_based': MedAgent_GuidelineBased(),
            'simple_medical_reasoning': SimpleMedicalReasoningAgent(),
            'claim_decomposing': ClaimDecomposingAgent(),
            'claim_verifying': ClaimVerifyingAgent(),
            'verifiable_medical_reasoning': VerifiableMedicalReasoner(),
            'verifiable_medical_mcq_solver': VerifiableMedicalMCQSolver(),
            'differential_diagnosis': DifferentialDiagnosisModule(),
            'pharmacology': PharmacologyModule()
        }
    
    def get_agent(self, agent_type: str):
        """Get a specific agent by type"""
        if agent_type not in self.agents:
            raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(self.agents.keys())}")
        return self.agents[agent_type]
    
    def answer_question(self, question: str, options: dict, agent_type: str = 'cot_cot') -> str:
        """Answer a question using specified agent type"""
        agent = self.get_agent(agent_type)
        result = agent(question=question, options=options)
        return result.answer

# Evaluation Framework
class MedicalAgentEvaluator:
    """Evaluation framework for medical agents"""
    
    def __init__(self, guideline_manager: GuidelineBasedAgentManager):
        self.guideline_manager = guideline_manager
        self.simple_agent = SimpleMedicalAgent(use_chain_of_thought=True)
    
    def load_test_data(self, filepath: str, specialty: str = None):
        """Load test data from JSON file"""
        try:
            with open(filepath, 'r') as f:
                test_data = json.load(f)
            
            if specialty:
                test_data = [item for item in test_data if item.get('Specialty') == specialty]
            
            # Convert to DSPy examples
            examples = []
            for item in test_data:
                try:
                    example = dspy.Example(
                        question=item['Question'], 
                        options=map_letters_to_options(item['Options']),
                        answer=get_option_letter(item['Options'], item['Answer'])
                    ).with_inputs("question", "options")
                    examples.append(example)
                except Exception as e:
                    print(f"Skipping malformed item: {e}")
            
            return examples
        except FileNotFoundError:
            print(f"Test file {filepath} not found. Using sample data.")
            return self._get_sample_data()
    
    def _get_sample_data(self):
        """Get sample medical questions for testing"""
        sample_questions = [
            {
                "question": "A 65-year-old patient presents with chest pain and shortness of breath. ECG shows ST elevation in leads II, III, aVF. What is the most likely diagnosis?",
                "options": {
                    "A": "Anterior myocardial infarction",
                    "B": "Inferior myocardial infarction", 
                    "C": "Pulmonary embolism",
                    "D": "Aortic dissection"
                },
                "answer": "B"
            },
            {
                "question": "A 45-year-old diabetic patient presents with fever, dysuria, and flank pain. What is the most appropriate initial treatment?",
                "options": {
                    "A": "Oral ciprofloxacin",
                    "B": "IV ceftriaxone",
                    "C": "Oral trimethoprim-sulfamethoxazole",
                    "D": "IV vancomycin"
                },
                "answer": "B"
            }
        ]
        
        return [dspy.Example(**item).with_inputs("question", "options") for item in sample_questions]
    
    def evaluate_agent(self, agent_type: str, test_examples: List[dspy.Example], parallel: bool = False):
        """Evaluate a specific agent type"""
        correct = 0
        total = len(test_examples)
        results = []
        
        print(f"\n🔍 Evaluating {agent_type} agent on {total} questions...")
        
        if parallel and total > 1:
            # Prepare data for parallel processing
            questions_and_options = [(ex.question, ex.options) for ex in test_examples]
            
            start_time = time.time()
            parallel_results = run_parallel_requests(
                lambda q, o: self.guideline_manager.answer_question(q, o, agent_type),
                questions_and_options,
                max_workers=4
            )
            duration = time.time() - start_time
            
            for i, result in enumerate(parallel_results):
                if result["success"]:
                    predicted = result["answer"]
                    actual = test_examples[i].answer
                    is_correct = predicted == actual
                    if is_correct:
                        correct += 1
                    results.append({
                        "question": test_examples[i].question,
                        "predicted": predicted,
                        "actual": actual,
                        "correct": is_correct
                    })
                else:
                    results.append({
                        "question": test_examples[i].question,
                        "error": result["error"],
                        "correct": False
                    })
            
            print(f"⚡ Parallel processing completed in {duration:.2f}s")
        else:
            # Sequential processing
            start_time = time.time()
            for example in test_examples:
                try:
                    predicted = self.guideline_manager.answer_question(
                        example.question, example.options, agent_type
                    )
                    actual = example.answer
                    is_correct = predicted == actual
                    if is_correct:
                        correct += 1
                    results.append({
                        "question": example.question,
                        "predicted": predicted,
                        "actual": actual,
                        "correct": is_correct
                    })
                except Exception as e:
                    results.append({
                        "question": example.question,
                        "error": str(e),
                        "correct": False
                    })
            duration = time.time() - start_time
            print(f"🐌 Sequential processing completed in {duration:.2f}s")
        
        accuracy = correct / total if total > 0 else 0
        print(f"✅ Accuracy: {correct}/{total} ({accuracy:.1%})")
        
        return {
            "agent_type": agent_type,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "duration": duration,
            "results": results
        }
    
    def compare_agents(self, test_examples: List[dspy.Example], agent_types: List[str] = None):
        """Compare multiple agent types"""
        if agent_types is None:
            agent_types = ['predict_predict', 'cot_predict', 'predict_cot', 'cot_cot', 'mg_ranking_enhanced', 'mg_ranking_fixed', 'mg_ranking_simple', 'mg_ranking_conservative', 'guideline_based']
        
        print(f"\n🏆 Comparing {len(agent_types)} agents on {len(test_examples)} questions")
        print("=" * 60)
        
        results = {}
        for agent_type in agent_types:
            try:
                results[agent_type] = self.evaluate_agent(agent_type, test_examples)
            except Exception as e:
                print(f"❌ Error evaluating {agent_type}: {e}")
                results[agent_type] = {"error": str(e)}
        
        # Summary
        print(f"\n📊 Summary:")
        print("-" * 60)
        for agent_type, result in results.items():
            if "error" not in result:
                print(f"{agent_type:15} | Accuracy: {result['accuracy']:.1%} | Time: {result['duration']:.1f}s")
            else:
                print(f"{agent_type:15} | Error: {result['error']}")
        
        return results

def main():
    """Main function to demonstrate basic functionality"""
    print("🏥 Unified Medical Agent Application")
    print("=" * 50)
    
    # Configure DSPy
    lm = configure_dspy()
    print(f"✅ Configured DSPy with model: {lm.model}")
    
    # Create agents
    simple_agent = SimpleMedicalAgent(use_chain_of_thought=True)
    guideline_manager = GuidelineBasedAgentManager()
    evaluator = MedicalAgentEvaluator(guideline_manager)
    
    # Example medical question
    sample_question = "A 65-year-old patient presents with chest pain and shortness of breath. ECG shows ST elevation in leads II, III, aVF. What is the most likely diagnosis?"
    sample_options = {
        "A": "Anterior myocardial infarction",
        "B": "Inferior myocardial infarction", 
        "C": "Pulmonary embolism",
        "D": "Aortic dissection"
    }
    
    print("\n📋 Sample Question:")
    print(sample_question)
    print("\nOptions:")
    for key, value in sample_options.items():
        print(f"  {key}: {value}")
    
    # Test simple agent
    print("\n🤖 Simple Agent Answer:")
    try:
        answer = simple_agent.answer_question(sample_question, sample_options)
        print(f"Selected option: {answer}")
        if answer in sample_options:
            print(f"Answer: {sample_options[answer]}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test guideline-based agent
    print("\n🎓 Teacher-Student Agent Answer (CoT-CoT):")
    try:
        answer = guideline_manager.answer_question(sample_question, sample_options, 'cot_cot')
        print(f"Selected option: {answer}")
        if answer in sample_options:
            print(f"Answer: {sample_options[answer]}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test evaluation framework
    print("\n🧪 Running Agent Comparison:")
    test_examples = evaluator.load_test_data('s_medqa_test.json', 'Cardiology')
    comparison_results = evaluator.compare_agents(test_examples, ['cot_predict', 'cot_cot', 'mg_ranking_enhanced', 'mg_ranking_fixed', 'mg_ranking_simple', 'mg_ranking_conservative', 'guideline_based'])

if __name__ == "__main__":
    main() 