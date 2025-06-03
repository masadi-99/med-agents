"""
Enhanced Medical Comparative Reasoning Framework v3.0
=====================================================

Streamlined framework focused on enhanced comparative reasoning for medical MCQs.
Combines systematic claim decomposition with level-based divergence analysis.
"""

import dspy
from typing import List, Dict, Optional, Set, Any, Tuple
from dataclasses import dataclass
import json
from enum import Enum
from collections import defaultdict

# ============= Core Data Structures =============

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

# ============= Normalization Functions =============

def normalize_verification_status(status: str) -> str:
    """Normalize verification status to ensure consistent format."""
    if not status:
        return VerificationStatus.UNVERIFIED.value
    
    normalized = status.upper().strip().rstrip('.').rstrip('!')
    
    status_map = {
        'VERIFIED': VerificationStatus.VERIFIED.value,
        'VERIFIED_WITH_CONTEXT': VerificationStatus.VERIFIED_WITH_CONTEXT.value,
        'PARTIALLY_VERIFIED': VerificationStatus.PARTIALLY_VERIFIED.value,
        'UNVERIFIED': VerificationStatus.UNVERIFIED.value,
        'CONTRADICTED': VerificationStatus.CONTRADICTED.value,
    }
    
    return status_map.get(normalized, VerificationStatus.UNVERIFIED.value)

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

# ============= DSPy Signatures for Enhanced Comparative Reasoning =============

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

class EnhancedClaimDecomposer(dspy.Signature):
    """Decomposes reasoning into structured, verifiable claims with explicit clinical prioritization.
    
    CLAIM HIERARCHY REQUIREMENTS:
    1. Level 1: Basic Facts (no dependencies)
    2. Level 2: Physiological Context (depends on Level 1)
    3. Level 3: Pathophysiological Mechanisms (depends on Level 2)
    4. Level 4: Clinical Manifestations (depends on Level 3)
    5. Level 5: Answer Justification (depends on Level 4)
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
    """Verifies claims explicitly separating truth from clinical prioritization."""
    
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

# ============= Optimized DSPy Signatures for Batch Processing =============

class BatchLevelAnalyzer(dspy.Signature):
    """Analyze all claims at a specific hierarchy level across all options simultaneously."""
    
    level: int = dspy.InputField(desc="Hierarchy level being analyzed (1-5)")
    level_claims_by_option: Dict[str, List[Dict]] = dspy.InputField(
        desc="Claims at this level for each option: {option_letter: [claims]}"
    )
    question_context: str = dspy.InputField()
    
    claim_relationships: List[Dict] = dspy.OutputField(
        desc="""Analysis of claim relationships with format:
        {
            relationship_id: str,
            claims_involved: Dict[str, str] (option -> claim_id),
            relationship_type: 'IDENTICAL'/'SIMILAR'/'RELATED'/'CONFLICTING'/'UNRELATED',
            is_divergence_point: bool,
            divergence_type: str (if divergence: 'mechanism'/'assumption'/'interpretation'/'factual'),
            description: str,
            clinical_significance: str,
            options_in_conflict: List[str] (if divergence)
        }"""
    )
    level_summary: str = dspy.OutputField(
        desc="Summary of key patterns and conflicts at this level"
    )

class BatchDivergenceJudge(dspy.Signature):
    """Judge multiple divergences at once with level-aware weighting."""
    
    level: int = dspy.InputField(desc="Hierarchy level of divergences")
    divergences: List[Dict] = dspy.InputField(desc="List of divergences to judge")
    option_claims: Dict[str, Dict] = dspy.InputField(
        desc="Claim details for each option involved in divergences"
    )
    clinical_context: str = dspy.InputField()
    patient_presentation: Dict[str, str] = dspy.InputField()
    
    batch_resolutions: List[Dict] = dspy.OutputField(
        desc="""Resolutions for all divergences with format:
        {
            divergence_id: str,
            winning_option: str,
            confidence: float (0-1),
            level_weight: float (0-1),
            divergence_impact: str,
            reasoning: str,
            evidence_quality: str (A/B/C/D/F),
            comparative_analysis: str (why this option beats others)
        }"""
    )
    level_importance: float = dspy.OutputField(
        desc="Overall importance of this level for final decision (0-1)"
    )

# ============= Optimized Comparative Reasoning Solver =============

class OptimizedComparativeReasoningSolver(dspy.Module):
    """Optimized comparative solver with batched processing to reduce LLM calls."""
    
    def __init__(self):
        super().__init__()
        self.option_analyzer = dspy.Predict(OptionSpecificAnalyzer)
        self.decomposer = dspy.Predict(EnhancedClaimDecomposer)
        self.verifier = dspy.Predict(ContextAwareVerifier)
        self.prioritizer = dspy.Predict(ClinicalPrioritizer)
        self.batch_level_analyzer = dspy.Predict(BatchLevelAnalyzer)
        self.batch_divergence_judge = dspy.Predict(BatchDivergenceJudge)
        self.final_selector = dspy.Predict(FinalAnswerSelector)
    
    def forward(self, question: str, options: Dict[str, str]) -> Dict:
        print("🚀 Starting optimized comparative reasoning analysis...")
        
        # Step 1: Generate reasoning trees for each option (parallelizable but kept sequential for now)
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
        
        print("⚡ Performing batched level-by-level analysis...")
        
        # Step 2: Optimized level-by-level batch analysis
        level_relationships = {}
        level_divergences = {}
        
        # Group claims by level across all options
        claims_by_level = defaultdict(dict)
        for option, claims in option_trees.items():
            for claim in claims:
                level = claim.get('hierarchy_level', 1)
                if option not in claims_by_level[level]:
                    claims_by_level[level][option] = []
                claims_by_level[level][option].append(claim)
        
        # Analyze each level in batch
        for level in sorted(claims_by_level.keys()):
            if level > 0:
                level_claims = claims_by_level[level]
                if len(level_claims) > 1:  # Only analyze if multiple options have claims at this level
                    print(f"🔍 Batch analyzing level {level}...")
                    
                    try:
                        level_analysis = self.batch_level_analyzer(
                            level=level,
                            level_claims_by_option=level_claims,
                            question_context=question
                        )
                        
                        level_relationships[level] = level_analysis.claim_relationships
                        
                        # Extract divergences from relationships
                        divergences = [
                            rel for rel in level_analysis.claim_relationships
                            if rel.get('is_divergence_point', False)
                        ]
                        
                        if divergences:
                            level_divergences[level] = divergences
                            
                    except Exception as e:
                        print(f"⚠️ Warning: Could not analyze level {level}: {e}")
                        # Create basic divergences as fallback
                        level_divergences[level] = self._create_fallback_divergences(level, level_claims)
        
        print(f"⚖️ Batch judging divergences across {len(level_divergences)} levels...")
        
        # Step 3: Optimized batch divergence judgment
        divergence_resolutions = {}
        option_scores = {opt: 0 for opt in options.keys()}
        level_scores = {opt: defaultdict(float) for opt in options.keys()}
        
        for level, divergences in level_divergences.items():
            if divergences:
                # Prepare option claims for this level
                option_claims = {}
                for opt in options.keys():
                    if opt in claims_by_level[level]:
                        level_claims = claims_by_level[level][opt]
                        if level_claims:
                            option_claims[opt] = level_claims[0]  # Use first claim as representative
                
                if option_claims:
                    try:
                        # Batch judge all divergences at this level
                        batch_judgment = self.batch_divergence_judge(
                            level=level,
                            divergences=divergences,
                            option_claims=option_claims,
                            clinical_context=list(option_analyses.values())[0]['clinical_context'],
                            patient_presentation=list(option_analyses.values())[0]['patient_presentation']
                        )
                        
                        # Process batch resolutions
                        for resolution in batch_judgment.batch_resolutions:
                            div_id = resolution['divergence_id']
                            divergence_resolutions[div_id] = resolution
                            
                            # Score with level weighting
                            winning_option = resolution['winning_option'].strip('"\'')
                            if winning_option in option_scores:
                                confidence = resolution['confidence']
                                level_weight = resolution.get('level_weight', batch_judgment.level_importance)
                                weighted_score = confidence * level_weight
                                option_scores[winning_option] += weighted_score
                                level_scores[winning_option][level] += weighted_score
                            else:
                                print(f"⚠️ Warning: Unknown option '{winning_option}' from batch judgment")
                                
                    except Exception as e:
                        print(f"⚠️ Warning: Could not judge divergences at level {level}: {e}")
                        # Add fallback scoring
                        for opt in option_claims.keys():
                            option_scores[opt] += 0.1
                            level_scores[opt][level] += 0.1
        
        # Step 4: Final answer selection
        best_option = max(option_scores, key=option_scores.get) if any(option_scores.values()) else list(options.keys())[0]
        best_score = option_scores[best_option] if best_option in option_scores else 0.0
        
        # Create final selection input
        final_selection = self.final_selector(
            question=question,
            options=options,
            divergence_resolutions=[
                {'divergence': div, 'judgment': res, 'level': level}
                for level, divs in level_divergences.items()
                for div in divs
                for res in [divergence_resolutions.get(div.get('relationship_id', ''), {})]
                if res
            ],
            option_claim_trees=option_trees
        )
        
        print(f"✅ Optimized analysis complete! Reduced to ~{2 + len(level_divergences)} LLM calls")
        
        return {
            'answer': best_option,
            'confidence': best_score,
            'option_trees': option_trees,
            'option_analyses': option_analyses,
            'level_relationships': level_relationships,
            'level_divergences': level_divergences,
            'divergence_resolutions': divergence_resolutions,
            'option_scores': option_scores,
            'level_scores': level_scores,
            'final_selection': final_selection,
            'reasoning_method': 'optimized_batch_comparative_analysis',
            'optimization_stats': {
                'levels_analyzed': len(level_divergences),
                'total_divergences': sum(len(divs) for divs in level_divergences.values()),
                'estimated_call_reduction': f"~{len(options) * (len(options) - 1) // 2 * 5}→{2 + len(level_divergences)} calls"
            }
        }
    
    def _create_fallback_divergences(self, level: int, level_claims: Dict[str, List[Dict]]) -> List[Dict]:
        """Create basic divergences when batch analysis fails."""
        divergences = []
        options = list(level_claims.keys())
        
        for i in range(len(options)):
            for j in range(i + 1, len(options)):
                opt1, opt2 = options[i], options[j]
                div_id = f"L{level}_FALLBACK_{opt1}v{opt2}"
                
                divergences.append({
                    'relationship_id': div_id,
                    'claims_involved': {
                        opt1: level_claims[opt1][0]['claim_id'] if level_claims[opt1] else f"{opt1}_missing",
                        opt2: level_claims[opt2][0]['claim_id'] if level_claims[opt2] else f"{opt2}_missing"
                    },
                    'relationship_type': 'CONFLICTING',
                    'is_divergence_point': True,
                    'divergence_type': 'basic_conflict',
                    'description': f'Basic conflict between options {opt1} and {opt2} at level {level}',
                    'clinical_significance': 'Requires judgment between competing explanations',
                    'options_in_conflict': [opt1, opt2]
                })
        
        return divergences
    
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

# ============= Complete Visualization Functions =============

def visualize_enhanced_option_trees(option_trees: Dict[str, List[Dict]], option_analyses: Dict[str, Dict]):
    """Enhanced visualization of option trees with beautiful structure and detailed information."""
    print("\n🌳 ENHANCED OPTION REASONING TREES:")
    print("=" * 80)
    
    for option_letter, claims in option_trees.items():
        analysis = option_analyses[option_letter]
        
        # Header with option details
        print(f"\n📋 OPTION {option_letter} DETAILED REASONING TREE")
        print("─" * 60)
        print(f"🎯 Clinical Context: {analysis['clinical_context'][:120]}...")
        print(f"🧠 Pathophysiology: {analysis['pathophysiology_explanation'][:120]}...")
        print(f"⚙️  Supporting Mechanisms: {', '.join(analysis['supporting_mechanisms'][:2])}...")
        print(f"📊 Total Claims: {len(claims)}")
        
        # Group by hierarchy level
        levels = defaultdict(list)
        for claim in claims:
            level = claim.get('hierarchy_level', 1)
            levels[level].append(claim)
        
        level_names = {
            1: "🔸 Level 1: Basic Facts & Patient Data",
            2: "🔹 Level 2: Physiological Context & Normal Function", 
            3: "🔶 Level 3: Pathophysiological Mechanisms & Disease Process",
            4: "🔷 Level 4: Clinical Manifestations & Symptoms",
            5: "⭐ Level 5: Answer Justification & Final Reasoning"
        }
        
        for level in sorted(levels.keys()):
            if level > 0:
                print(f"\n{level_names.get(level, f'📍 Level {level}')}:")
                print("┌" + "─" * 70 + "┐")
                
                for i, claim in enumerate(levels[level]):
                    # Status and relevance indicators
                    truth_status = claim.get('truth_status', 'UNKNOWN')
                    clinical_relevance = claim.get('clinical_relevance', 'UNKNOWN')
                    
                    status_icon = {
                        'VERIFIED': '✅',
                        'VERIFIED_WITH_CONTEXT': '🟢',
                        'PARTIALLY_VERIFIED': '🟡',
                        'UNVERIFIED': '⚠️',
                        'CONTRADICTED': '❌'
                    }.get(truth_status, '❓')
                    
                    relevance_icon = {
                        'RELEVANT': '🎯',
                        'PARTIALLY_RELEVANT': '📍',
                        'IRRELEVANT': '🔘'
                    }.get(clinical_relevance, '❓')
                    
                    confidence_icon = {
                        'HIGH': '🔥',
                        'MODERATE': '🔶',
                        'LOW': '🔸'
                    }.get(claim.get('confidence', 'UNKNOWN'), '❓')
                    
                    # Claim display
                    claim_id = claim['claim_id']
                    statement = claim['statement']
                    
                    print(f"│ {status_icon}{relevance_icon}{confidence_icon} {claim_id}: {statement[:55]}...")
                    
                    # Additional details
                    if claim.get('depends_on'):
                        deps = ', '.join(claim['depends_on'])
                        print(f"│   🔗 Dependencies: {deps}")
                    
                    if claim.get('verification_explanation'):
                        explanation = claim['verification_explanation'][:60]
                        print(f"│   💭 Verification: {explanation}...")
                    
                    if claim.get('reasoning_bridge'):
                        bridge = claim['reasoning_bridge'][:60]
                        print(f"│   🌉 Bridge: {bridge}...")
                    
                    # Separator between claims
                    if i < len(levels[level]) - 1:
                        print("│" + "─" * 70)
                
                print("└" + "─" * 70 + "┘")

def visualize_claim_comparisons(claim_comparisons: List):
    """Beautiful visualization of claim comparisons across options."""
    print("\n🔍 DETAILED CLAIM COMPARISONS:")
    print("=" * 80)
    
    if not claim_comparisons:
        print("No claim comparisons found.")
        return
    
    # Group by similarity type
    similarity_groups = defaultdict(list)
    for comp in claim_comparisons:
        similarity_groups[comp.similarity.value].append(comp)
    
    similarity_icons = {
        'IDENTICAL': '🔗',
        'SIMILAR': '🔀',
        'RELATED': '📋',
        'CONFLICTING': '⚔️',
        'UNRELATED': '🚫'
    }
    
    for similarity, comparisons in similarity_groups.items():
        if not comparisons:
            continue
            
        icon = similarity_icons.get(similarity, '❓')
        print(f"\n{icon} {similarity} COMPARISONS ({len(comparisons)} total):")
        print("─" * 60)
        
        for i, comp in enumerate(comparisons[:10]):  # Show first 10 to avoid overwhelming
            divergence_status = "🔥 DIVERGENCE POINT" if comp.divergence_point else "📍 Similarity Point"
            
            print(f"\n📊 Comparison #{i+1}: Options {comp.option1} vs {comp.option2}")
            print(f"   Status: {divergence_status}")
            print(f"   Level: {comp.level} | Similarity: {comp.similarity.value}")
            print(f"   Claims: {comp.claim1_id} ↔ {comp.claim2_id}")
            print(f"   Notes: {comp.comparison_notes[:80]}...")
            
            if i < len(comparisons) - 1:
                print("   " + "─" * 50)
        
        if len(comparisons) > 10:
            print(f"\n   ... and {len(comparisons) - 10} more {similarity.lower()} comparisons")

def visualize_level_divergences(level_divergences: Dict[int, List[Dict]]):
    """Enhanced visualization of divergences organized by hierarchy level."""
    print("\n📊 LEVEL-BASED DIVERGENCE ANALYSIS:")
    print("=" * 80)
    
    if not level_divergences:
        print("No level divergences found.")
        return
    
    level_names = {
        1: "🔸 Level 1: Basic Facts Divergences",
        2: "🔹 Level 2: Physiological Context Divergences", 
        3: "🔶 Level 3: Mechanism Divergences",
        4: "🔷 Level 4: Clinical Manifestation Divergences",
        5: "⭐ Level 5: Answer Justification Divergences"
    }
    
    total_divergences = sum(len(divs) for divs in level_divergences.values())
    critical_divergences = sum(sum(1 for d in divs if d.get('critical_for_answer', False)) for divs in level_divergences.values())
    
    print(f"📈 Overview: {total_divergences} total divergences, {critical_divergences} critical for answer")
    
    for level in sorted(level_divergences.keys()):
        divergences = level_divergences[level]
        if not divergences:
            continue
            
        print(f"\n{level_names.get(level, f'📍 Level {level} Divergences')}:")
        print("┌" + "═" * 70 + "┐")
        
        for i, div in enumerate(divergences):
            critical_icon = "🔥" if div.get('critical_for_answer', False) else "⚠️"
            divergence_type = div.get('divergence_type', 'unknown').upper()
            
            type_icons = {
                'MECHANISM': '⚙️',
                'ASSUMPTION': '💭',
                'INTERPRETATION': '🧠',
                'FACTUAL': '📋',
                'CONFLICTING': '⚔️',
                'BASIC_CONFLICT': '🔄',
                'UNKNOWN': '❓'
            }
            
            type_icon = type_icons.get(divergence_type, '❓')
            
            div_id = div.get('divergence_id', div.get('relationship_id', 'Unknown'))
            options_involved = div.get('options', div.get('options_in_conflict', []))
            description = div.get('description', 'No description')
            
            print(f"│ {critical_icon} {type_icon} {div_id} ({divergence_type})")
            print(f"│ ├─ Options Involved: {' vs '.join(options_involved)}")
            print(f"│ ├─ Claim Pairs: {len(div.get('claim_pairs', []))} pairs")
            print(f"│ └─ Description: {description[:60]}...")
            
            if i < len(divergences) - 1:
                print("│" + "─" * 70)
        
        print("└" + "═" * 70 + "┘")

def visualize_divergence_details(level_divergences: Dict[int, List[Dict]], option_trees: Dict[str, List[Dict]]):
    """Detailed visualization of specific divergence points with claim details."""
    print("\n🔍 DETAILED DIVERGENCE ANALYSIS:")
    print("=" * 80)
    
    critical_divergences = []
    for level, divs in level_divergences.items():
        for div in divs:
            if div.get('critical_for_answer', False):
                critical_divergences.append((level, div))
    
    if not critical_divergences:
        print("No critical divergences found.")
        return
    
    print(f"📊 Analyzing {len(critical_divergences)} critical divergences in detail:\n")
    
    for level, div in critical_divergences[:5]:  # Show first 5 for detail
        div_id = div.get('divergence_id', div.get('relationship_id', 'Unknown'))
        divergence_type = div.get('divergence_type', 'unknown').upper()
        options_involved = div.get('options', div.get('options_in_conflict', []))
        description = div.get('description', 'No description')
        
        print(f"🔥 CRITICAL DIVERGENCE: {div_id}")
        print("═" * 60)
        print(f"📍 Level: {level} | Type: {divergence_type}")
        print(f"⚔️ Options: {' vs '.join(options_involved)}")
        print(f"📝 Description: {description}")
        
        # Show the actual conflicting claims
        print(f"\n🔍 Conflicting Claims Analysis:")
        
        for opt in options_involved:
            if opt in option_trees:
                level_claims = [c for c in option_trees[opt] if c.get('hierarchy_level', 1) == level]
                if level_claims:
                    claim = level_claims[0]  # Take first claim at this level
                    status_icon = "✅" if claim.get('truth_status') == 'VERIFIED' else "⚠️"
                    
                    print(f"\n   {status_icon} Option {opt} Position:")
                    print(f"      Claim: {claim.get('statement', 'No statement')[:100]}...")
                    print(f"      Evidence: Grade {claim.get('evidence_quality', 'N/A')}")
                    print(f"      Confidence: {claim.get('confidence', 'N/A')}")
        
        print("\n" + "─" * 60 + "\n")

def visualize_structured_resolutions(divergence_resolutions: Dict[str, Any], level_scores: Dict[str, Dict[int, float]]):
    """Enhanced visualization of structured divergence resolutions with detailed analysis."""
    print("\n🏛️ STRUCTURED DIVERGENCE RESOLUTIONS:")
    print("=" * 80)
    
    if not divergence_resolutions:
        print("No divergence resolutions found.")
        return
    
    # Group by confidence level
    high_confidence = []
    medium_confidence = []
    low_confidence = []
    
    for div_id, resolution in divergence_resolutions.items():
        # Handle both dict and object formats
        if isinstance(resolution, dict):
            confidence = resolution.get('confidence', 0)
        else:
            confidence = getattr(resolution, 'confidence', 0)
            
        if confidence >= 0.8:
            high_confidence.append((div_id, resolution))
        elif confidence >= 0.6:
            medium_confidence.append((div_id, resolution))
        else:
            low_confidence.append((div_id, resolution))
    
    confidence_groups = [
        ("🔥 HIGH CONFIDENCE RESOLUTIONS", high_confidence),
        ("🔶 MEDIUM CONFIDENCE RESOLUTIONS", medium_confidence),
        ("🔸 LOW CONFIDENCE RESOLUTIONS", low_confidence)
    ]
    
    for group_name, resolutions in confidence_groups:
        if not resolutions:
            continue
            
        print(f"\n{group_name} ({len(resolutions)} total):")
        print("─" * 60)
        
        for div_id, resolution in resolutions:
            winner_icon = "🏆"
            
            # Handle both dict and object formats
            if isinstance(resolution, dict):
                winning_option = resolution.get('winning_option', 'N/A')
                confidence = resolution.get('confidence', 0)
                level_weight = resolution.get('level_weight', 0)
                evidence_grade = resolution.get('evidence_quality', 'N/A')
                divergence_impact = resolution.get('divergence_impact', 'No impact specified')
                reasoning = resolution.get('reasoning', 'No reasoning provided')
            else:
                winning_option = getattr(resolution, 'winning_option', 'N/A')
                confidence = getattr(resolution, 'confidence', 0)
                level_weight = getattr(resolution, 'level_weight', 0)
                evidence_grade = getattr(resolution, 'evidence_quality', 'N/A')
                divergence_impact = getattr(resolution, 'divergence_impact', 'No impact specified')
                reasoning = getattr(resolution, 'reasoning', 'No reasoning provided')
            
            grade_icon = {
                'A': '🥇', 'B': '🥈', 'C': '🥉', 
                'D': '📝', 'F': '❌'
            }.get(evidence_grade, '❓')
            
            print(f"\n{winner_icon} Resolution: {div_id}")
            print(f"   🏆 Winner: Option {winning_option}")
            print(f"   📊 Confidence: {confidence:.2f}")
            print(f"   ⚖️ Level Weight: {level_weight:.2f}")
            print(f"   {grade_icon} Evidence Quality: Grade {evidence_grade}")
            print(f"   📝 Impact: {divergence_impact[:80]}...")
            print(f"   💡 Reasoning: {reasoning[:100]}...")
    
    # Level score breakdown with visual bars
    print(f"\n📈 LEVEL SCORE BREAKDOWN:")
    print("=" * 50)
    
    if level_scores:
        max_score = max(sum(scores.values()) for scores in level_scores.values()) if level_scores else 1
        
        for option, scores in sorted(level_scores.items()):
            total = sum(scores.values())
            if total > 0:
                # Create visual bar
                bar_length = int((total / max_score) * 30) if max_score > 0 else 0
                bar = "█" * bar_length + "░" * (30 - bar_length)
                
                print(f"\nOption {option}: {total:.2f}")
                print(f"  [{bar}]")
                
                for level, score in sorted(scores.items()):
                    if score > 0:
                        level_bar_length = int((score / total) * 20) if total > 0 else 0
                        level_bar = "▓" * level_bar_length + "░" * (20 - level_bar_length)
                        print(f"    Level {level}: {score:.2f} [{level_bar}]")

def visualize_enhanced_comparative_summary(result: Dict):
    """Enhanced visualization of comparative reasoning summary with detailed metrics."""
    print("\n🎯 ENHANCED COMPARATIVE REASONING SUMMARY:")
    print("=" * 80)
    
    # Main results
    answer = result['answer']
    confidence = result['confidence']
    method = result['reasoning_method']
    
    print(f"🏆 SELECTED ANSWER: {answer}")
    print(f"📊 CONFIDENCE SCORE: {confidence:.2f}")
    print(f"🔬 REASONING METHOD: {method}")
    
    # Optimization statistics
    visualize_optimization_stats(result)
    
    # Enhanced statistics
    total_claims = sum(len(claims) for claims in result['option_trees'].values())
    total_relationships = sum(len(rels) for rels in result.get('level_relationships', {}).values())
    total_divergences = sum(len(divs) for divs in result.get('level_divergences', {}).values())
    
    print(f"\n📈 ANALYSIS METRICS:")
    print("┌─────────────────────────────────┐")
    print(f"│ Total Claims Generated: {total_claims:8} │")
    print(f"│ Level Relationships:    {total_relationships:8} │")
    print(f"│ Level Divergences:      {total_divergences:8} │")
    print(f"│ Resolutions Found:      {len(result.get('divergence_resolutions', {})):8} │")
    print("└─────────────────────────────────┘")
    
    # Option performance
    option_scores = result.get('option_scores', {})
    if option_scores:
        print(f"\n🏅 OPTION PERFORMANCE RANKING:")
        print("─" * 40)
        
        sorted_options = sorted(option_scores.items(), key=lambda x: x[1], reverse=True)
        max_score = sorted_options[0][1] if sorted_options else 1
        
        for i, (option, score) in enumerate(sorted_options):
            rank_icon = ["🥇", "🥈", "🥉", "🏅", "🎖️"][min(i, 4)]
            bar_length = int((score / max_score) * 25) if max_score > 0 else 0
            bar = "█" * bar_length + "░" * (25 - bar_length)
            
            print(f"{rank_icon} Option {option}: {score:.2f} [{bar}]")

def visualize_complete_analysis(result: Dict):
    """Master function to display all visualizations in a comprehensive manner."""
    print("\n" + "🔬" * 40)
    print("COMPLETE COMPARATIVE REASONING ANALYSIS")
    print("🔬" * 40)
    
    # Check if this is optimized or enhanced
    method = result.get('reasoning_method', '')
    if 'optimized' in method:
        # Main summary
        visualize_optimized_comparative_summary(result)
        
        # Detailed option trees
        visualize_enhanced_option_trees(result['option_trees'], result['option_analyses'])
        
        # Batch relationships
        visualize_batch_relationships(result.get('level_relationships', {}))
        
        # Level divergences (optimized format)
        level_divergences = result.get('level_divergences', {})
        if level_divergences:
            visualize_level_divergences(level_divergences)
        
        # Structured resolutions (optimized format) 
        divergence_resolutions = result.get('divergence_resolutions', {})
        if divergence_resolutions:
            level_scores = result.get('level_scores', {})
            visualize_structured_resolutions(divergence_resolutions, level_scores)
    else:
        # Main summary
        visualize_enhanced_comparative_summary(result)
        
        # Detailed option trees
        visualize_enhanced_option_trees(result['option_trees'], result['option_analyses'])
        
        # Claim comparisons
        visualize_claim_comparisons(result.get('claim_comparisons', []))
        
        # Level divergences
        visualize_level_divergences(result.get('level_divergences', {}))
        
        # Detailed divergence analysis
        visualize_divergence_details(result.get('level_divergences', {}), result['option_trees'])
        
        # Structured resolutions
        visualize_structured_resolutions(result.get('divergence_resolutions', {}), result.get('level_scores', {}))
    
    print("\n" + "✅" * 40)
    print("ANALYSIS COMPLETE")
    print("✅" * 40)

# ============= Optimization-Specific Visualization Functions =============

def visualize_optimization_stats(result: Dict):
    """Visualize the optimization improvements."""
    print("\n⚡ OPTIMIZATION STATISTICS:")
    print("=" * 50)
    
    stats = result.get('optimization_stats', {})
    method = result.get('reasoning_method', 'unknown')
    
    print(f"🔬 Method: {method}")
    print(f"📊 Levels Analyzed: {stats.get('levels_analyzed', 0)}")
    print(f"⚔️ Total Divergences: {stats.get('total_divergences', 0)}")
    print(f"⚡ Call Reduction: {stats.get('estimated_call_reduction', 'N/A')}")
    
    # Calculate actual efficiency
    num_options = len(result.get('option_trees', {}))
    if num_options > 1:
        old_calls = num_options * (num_options - 1) // 2 * 5  # Pairwise comparisons across 5 levels
        new_calls = 2 + stats.get('levels_analyzed', 0)  # Batch analysis + batch judging
        efficiency = ((old_calls - new_calls) / old_calls * 100) if old_calls > 0 else 0
        
        print(f"📈 Efficiency Gain: {efficiency:.1f}% reduction in LLM calls")
        print(f"   Previous: ~{old_calls} calls")
        print(f"   Optimized: ~{new_calls} calls")

def visualize_batch_relationships(level_relationships: Dict[int, List[Dict]]):
    """Visualize batch-analyzed relationships by level."""
    print("\n🔗 BATCH RELATIONSHIP ANALYSIS:")
    print("=" * 60)
    
    if not level_relationships:
        print("No level relationships found.")
        return
    
    for level in sorted(level_relationships.keys()):
        relationships = level_relationships[level]
        if not relationships:
            continue
            
        level_names = {
            1: "🔸 Level 1: Basic Facts",
            2: "🔹 Level 2: Physiological Context", 
            3: "🔶 Level 3: Pathophysiological Mechanisms",
            4: "🔷 Level 4: Clinical Manifestations",
            5: "⭐ Level 5: Answer Justification"
        }
        
        print(f"\n{level_names.get(level, f'📍 Level {level}')}:")
        print("─" * 50)
        
        # Group by relationship type
        by_type = defaultdict(list)
        for rel in relationships:
            by_type[rel.get('relationship_type', 'UNKNOWN')].append(rel)
        
        for rel_type, rels in by_type.items():
            type_icons = {
                'IDENTICAL': '🔗',
                'SIMILAR': '🔀',
                'RELATED': '📋',
                'CONFLICTING': '⚔️',
                'UNRELATED': '🚫'
            }
            
            icon = type_icons.get(rel_type, '❓')
            divergence_count = sum(1 for r in rels if r.get('is_divergence_point', False))
            
            print(f"  {icon} {rel_type}: {len(rels)} relationships ({divergence_count} divergences)")
            
            # Show key divergences
            for rel in rels[:3]:  # Show first 3
                if rel.get('is_divergence_point', False):
                    options = rel.get('options_in_conflict', [])
                    print(f"    🔥 {rel.get('relationship_id', 'Unknown')}: {' vs '.join(options)}")
                    print(f"       {rel.get('description', 'No description')[:60]}...")

def visualize_optimized_comparative_summary(result: Dict):
    """Enhanced summary for optimized framework."""
    print("\n🚀 OPTIMIZED COMPARATIVE REASONING SUMMARY:")
    print("=" * 70)
    
    # Main results
    answer = result['answer']
    confidence = result['confidence']
    method = result['reasoning_method']
    
    print(f"🏆 SELECTED ANSWER: {answer}")
    print(f"📊 CONFIDENCE SCORE: {confidence:.2f}")
    print(f"⚡ REASONING METHOD: {method}")
    
    # Optimization statistics
    visualize_optimization_stats(result)
    
    # Enhanced statistics
    total_claims = sum(len(claims) for claims in result['option_trees'].values())
    total_relationships = sum(len(rels) for rels in result.get('level_relationships', {}).values())
    total_divergences = sum(len(divs) for divs in result.get('level_divergences', {}).values())
    
    print(f"\n📈 ANALYSIS METRICS:")
    print("┌─────────────────────────────────┐")
    print(f"│ Total Claims Generated: {total_claims:8} │")
    print(f"│ Level Relationships:    {total_relationships:8} │")
    print(f"│ Level Divergences:      {total_divergences:8} │")
    print(f"│ Resolutions Found:      {len(result.get('divergence_resolutions', {})):8} │")
    print("└─────────────────────────────────┘")
    
    # Option performance
    option_scores = result.get('option_scores', {})
    if option_scores:
        print(f"\n🏅 OPTION PERFORMANCE RANKING:")
        print("─" * 40)
        
        sorted_options = sorted(option_scores.items(), key=lambda x: x[1], reverse=True)
        max_score = sorted_options[0][1] if sorted_options else 1
        
        for i, (option, score) in enumerate(sorted_options):
            rank_icon = ["🥇", "🥈", "🥉", "🏅", "🎖️"][min(i, 4)]
            bar_length = int((score / max_score) * 25) if max_score > 0 else 0
            bar = "█" * bar_length + "░" * (25 - bar_length)
            
            print(f"{rank_icon} Option {option}: {score:.2f} [{bar}]")

def visualize_complete_analysis(result: Dict):
    """Master function to display all visualizations in a comprehensive manner."""
    print("\n" + "🔬" * 40)
    print("COMPLETE COMPARATIVE REASONING ANALYSIS")
    print("🔬" * 40)
    
    # Check if this is optimized or enhanced
    method = result.get('reasoning_method', '')
    if 'optimized' in method:
        # Main summary
        visualize_optimized_comparative_summary(result)
        
        # Detailed option trees
        visualize_enhanced_option_trees(result['option_trees'], result['option_analyses'])
        
        # Batch relationships
        visualize_batch_relationships(result.get('level_relationships', {}))
        
        # Level divergences (optimized format)
        level_divergences = result.get('level_divergences', {})
        if level_divergences:
            visualize_level_divergences(level_divergences)
        
        # Structured resolutions (optimized format) 
        divergence_resolutions = result.get('divergence_resolutions', {})
        if divergence_resolutions:
            level_scores = result.get('level_scores', {})
            visualize_structured_resolutions(divergence_resolutions, level_scores)
    else:
        # Main summary
        visualize_enhanced_comparative_summary(result)
        
        # Detailed option trees
        visualize_enhanced_option_trees(result['option_trees'], result['option_analyses'])
        
        # Claim comparisons
        visualize_claim_comparisons(result.get('claim_comparisons', []))
        
        # Level divergences
        visualize_level_divergences(result.get('level_divergences', {}))
        
        # Detailed divergence analysis
        visualize_divergence_details(result.get('level_divergences', {}), result['option_trees'])
        
        # Structured resolutions
        visualize_structured_resolutions(result.get('divergence_resolutions', {}), result.get('level_scores', {}))
    
    print("\n" + "✅" * 40)
    print("ANALYSIS COMPLETE")
    print("✅" * 40) 