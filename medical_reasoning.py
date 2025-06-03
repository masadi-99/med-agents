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
                    div_options = divergence.get('options', [])
                    
                    # Try to get claim details for each option in the divergence
                    for opt in div_options:
                        if opt in option_trees:
                            # Look for relevant claims at this level
                            level_claims = [c for c in option_trees[opt] if c.get('hierarchy_level', 1) == level]
                            if level_claims:
                                # Use the first claim at this level as representative
                                claim_details[opt] = level_claims[0]
                    
                    # Only judge if we have claim details for at least one option
                    if claim_details:
                        try:
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
                                
                        except Exception as e:
                            print(f"⚠️ Warning: Could not judge divergence {divergence.get('divergence_id', 'unknown')}: {e}")
                            # Add a minimal score for the first option as fallback
                            if div_options:
                                fallback_option = div_options[0]
                                if fallback_option in option_scores:
                                    option_scores[fallback_option] += 0.1
                                    level_scores[fallback_option][level] += 0.1
                    else:
                        print(f"⚠️ Warning: No claim details found for divergence {divergence.get('divergence_id', 'unknown')}")
        
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
            
            try:
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
                    # Ensure div_data has all required fields
                    div_data_safe = {
                        'divergence_id': div_data.get('divergence_id', f'L{level}_DIV_{len(level_divergences[level])+1}'),
                        'claim_pairs': div_data.get('claim_pairs', []),
                        'options': div_data.get('options', [comp.option1, comp.option2]),
                        'divergence_type': div_data.get('divergence_type', 'unknown'),
                        'description': div_data.get('description', 'Divergence in reasoning at this level'),
                        'level': level,
                        'critical_for_answer': div_data.get('critical_for_answer', True)
                    }
                    level_divergences[level].append(div_data_safe)  # Store dict for easier access
                    
            except Exception as e:
                print(f"⚠️ Warning: Could not analyze divergences at level {level}: {e}")
                # Create a basic divergence entry for this level
                for i, comp in enumerate(comps):
                    basic_divergence = {
                        'divergence_id': f'L{level}_DIV_BASIC_{i+1}',
                        'claim_pairs': [(comp.claim1_id, comp.claim2_id)],
                        'options': [comp.option1, comp.option2],
                        'divergence_type': 'basic_conflict',
                        'description': f'Basic divergence between options {comp.option1} and {comp.option2}',
                        'level': level,
                        'critical_for_answer': True
                    }
                    level_divergences[level].append(basic_divergence)
        
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