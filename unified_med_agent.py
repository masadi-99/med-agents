import dspy
import json
import os
from dotenv import load_dotenv
from typing import Dict, List
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

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

# Guideline-based Agent Manager
class GuidelineBasedAgentManager:
    """Manager for different teacher-student agent configurations"""
    
    def __init__(self):
        self.agents = {
            'predict_predict': MedAgent_Guideline_Simple_Predict_Predict(),
            'cot_predict': MedAgent_Guideline_Simple_CoT_Predict(),
            'predict_cot': MedAgent_Guideline_Simple_Predict_CoT(),
            'cot_cot': MedAgent_Guideline_Simple_CoT_CoT(),
            'advanced_planning': AdvancedPlanningAgent()
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
    
    # Test advanced planning agent
    print("\n🧠 Advanced Planning Agent Answer:")
    try:
        answer = guideline_manager.answer_question(sample_question, sample_options, 'advanced_planning')
        print(f"Selected option: {answer}")
        if answer in sample_options:
            print(f"Answer: {sample_options[answer]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 