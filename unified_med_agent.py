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

def main():
    """Main function to demonstrate basic functionality"""
    print("🏥 Unified Medical Agent Application")
    print("=" * 50)
    
    # Configure DSPy
    lm = configure_dspy()
    print(f"✅ Configured DSPy with model: {lm.model}")
    
    # Create simple agent
    agent = SimpleMedicalAgent(use_chain_of_thought=True)
    
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
    
    print("\n🤖 Agent Answer:")
    try:
        answer = agent.answer_question(sample_question, sample_options)
        print(f"Selected option: {answer}")
        if answer in sample_options:
            print(f"Answer: {sample_options[answer]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 