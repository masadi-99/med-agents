import dspy
import json
import os
from dotenv import load_dotenv
from typing import Dict, List
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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

# Evaluation Framework
class MedicalAgentEvaluator:
    """Evaluation framework for medical agents"""
    
    def __init__(self, guideline_manager: GuidelineBasedAgentManager):
        self.guideline_manager = guideline_manager
        self.simple_agent = SimpleMedicalAgent()
    
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
            agent_types = ['predict_predict', 'cot_predict', 'predict_cot', 'cot_cot']
        
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
    comparison_results = evaluator.compare_agents(test_examples, ['cot_predict', 'cot_cot'])

if __name__ == "__main__":
    main() 