import dspy
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MedicalQuestionAnswering(dspy.Signature):
    """Answer multiple choice medical questions by selecting the correct option."""
    
    question = dspy.InputField(desc="A multiple choice medical question with options")
    answer = dspy.OutputField(desc="The correct option (A, B, C, or D) with brief reasoning")

class MedicalAgent:
    def __init__(self, model_name="gpt-3.5-turbo"):
        """Initialize the medical agent with a language model."""
        # Configure DSPy with OpenAI
        lm = dspy.OpenAI(
            model=model_name,
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=150
        )
        dspy.settings.configure(lm=lm)
        
        # Create the chain of thought predictor
        self.predictor = dspy.ChainOfThought(MedicalQuestionAnswering)
    
    def answer_question(self, question):
        """
        Answer a multiple choice medical question.
        
        Args:
            question (str): The medical question with multiple choice options
            
        Returns:
            str: The predicted answer with reasoning
        """
        try:
            result = self.predictor(question=question)
            return result.answer
        except Exception as e:
            return f"Error processing question: {str(e)}"

def main():
    """Example usage of the medical agent."""
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY environment variable.")
        print("You can create a .env file with: OPENAI_API_KEY=your_api_key_here")
        return
    
    # Initialize the agent
    agent = MedicalAgent()
    
    # Example medical question
    sample_question = """
    A 45-year-old patient presents with chest pain, shortness of breath, and diaphoresis. 
    The ECG shows ST-segment elevation in leads II, III, and aVF. What is the most likely diagnosis?
    
    A) Anterior myocardial infarction
    B) Inferior myocardial infarction  
    C) Pulmonary embolism
    D) Aortic dissection
    """
    
    print("Medical Question Answering Agent")
    print("=" * 50)
    print("\nSample Question:")
    print(sample_question)
    
    print("\nAgent's Answer:")
    answer = agent.answer_question(sample_question)
    print(answer)
    
    # Interactive mode
    print("\n" + "=" * 50)
    print("Interactive Mode (type 'quit' to exit)")
    print("=" * 50)
    
    while True:
        user_question = input("\nEnter your medical question: ")
        if user_question.lower() == 'quit':
            break
        
        if user_question.strip():
            answer = agent.answer_question(user_question)
            print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    main() 