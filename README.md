# Unified Medical Agent Application with DSPy

A comprehensive DSPy-based agentic application for medical question answering with multiple agent architectures, parallel processing, and evaluation framework.

## 🚀 Features

### **Multiple Agent Architectures**
- **Simple Agent**: Basic question answering with Chain-of-Thought reasoning
- **Teacher-Student Framework**: Guideline-based agents with 4 configurations:
  - `predict_predict`: Teacher uses Predict, Student uses Predict
  - `cot_predict`: Teacher uses ChainOfThought, Student uses Predict  
  - `predict_cot`: Teacher uses Predict, Student uses ChainOfThought
  - `cot_cot`: Teacher uses ChainOfThought, Student uses ChainOfThought
- **Advanced Planning Agent**: Multi-step reasoning with planning, fetching, and ranking

### **Parallel Processing**
- Thread-based parallel request processing
- Support for both local vLLM and OpenAI APIs
- Configurable worker count for optimal performance
- Performance benchmarking and comparison

### **Evaluation Framework**
- Comprehensive agent comparison system
- Support for medical question datasets (JSON format)
- Specialty filtering (e.g., Cardiology, Internal Medicine)
- Accuracy metrics and timing analysis
- Sample data fallback when datasets unavailable

### **Flexible Model Support**
- **OpenAI Models**: GPT-4, GPT-3.5-turbo, GPT-4o-mini
- **Local vLLM**: DeepSeek R1 Distill Llama 8B
- Automatic fallback to local server if API key unavailable

## 📦 Setup

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure API Keys**:
   - Copy `env_example.txt` to `.env`
   - Add your OpenAI API key: `OPENAI_API_KEY=your_key_here`
   - Or use local vLLM server (see Local Setup section)

3. **Optional: Add Test Data**:
   - Place your medical QA dataset as `s_medqa_test.json`
   - Format: `[{"Question": "...", "Options": [...], "Answer": "...", "Specialty": "..."}]`

## 🔧 Local vLLM Setup

To use the DeepSeek R1 model locally:

```bash
# Start vLLM server with DeepSeek R1
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --tensor-parallel-size 8 \
  --enable-reasoning \
  --reasoning-parser deepseek_r1 \
  --enforce-eager \
  --max-model-len 8192
```

The application will automatically detect and use the local server at `http://localhost:8000/v1`.

## 🚀 Usage

### **Basic Usage**
```bash
python unified_med_agent.py
```

### **Programmatic Usage**
```python
from unified_med_agent import GuidelineBasedAgentManager, MedicalAgentEvaluator

# Initialize agents
manager = GuidelineBasedAgentManager()

# Answer a question
answer = manager.answer_question(
    question="Patient presents with chest pain...",
    options={"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"},
    agent_type="cot_cot"
)

# Evaluate agents
evaluator = MedicalAgentEvaluator(manager)
test_data = evaluator.load_test_data('your_dataset.json', 'Cardiology')
results = evaluator.compare_agents(test_data, ['cot_predict', 'cot_cot'])
```

### **Available Agent Types**
- `predict_predict`: Fastest, basic reasoning
- `cot_predict`: Teacher reasoning + fast student
- `predict_cot`: Fast teacher + student reasoning  
- `cot_cot`: Full reasoning (recommended)
- `advanced_planning`: Multi-step planning and ranking

## 🧪 Evaluation

The application includes a comprehensive evaluation framework:

### **Agent Comparison**
```python
evaluator = MedicalAgentEvaluator(guideline_manager)
test_examples = evaluator.load_test_data('s_medqa_test.json', 'Cardiology')
results = evaluator.compare_agents(test_examples, ['cot_predict', 'cot_cot'])
```

### **Parallel Evaluation**
```python
# Enable parallel processing for faster evaluation
results = evaluator.evaluate_agent('cot_cot', test_examples, parallel=True)
```

### **Performance Metrics**
- Accuracy percentage
- Processing time
- Individual question results
- Error tracking

## 🏗️ Architecture

### **Core Components**
1. **DSPy Signatures**: Define input/output structures for each agent type
2. **Agent Modules**: Implement different reasoning strategies
3. **Manager Classes**: Coordinate multiple agents and configurations
4. **Parallel Processing**: Handle concurrent requests efficiently
5. **Evaluation Framework**: Systematic testing and comparison

### **Agent Flow**
```
Question + Options → Agent Selection → Processing → Answer

Teacher-Student Flow:
Question → Teacher (Guidelines) → Student (Answer with Guidelines)

Advanced Planning Flow:
Question → Planner → Fetcher → Reasoner → Ranker → Answer
```

## 📊 Performance

With your DeepSeek R1 setup (`--tensor-parallel-size 8`):
- **Sequential Processing**: ~2-5 seconds per question
- **Parallel Processing**: ~0.5-1 second per question (4 workers)
- **Expected Speedup**: 3-4x with parallel processing
- **Memory Usage**: Scales with model size and parallel workers

## 🔬 Medical Specialties

The application supports specialty-specific evaluation:
- Cardiology
- Internal Medicine  
- Emergency Medicine
- Pulmonology
- Endocrinology
- And more...

## 📝 Example Output

```
🏥 Unified Medical Agent Application
==================================================
✅ Configured DSPy with model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B

🧪 Running Agent Comparison:
🏆 Comparing 2 agents on 25 questions
============================================================

🔍 Evaluating cot_predict agent on 25 questions...
🐌 Sequential processing completed in 45.2s
✅ Accuracy: 22/25 (88.0%)

🔍 Evaluating cot_cot agent on 25 questions...
🐌 Sequential processing completed in 52.1s  
✅ Accuracy: 24/25 (96.0%)

📊 Summary:
------------------------------------------------------------
cot_predict     | Accuracy: 88.0% | Time: 45.2s
cot_cot         | Accuracy: 96.0% | Time: 52.1s
```

## 🛠️ Development

### **Adding New Agents**
1. Create DSPy signature in the signatures section
2. Implement agent module inheriting from `dspy.Module`
3. Add to `GuidelineBasedAgentManager.agents` dictionary
4. Test with evaluation framework

### **Custom Evaluation**
1. Prepare dataset in required JSON format
2. Use `MedicalAgentEvaluator.load_test_data()`
3. Run `compare_agents()` or `evaluate_agent()`

## 📄 Requirements

- Python 3.8+
- DSPy framework 
- OpenAI API access (optional)
- vLLM for local models (optional)
- aiohttp for async processing

## 🔗 Related Files

- `unified_med_agent.py`: Main application
- `requirements.txt`: Python dependencies
- `env_example.txt`: Environment variable template
- `.env`: Your API keys (create from template) 