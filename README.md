# Enhanced Medical Verifiable Reasoning Framework v2.0

A DSPy-based framework for medical multiple choice question answering with verifiable reasoning, explicit claim dependencies, and enhanced context awareness.

## Features

- **Enhanced Claim Structure**: Explicit context, assumptions, and dependencies
- **Context-Aware Verification**: Claims verified with clinical context matching
- **Dependency-Aware Selection**: Topological verification of claim dependencies
- **Specialized Modules**: Pathophysiology analyzer for complex questions
- **Comprehensive Analysis**: Deep clinical context recognition

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your API key:
```bash
cp config_example.py config.py
# Edit config.py and add your OpenAI API key
```

3. Run the example:
```bash
python example.py
```

## Usage

```python
import dspy
from medical_reasoning import EnhancedMedicalMCQSolver

# Configure DSPy
lm = dspy.LM('openai/gpt-4o-mini', api_key="your-key")
dspy.configure(lm=lm)

# Initialize solver
solver = EnhancedMedicalMCQSolver()

# Solve medical MCQ
result = solver(question="...", options={"A": "...", "B": "..."})
print(f"Answer: {result['answer']}")
```

## Framework Components

- **EnhancedMedicalAnalyzer**: Deep clinical context recognition
- **EnhancedClaimDecomposer**: Structured claim decomposition with dependencies
- **ContextAwareVerifier**: Context-aware claim verification
- **DependencyAwareSelector**: Dependency-aware answer selection
- **PathophysiologyAnalyzer**: Specialized pathophysiology analysis

## Claim Structure

Each claim includes:
- `claim_type`: FACT/INFERENCE/DEFINITION/ASSUMPTION/CONDITION
- `statement`: Explicit and specific claim
- `context`: Conditions under which claim is true
- `assumptions`: Explicit assumptions
- `depends_on`: IDs of prerequisite claims
- `verification_method`: Evidence source type
- `supports_option`: Which option it supports
- `contradicts_options`: Which options it contradicts

## Security

Your API key is stored in `config.py` which is excluded from version control via `.gitignore`. Never commit your actual API key to the repository. 