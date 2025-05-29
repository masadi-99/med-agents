# Medical Question Answering Agent with DSPy

A simple and minimalistic DSPy-based agent that answers multiple choice medical questions.

## Features

- Uses DSPy framework for structured prompting
- Chain-of-thought reasoning for medical questions
- Interactive command-line interface
- Support for multiple choice questions with options A, B, C, D

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your OpenAI API key:
   - Copy `env_example.txt` to `.env`
   - Replace `your_openai_api_key_here` with your actual OpenAI API key

## Usage

Run the agent:
```bash
python medical_agent.py
```

The agent will:
1. Show a sample medical question and answer
2. Enter interactive mode where you can ask your own questions
3. Type 'quit' to exit

## Example

```
Medical Question Answering Agent
==================================================

Sample Question:
A 45-year-old patient presents with chest pain, shortness of breath, and diaphoresis. 
The ECG shows ST-segment elevation in leads II, III, and aVF. What is the most likely diagnosis?

A) Anterior myocardial infarction
B) Inferior myocardial infarction  
C) Pulmonary embolism
D) Aortic dissection

Agent's Answer:
B) Inferior myocardial infarction

The ECG findings of ST-segment elevation in leads II, III, and aVF are characteristic of inferior wall myocardial infarction, as these leads represent the inferior aspect of the heart.
```

## Code Structure

- `MedicalQuestionAnswering`: DSPy signature defining input/output structure
- `MedicalAgent`: Main agent class with question answering capability
- Chain-of-thought reasoning for better medical decision making

## Requirements

- Python 3.8+
- DSPy framework
- OpenAI API access 