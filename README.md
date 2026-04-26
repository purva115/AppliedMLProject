# PharmaLLM – Medicine Prescriber Chatbot

> ⚕️ **Medical Disclaimer**: This is a research/educational project. Do _not_ use it as a substitute for professional medical consultation.

## Overview

PharmaLLM is a fine-tuned LLM-based medicine prescriber chatbot built on **TinyLlama-1.1B** + **LoRA**, with multi-modal input/output (text + speech) and multi-language support.

Primary reference paper: [PharmaLLM: A Medicine Prescriber Chatbot Exploiting Open-Source Large Language Models](https://link.springer.com/article/10.1007/s44230-024-00085-z)

| Metric | Target |
|---|---|
| Accuracy | 87% |
| F1 Score | 92.16% |
| Precision | 90% |
| Recall | 94% |

---

## Project Structure

```
pharma-llm/
├── backend/          # Flask API server
│   ├── app.py
│   ├── config.py
│   ├── requirements.txt
│   ├── models/       # TinyLlama + LoRA loader & inference
│   ├── services/     # NLP, TTS, STT, Translation
│   └── routes/       # Chat API blueprints
├── data/
│   ├── raw/          # Kaggle CSV (place here)
│   ├── processed/    # Output JSONL files
│   └── prepare_dataset.py
├── training/
│   ├── config_lora.py
│   ├── fine_tune.py
│   └── evaluate.py
├── frontend/         # React + Vite app
├── report/           # Report generator + final PDF
└── notebooks/        # Data exploration
```

---

## Quick Start

### 1. Backend

```bash
cd pharma-llm/backend
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
pip install langdetect openai-whisper  # Essential for translation and STT
```

### 2. Run Server
Start the Flask server. It is currently configured to load fine-tuned weights from `training/fine_tuned_model/checkpoint-200`.

```bash
python app.py
```

### 3. Frontend

```bash
cd pharma-llm/frontend
npm install
npm run dev
```

### 4. Generate Project Report PDF

```bash
cd pharma-llm
python report/generate_report_pdf.py
```

Output:
- `report/PharmaLLM_Written_Report.pdf`

---

## Inference Tuning

The model has been calibrated for cleaner, more concise medical advice:
- **Temperature**: 0.3 (Reduced hallucination)
- **Repetition Penalty**: 1.2
- **N-Gram Blocking**: Prevents 3-word repeating loops.
- **Max Response**: Limited to ~50 words for readability.

---

## Tech Stack

**Backend**: Python, Flask, PyTorch, Transformers, PEFT (LoRA), langdetect, Whisper, gTTS  
**Frontend**: React 18, Vite, Axios  
**Model**: TinyLlama-1.1B-Chat + LoRA adapters
