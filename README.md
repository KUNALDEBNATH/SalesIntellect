# Platinum LLM — Sales Intelligence Chatbot

A modular **AI-powered sales intelligence and document-question-answering system** built for automotive sales workflows.

Platinum LLM combines structured sales-data retrieval, document understanding, intent detection, confidence handling, provenance tracking, and an optional locally trained language model to turn business data into natural-language answers.

---

## 🚀 Key Capabilities

- **Sales data intelligence** — Works with enquiry, appointment, and feedback records.
- **Retrieval-Augmented Generation (RAG)** — Retrieves relevant records before generating an answer.
- **Natural-language querying** — Users can ask business questions without knowing database syntax.
- **Intent detection** — Identifies what the user is trying to find before retrieval.
- **Ambiguity detection** — Handles unclear customer/entity references.
- **Document intelligence** — Parses, verifies, understands, and answers questions from uploaded documents.
- **Confidence-aware responses** — Uses confidence and verification layers to reduce unreliable answers.
- **Provenance tracking** — Keeps track of where retrieved information originated.
- **Optional local LLM** — Supports a locally trained/fine-tuned language model for response generation.
- **Web interface** — Provides a browser-based chatbot through a Django API.
- **Modular architecture** — Individual components can be improved or replaced without redesigning the complete system.

---

## 🧠 System Architecture

```text
                         ┌─────────────────────┐
                         │      User Query     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Query / Intent    │
                         │      Analysis       │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
             ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
             │ Sales Data  │ │  Document   │ │  Session /  │
             │ Retrieval   │ │ Intelligence│ │   Context   │
             └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Verification &      │
                         │ Confidence Layer    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Answer Generation   │
                         │ / Optional LLM      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Final Response +    │
                         │ Provenance/Intent   │
                         └─────────────────────┘
```

---

## 📁 Project Structure

```text
platinum_llm/
│
├── aggregation_engine.py
├── ambiguity_detector.py
├── api.py
├── attachment_handler.py
├── bot_identity.py
├── confidence.py
├── data_quality.py
├── doc_training_data.py
├── document_answer_engine.py
├── document_parser.py
├── document_understanding.py
├── document_verifier.py
├── domain_guard.py
├── evaluate_document_intelligence.py
├── evaluate_intelligence.py
├── patch_test_py.py
├── provenance.py
├── query_engine.py
├── rag_utils.py
├── regression_tests.py
├── scratch_llm.py
├── scratch_llm_patch.py
├── session_state.py
├── test.py
├── train.py
├── train_with_documents.py
├── verification.py
├── vision_parser.py
│
├── scratch_model/
│   ├── model.pt
│   └── vocab.json
│
├── sales_llm_model/
│   └── Fine-tuned / local model artifacts
│
├── index.html
├── requirements.txt
├── .gitignore
│
└── Data files
    ├── sales_enquiry_dataset.csv
    ├── sales_appointment_dataset.csv
    └── sales_feedback_dataset.csv
```

> Generated model files, virtual environments, caches, datasets, and other local artifacts should remain excluded through `.gitignore` where appropriate.

---

## 🔄 Processing Pipeline

### 1. Input

The user submits a natural-language question through the CLI or web interface.

Example:

```text
Show me the enquiries for customers interested in SUVs.
```

### 2. Query Understanding

The system analyses the question to determine:

- User intent
- Relevant entities
- Customer names / enquiry IDs
- Required data source
- Ambiguity
- Query confidence

### 3. Retrieval

Relevant records are retrieved from the available sales information.

The system can work with:

- Enquiry records
- Appointment records
- Feedback records
- Document content

### 4. Verification

Retrieved information is checked through confidence, verification, and data-quality components before being used for the final answer.

### 5. Answer Generation

The system constructs an answer from the verified information.

When the optional LLM is available, it can convert the structured result into a more natural conversational response.

### 6. Response

The final response can include:

- Answer
- Detected intent
- Confidence information
- Relevant context
- Provenance/source information

---

## 📊 Sales Data

The project can use three primary sales datasets.

### Enquiry Dataset

Contains customer enquiry information such as:

- Enquiry ID
- Customer name
- Contact information
- Vehicle/model
- Enquiry source
- Enquiry date
- Appointment date
- Location
- Customer type
- Payment type
- Test-ride status
- Enquiry status

### Appointment Dataset

Contains:

- Enquiry ID
- Customer name
- Appointment date
- Appointment time
- Vehicle
- Appointment status

### Feedback Dataset

Contains:

- Enquiry ID
- Customer name
- Feedback
- Rating
- Feedback date

---

## 📄 Document Intelligence

The document pipeline extends the chatbot beyond structured CSV records.

Relevant components include:

```text
Document
   │
   ▼
Document Parser
   │
   ▼
Document Understanding
   │
   ▼
Document Verification
   │
   ▼
Document Answer Engine
   │
   ▼
Verified Natural-Language Answer
```

The project also includes training and evaluation utilities for document intelligence.

Supported document processing can be extended depending on the parser implementation and installed dependencies.

---

## 🤖 Local LLM

The repository includes components for a local/scratch language-model workflow.

Important files include:

| File | Purpose |
|---|---|
| `scratch_llm.py` | Scratch language-model implementation |
| `scratch_llm_patch.py` | Model/pipeline patches |
| `train.py` | LLM training pipeline |
| `train_with_documents.py` | Training using document-oriented data |
| `doc_training_data.py` | Document training-data preparation |
| `sales_llm_model/` | Local/fine-tuned model artifacts |
| `scratch_model/` | Scratch model artifacts |

The LLM layer is designed to work with the retrieval and intelligence pipeline rather than replacing factual retrieval.

---

## 🌐 Web Application

The project provides a Django-based API and browser interface.

### Start the API

```bash
python api.py
```

The development server runs on:

```text
http://localhost:8000
```

Open the web interface in your browser:

```text
http://localhost:8000
```

---

## 🔌 API

The application exposes endpoints for chatbot interaction and system management.

Typical endpoints include:

```text
GET  /api/health/
POST /api/chat/
POST /api/reset/
GET  /api/suggestions/
GET  /
```

### Example Chat Request

```json
{
  "query": "Show details for ENQ001"
}
```

A response can contain the generated answer together with information such as intent and processing time.

---

## 💻 CLI Usage

Run:

```bash
python test.py
```

Then ask questions such as:

```text
Show ENQ001 details
Who gave bad feedback?
Show cancelled appointments
Which customers are interested in SUVs?
What vehicle did the customer enquire about?
Show customers who have not taken a test ride
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/KUNALDEBNATH/platinum_llm.git
cd platinum_llm
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If the Django dependencies are not already included:

```bash
pip install django djangorestframework django-cors-headers
```

### 4. Start the application

```bash
python api.py
```

---

## 🧪 Evaluation and Testing

The repository includes evaluation and regression utilities:

```text
evaluate_intelligence.py
evaluate_document_intelligence.py
regression_tests.py
verification.py
```

These components can be used to test retrieval quality, document intelligence, verification behavior, and regressions as the system evolves.

---

## 🔐 Security & Repository Hygiene

Do **not** commit:

- `.env` files
- API keys
- passwords
- credentials
- private certificates
- virtual environments
- Python cache files
- temporary files
- unnecessary generated model artifacts
- private/customer-sensitive datasets unless explicitly approved

The included `.gitignore` is intended to keep common local and sensitive artifacts out of the repository.

---

## 🛠️ Technology Stack

- **Python**
- **Django**
- **Django REST Framework**
- **PyTorch**
- **Transformers**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **RAG / Information Retrieval**
- **Natural Language Processing**
- **Local/Fine-tuned LLM**
- **HTML/CSS/JavaScript**

---

## 🎯 Project Goals

The main objective of Platinum LLM is to provide a reliable conversational interface for sales intelligence while keeping factual responses grounded in available business information.

The architecture focuses on:

1. **Retrieval before generation**
2. **Context-aware query understanding**
3. **Verification and confidence handling**
4. **Document-level intelligence**
5. **Source/provenance awareness**
6. **Modular LLM integration**
7. **Scalable API-based deployment**

---

## 🔮 Future Enhancements

Potential future improvements include:

- Graph-based retrieval and Graph RAG
- Multimodal document understanding
- Better table and spreadsheet reasoning
- Advanced semantic retrieval
- Larger domain-specific training datasets
- Vector database integration
- Production authentication and authorization
- Monitoring and observability
- Containerized deployment
- Cloud deployment
- Automated evaluation pipelines

---

## 👨‍💻 Author

**Kunal Debnath**

GitHub:  
https://github.com/KUNALDEBNATH

---

## 📄 License

This project is intended for academic, research, and development purposes unless a separate license is added to the repository.
