# AI Knowledge Assistant

An AI-powered document question-answering system built using **Next.js, FastAPI, Hybrid RAG, local embeddings, and Amazon Bedrock**.

The application allows users to ask questions about PDF documents and receive AI-generated answers grounded in retrieved document passages, with source citations.

## Architecture

```text
                       User
                        |
                        v
                 Next.js Frontend
                        |
                        v
                 Next.js API Proxy
                        |
                        v
                  FastAPI Backend
                        |
                        v
               Hybrid Retrieval (RAG)
                  /           \
                 v             v
         Semantic Search   Keyword Search
             (MiniLM)        (Custom BM25)
                 \             /
                  v           v
               Reciprocal Rank Fusion
                        |
                        v
               Top Relevant Passages
                        |
                        v
                 Amazon Bedrock
                   (Nova Micro)
                        |
                        v
             Grounded Answer + Citations
```

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js, React, TypeScript |
| Backend | Python, FastAPI |
| Document extraction | PyMuPDF |
| Chunking | Paragraph-aware text chunking |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Semantic retrieval | NumPy cosine similarity |
| Keyword retrieval | Custom BM25 implementation |
| Hybrid ranking | Reciprocal Rank Fusion (RRF) |
| LLM | Amazon Bedrock Nova Micro |
| Cloud | Amazon Web Services (AWS) |

## Key Features

- **PDF document ingestion:** Extract text and page information from PDF documents.
- **Paragraph-aware chunking:** Split extracted content into meaningful passages.
- **Local embeddings:** Generate document embeddings using Sentence Transformers.
- **Semantic search:** Retrieve relevant passages using vector similarity.
- **Keyword search:** Implement BM25-based keyword retrieval.
- **Hybrid retrieval:** Combine semantic and keyword search using Reciprocal Rank Fusion.
- **Grounded AI generation:** Generate answers using retrieved evidence and Amazon Bedrock.
- **Source citations:** Associate generated answers with their supporting document passages.
- **Retrieval evaluation:** Benchmark retrieval performance across different strategies.
- **Unsupported-question testing:** Evaluate system behaviour when the available documents do not contain the requested information.

## Retrieval Evaluation

The project includes an initial benchmark of 20 questions covering:

- Direct questions
- Multi-part questions
- Cross-document questions
- Unsupported questions

### Initial retrieval comparison

| Retrieval strategy | Top-3 source/page coverage | Top-5 source/page coverage |
|---|---:|---:|
| Semantic | 71.9% | 78.1% |
| Hybrid | 78.1% | 81.2% |
| Document-aware | 78.1% | 81.2% |

**Observations:**

- Hybrid retrieval improved source/page coverage compared with semantic retrieval in this benchmark.
- Document-aware retrieval achieved the same measured coverage as hybrid retrieval.
- These results represent retrieval source/page coverage, not answer accuracy, factual correctness, or citation correctness.

## Project Structure

```text
ai-knowledge-assistant/
├── backend/
│   ├── main.py
│   ├── chunking.py
│   ├── retrieval.py
│   ├── generation.py
│   └── index_documents.py
├── documents/
├── evaluation/
│   ├── questions.json
│   └── results.json
│   └── retrieval_comparison.json
│   ├── run_evaluation.py
│   ├── diagnose_retrieval.py
│   ├── compare_retrieval.py
│   └── inspect_document_chunks.py
│   └── inspect_failures.py
├── tests/
│   ├── test_chunking.py
│   └── test_retrieval.py
├── src/
│   └── app/
│       ├── api/
│       │   ├── ask/route.ts
│       │   └── health/route.ts
│       ├── page.tsx
│       ├── layout.tsx
│       └── globals.css
├── requirements.txt
├── requirements-dev.txt
├── package.json
└── README.md
```

## Getting Started

### Prerequisites

- Node.js and npm
- Python 3.10+
- AWS account with Amazon Bedrock access
- AWS credentials configured locally
- Access to the Amazon Nova Micro model in the AWS region being used

### 1. Clone the repository

```bash
git clone https://github.com/sandeepdevalla/ai-knowledge-assistant.git

cd ai-knowledge-assistant
```

### 2. Download the sample PDF documents

The sample knowledge base uses two publicly available NIST publications.

Download the documents from their official NIST pages:

- [NIST SP 800-145 – The NIST Definition of Cloud Computing](https://csrc.nist.gov/pubs/sp/800/145/final)
- [NIST SP 800-207 – Zero Trust Architecture](https://csrc.nist.gov/pubs/sp/800/207/final)

Create the `documents/` directory if it does not exist and save the PDFs using these exact filenames:

```text
documents/
├── NIST-SP-800-145.pdf
└── NIST-SP-800-207.pdf
```

**Note:** PDF files are not included in this GitHub repository. Download them separately before indexing the documents.

### 3. Backend setup

Create and activate a Python virtual environment:

```bash
python -m venv .venv

source .venv/bin/activate
```

Install backend dependencies:

```bash
pip install -r requirements.txt
```

For development and testing:

```bash
pip install -r requirements-dev.txt
```

### 4. Configure AWS Bedrock

The application uses Amazon Bedrock Converse API with the Nova Micro model.

Current configuration:

| Configuration | Value |
|---|---|
| AWS Region | us-east-1 |
| Model ID | amazon.nova-micro-v1:0 |
| SDK | boto3 |

Configure AWS credentials using an appropriate AWS authentication method.

Ensure that:

- Your AWS identity has permission to invoke the selected Bedrock model.
- The model is available in your configured region.
- Your AWS credentials are valid.

Do not hardcode AWS access keys or commit credentials to the repository.

### 5. Index documents

After downloading the PDFs and activating the virtual environment, run:

```bash
python backend/index_documents.py
```

This loads the documents, extracts their text, creates chunks and prepares the retrieval index.

### 6. Start the FastAPI backend

```bash
uvicorn backend.main:app --reload --port 8000
```

The backend will be available at:

```text
http://localhost:8000
```

### 7. Configure and start the Next.js frontend

Create a `.env.local` file in the project root:

```env
BACKEND_URL=http://127.0.0.1:8000
```

Install frontend dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Open the application:

[http://localhost:3000](http://localhost:3000)

**Note:** Keep the FastAPI backend and Next.js frontend running in separate terminal sessions.

## Running Tests

Run the backend test suite:

```bash
python -m pytest -q
```

The tests cover document chunking and retrieval functionality.

## Security Considerations

- Never commit AWS credentials or access keys.
- Keep `.env.local` and other environment-specific configuration files out of version control.
- Keep private or sensitive documents out of public repositories.
- Use IAM roles and least-privilege permissions when deploying to AWS.
- Review document access and data-handling requirements before using enterprise documents.

## Current Scope and Future Improvements

This project is actively evolving as a practical exploration of RAG architecture and AI engineering.

Potential improvements include:

- More robust answerability detection and abstention.
- Stronger citation-to-claim validation.
- Retrieval and generation latency instrumentation.
- Improved retrieval evaluation and answer-quality measurement.
- Production deployment using AWS cloud infrastructure.
- Authentication and document-level access control.

## Author

Built as a hands-on engineering project exploring Retrieval-Augmented Generation (RAG), hybrid information retrieval, local embeddings, and cloud-based generative AI using Amazon Bedrock.

**GitHub:** [sandeepdevalla](https://github.com/sandeepdevalla)
