# DLA Piper Technical Assessment: Document FAQ Generation Pipeline

## Overview

Your task is to design and evaluate a system that automatically generates accurate FAQs from legal documents using RAG (Retrieval-Augmented Generation) techniques. This assessment focuses on your ability to build, evaluate, and plan deployment of an ML pipeline, with particular emphasis on evaluation methodologies and production considerations.

## Provided Resources

In the accompanying zip file, you will find:

- 20 randomly selected legal PDF documents from the [CUAD dataset](https://www.atticusprojectai.org/cuad). These PDFs serve as the ground truth documents.
- A JSON file `data/CUAD.json` containing the parsed and cleaned text from the associated PDF documents, along with evaluation data.

### CUAD JSON Structure

The `data/CUAD.json` file contains the following fields:

- `data`: A list of documents, each represented as a dictionary with the following keys:
  - `title`: The title of the document.
  - `paragraphs`: A list of paragraphs within the document, each containing:
    - `context`: The text content of the paragraph.
    - `qas`: A list of question-answer pairs related to the paragraph, each containing:
      - `id`: A unique identifier for the question.
      - `question`: The question related to the context.
      - `answers`: A list of answers, each containing:
        - `text`: The answer text.
        - `answer_start`: The character position where the answer begins in the context.
      - `is_impossible`: A boolean indicating if it's impossible to answer the question based on the context.

#### Usage

Use the `context` field from the JSON file as the source of your document text for processing. The PDF documents are included for reference and to allow you to see the original formatting and content if needed, but the JSON file should be used as the primary data source.

## Task Requirements

### Part 1: Basic RAG Pipeline Implementation (30%)

Build a basic RAG pipeline using open-source models that will answer the questions in the JSON file:

- Use Hugging Face models for:
  - LLM generation (e.g., [`meta-llama/Llama-3.2-1B`](https://huggingface.co/meta-llama/Llama-3.2-1B))
  - Sentence transformers for embeddings (e.g., [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2))
- Implement the following components:
  - Document chunking strategy using the `context` fields from the JSON file.
  - Embedding generation for each chunk.
  - Vector storage and indexing.
  - Query processing.
  - Answer generation based on the questions provided in the `question` fields of the JSON file.

#### Notes

Your RAG pipeline should process the questions from the JSON file and generate answers using the associated contexts.

While a working implementation is required, the primary focus of evaluation will be on Parts 2 and 3, so don't spend too much time configuring hyperparameters or tuning models.

> **The goal is to demonstrate your understanding of the pipeline and evaluation methodologies rather than achieving perfect performance. A simple implementation is sufficient.**

Any open-source models can be used for the implementation, not limited to the ones mentioned above.

Ensure that your code is well-documented and easy to understand. Include a brief description of the pipeline components and how they interact.

---

### Part 2: Pipeline Evaluation (40%)

Design and implement a comprehensive evaluation strategy for your pipeline. Consider:

1. Retrieval Quality:
   - Relevance of retrieved chunks.
   - Ranking accuracy.
   - Coverage of important information.

2. Answer Quality:
   - Accuracy compared to ground truth.
   - Answer completeness.
   - Factual consistency.

3. System Performance:
   - Latency measurements.
   - Resource utilization.
   - Scalability considerations.

Your evaluation should:

- Define clear metrics and justify their selection.
- Include both quantitative and qualitative analysis.
- Identify potential failure modes.
- Suggest improvements based on findings.

---

### Part 3: Deployment Planning (30%)

Outline a deployment strategy considering:

1. Production Architecture:
   - System components and interactions.
   - Scaling considerations.
   - Performance optimization.

2. Operational Requirements:
   - Monitoring and logging.
   - Error handling.
   - Update mechanisms.

3. Quality Assurance:
   - Testing strategy.
   - Validation processes.
   - Performance benchmarks.

> **Note:** You don't need to implement the deployment plan.

## Deliverables

1. **Code repository** containing:
   - A reproducible environment.
   - Implementation of the RAG pipeline.
   - Evaluation scripts and results.
   - Documentation of methods used.
   > Part 1 & 2 should be included in the code repository.

2. **Technical report** (max 1 page) covering:
   - Evaluation methodology and results.
   - Analysis of findings.
   - Deployment strategy and considerations.
   - Recommendations for improvements.
   > Part 3 should be included in the technical report.
   >
   > **Note:** The technical report can be in any format (e.g., PDF, Markdown) and should be concise and to the point.

## Evaluation Criteria

You will be assessed on:

- Thoroughness of evaluation methodology.
- Quality of analysis and insights.
- Practical considerations for deployment.
- Code quality and documentation.
- Critical thinking and problem-solving approach.

## Technical Constraints

- Use open-source Hugging Face models only.
- Python 3.8+.
- Use any libraries and frameworks you prefer.
- Ensure that your code is reproducible with clear instructions on how to run it.

## Time Allocation

- Recommended time: 3 hours.

> **Note:** The time allocation is a guideline. You can spend more or less time based on your preference. If you run out of time please submit what you have completed and what you would have done next.

## Extra Information

- Focus on evaluation methodology rather than achieving perfect performance.
- Document assumptions and trade-offs.
- Consider real-world applications and constraints.
- Use the `context` field from the JSON file as your parsed documents for downstream tasks.
- The PDFs are provided for reference and visualization purposes only.

## Submission Instructions

- Submit your code repository and technical report as a zip file. Include a README with instructions on how to run your code.
