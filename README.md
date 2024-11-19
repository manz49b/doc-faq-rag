# FAQ Generation System

This document outlines the design and implementation of a comprehensive evaluation strategy for a retrieval-augmented generation (RAG) pipeline used for generating FAQs from legal documents. The evaluation strategy was carefully crafted to assess different aspects of the system, including **retrieval quality**, **answer quality**, and **system performance**. Below, you will find the rationale for selecting specific metrics, as well as explanations of the methods and their relevance to the system's goals.

**Repository Structure** 
```
├── conda.yaml                         - base env file
├── data
│   ├── embeddings                     - stored ChromaDB embeddings
│   ├── input              
│   ├── output                         - output data from RAG process
│   ├── raw_pdfs      
│   └── steer                          - keywords to steer query
└── src
    ├── __init__.py
    ├── __pycache__
    ├── base.py                        - base references for pipeline
    ├── claude.py                      - claude functions
    ├── discovery.ipynb                - used for design/discovery/testing
    ├── evidently.ipynb                - to provide an initial evidently example
    ├── embeddings.py                  - embeddings functions
    ├── eval.py                        - eval methods 
    ├── gpt.py                         - gpt functions
    ├── main_embeddings.py             - entry point to rerun/create embeddings for files
    ├── main_rag.py                    - entry point to rerun FAQ RAG
    ├── prompt.py                      - prompt store
    ├── rag.py                         - rag methods
    ├── tests                          - for future unit tests
    └── utils.py                       - any extra tools/functions
```
---

**Install dependencies**:
Assuming you have conda, install Mamba (a fast, drop-in replacement for Conda) by running
```
conda install -n base -c conda-forge mamba
```
Set your base environment (replace the path with the actual path you want to use).
```
    export BASE_ENV_DIR=/opt/homebrew/Caskroom/mambaforge/base/envs
```

To install the conda environment, update the yaml file with your environment prefix and add a Jupyter kernel.
```
    mamba env create --file conda.yaml
```

Every time you update a package make sure you update the conda.yaml file and 
run the code below to auto update your environment.
```
    mamba env update --file conda.yaml --prune
```

You need an .env file containing:
OPENAI_API_KEY
VOYAGE_API_KEY

During the build/test work was explored with additional functionality but this is not required for a run:
CLAUDE_API_KEY
HF_TOKEN
MISTRAL_API_KEY
EVIDENTLY_API_KEY

**Run**

Assuming you have set your .env file and environment up, it is simple to rerun parts of the code as shown below.

However sample output data is stored an available here:
data/output/chunk_engineering/method-001/summary.parquet
data/output/main/method-001/data.parquet

Re-run the embeddings by running in terminal:
    ```bash
        python main_embeddings.py
    ```

Re-run the FAQ RAQ process by first updating the variable 'run_name' in main_rag.py, then in terminal run:
    ```bash
        python main_rag.py
    ```
Note: I have hardcoded limits on how many documents the FAQ rag will run within rag.py (e.g. for document in data[:2]:) due to unnessecary additional api costs.

# **Summary**

In this project, I developed a Retrieval-Augmented Generation (RAG) system designed to extract accurate, contextually relevant answers from legal documents. I focused on designing a robust evaluation methodology that would best assess retrieval accuracy and answer relevance, particularly in the complex domain of law. 

While I initially used Hugging Face models, I chose to leverage APIs instead due to hardware constraints with my local environment, which would not exist were I working in a cloud environment. This decision enabled me to move forward quickly, using cloud-based solutions for inference and optimizing the workflow around API integration rather than getting bogged down by the limitations of open-source models. I prioritized a flexible and scalable approach that could be adapted to varying hardware setups without compromising evaluation rigor.

Throughout the process, I identified that traditional evaluation methods, such as BLEU or ROUGE, did not always provide a clear measure of term relevance in legal contexts. To address this, I created custom evaluation functions that better captured the effectiveness of retrieval and accuracy, providing a clearer metric for assessing the system’s performance.

Looking ahead, I see opportunities to improve the system by incorporating a graph-based approach using Neo4j, allowing for more structured document chunking and more precise retrieval. Additionally, training a smaller model to classify document sections could improve chunk selection, enhancing both retrieval precision and overall system efficiency as the dataset grows.

This work aligns closely with the goals of building a scalable and reliable RAG system for legal documents, and the methodology I developed is well-documented below, in all files in this project and also with data examples and notebooks for deeper exploration.

---

# **Evaluation Metrics**

## 1. **Retrieval Quality**

### a. **Relevance of Retrieved Chunks**

- **Metrics Used:**
  - **BLEU** (Bilingual Evaluation Understudy)
  - **ROUGE-2** (Recall-Oriented Understudy for Gisting Evaluation)

- **Rationale for Selection:**
  - **BLEU** and **ROUGE-2** were used to measure the relevance of the retrieved chunks because these metrics are effective at capturing the **overlap of n-grams** between the retrieved context and the actual relevant content. In particular, **ROUGE-2** is suited for evaluating the overlap of bigrams, which is useful when assessing the context in a more granular manner, especially when multiple pieces of text are being considered.

### b. **Ranking Accuracy**

- **Metrics Used:**
  - **Mean Reciprocal Rank (MRR)**
  - **Precision at k (P@k)**, particularly **P@5**

- **Rationale for Selection:**
  - **MRR** measures the rank of the first relevant result across multiple queries, helping us understand how well the system is retrieving the most relevant chunks. A high MRR means that relevant results are ranked highly, which is crucial for effective RAG-based systems.
  - **P@k** is used to measure how many of the top k results are relevant, providing insight into the system's ranking ability. **P@5** is particularly useful because it directly assesses how well the top 5 results contain the most relevant information.

### c. **Coverage of Important Information**

- **Metric Used:**
  - **Percentage of answers that exist in context**

- **Rationale for Selection:**
  - This metric is designed to evaluate how much of the relevant answer can be found within the context retrieved. Ensuring that the context contains all the important information is vital for generating accurate and useful FAQs.

---

## 2. **Answer Quality**

### a. **Accuracy Compared to Ground Truth**

- **Metrics Used:**
  - **Containment Method**
  - **N-gram Precision, Recall, and F1**
  - **ROUGE-1 Precision, Recall, and F1**

- **Rationale for Selection:**
  - The **containment method** is used to measure if the generated answer is found within the ground truth. This method helps us assess the basic **accuracy** of the answer in the context of the known correct answer.
  - **N-gram Precision, Recall, and F1** metrics were chosen to provide a more detailed analysis of the model's ability to match the ground truth at different levels of granularity. N-gram metrics allow us to evaluate how well the generated answer covers the **same terms** and **concepts** as the ground truth, which is critical for assessing correctness.
  - **ROUGE-1** is a natural choice for evaluating the overlap of unigrams (individual words) between the generated and reference texts. It gives us a reliable metric for comparing the overall content, making it easier to identify how well the model captures the essential information.

### b. **Answer Completeness**

- **Metrics Used:**
  - **N-gram Overlaps (Precision, Recall, F1)**
  - **Ground Truth Containment**

- **Rationale for Selection:**
  - **N-gram overlaps** are used again to assess how well the generated response includes the necessary components to form a complete answer. By examining the **precision**, **recall**, and **F1 score** of overlapping n-grams, we gain insights into how well the system is capturing the full range of information in the answer.
  - The **containment method** is revisited here to ensure that not only accuracy but also the completeness of the answer is evaluated. If the model’s response doesn’t fully contain the key information from the ground truth, it can be flagged as incomplete.

### c. **Factual Consistency**

- **Metric Used:**
  - **Factual Consistency Check** (manual inspection or automated fact-checking)

- **Rationale for Selection:**
  - Since the FAQ system is based on legal documents, factual consistency is crucial. We either manually or automatically verify that the generated answer aligns with the factual content of the source document. This helps to ensure that the system provides **factually accurate** answers, particularly in sensitive domains like legal information.

---

## 3. **System Performance**

### a. **Latency Measurements**

- **Metric Used:**
  - **Time Taken (Response Time)**

- **Rationale for Selection:**
  - **Latency** is measured by the time taken to generate a response, which is critical for user experience, especially in production environments. Minimizing response time ensures that the system can scale to handle multiple requests without significant delays.

### b. **Resource Utilization**

- **Metric Used:**
  - **Resource Utilization (avoided with external API usage)**

- **Rationale for Selection:**
  - Due to the system's reliance on external APIs (e.g., OpenAI), we don’t need to directly monitor resource utilization for the core operations, but we do consider it indirectly through API cost and usage efficiency. Ensuring minimal computational overhead and cost per query is a priority for large-scale deployment.

### c. **Scalability Considerations**

- **Metric Used:**
  - **Scalability** of the pipeline is considered, though not directly measured in this evaluation.

- **Rationale for Selection:**
  - While **scalability** was not directly measured, it was a key consideration when designing the pipeline. We anticipate that scaling to handle a large number of queries should maintain the same level of accuracy and performance.

---

## 4. **Impossible Count**

- **Metric Used:**
  - **Impossible Answer Count**

- **Rationale for Selection:**
  - This metric tracks how often the model correctly negates providing an answer when the answer is **impossible** (i.e., the question cannot be answered). This is crucial for understanding the **reliability** of the model in scenarios where it might encounter questions without clear answers or those outside its training data.

---

## Evaluation Strategy Conclusion

My evaluation strategy is designed to provide a **holistic assessment** by using a combination of different metrics, we can evaluate the system’s performance from various angles, ensuring that it generates **accurate**, **reliable**, and **contextually relevant** answers. Furthermore, the selected metrics allow us to track performance changes over time and make informed decisions about how to **improve** the system.

# **Model Choice Explanation**

The model selection for the RAG pipeline was a critical decision, and after evaluating open-source alternatives, I opted for a pragmatic approach, given the project's timeline and the objective being focused on evaluating the RAG pipeline rather than implementing the most optimized solution, the following models were chosen:

- **Claude and ChatGPT**: For the **LLM question retrieval**, Claude Sonnet (for its understanding of complex queries and answers, and its latest ranking over GPT4o), however due to hitting daily rate limits, and being confined to time limits, I switched to ChatGPT (as it is known to perform fairly well). These models are reliable and widely used, offering a good balance of performance and accessibility. While open-source models like **Mistral 7B** could provide a more scalable and cost-effective solution in the future, Claude and ChatGPT were chosen due to their availability and ease of use for rapid experimentation.
  
- **Voyage**: For **embeddings**, **Voyage** was selected as it excels in transforming text into embeddings with high-quality retrieval capabilities. This choice was driven by its ability to handle large-scale document indexing and retrieval efficiently, ensuring relevant chunks of information can be fetched quickly.

### **Future Considerations:**
Though **Claude** and **ChatGPT** are suitable for the current project’s scope, scalability and cost-efficiency considerations could be addressed by switching to **Mistral 7B** in combination with **fine-tuning** for a tailored question-answering solution. Fine-tuning with **question-answer examples** could help Mistral produce as high-quality results as larger models, when trained on items specific to the task. Furthermore, we could implement **Multi LoRA** (Low-Rank Adaptation) to host hundreds of fine-tuned adapters across one GPU, significantly reducing costs.

---

## **Scalability Strategy**

While using **pay-to-use endpoints** like Claude and ChatGPT is easy to scale in the short term, there are strategies for reducing costs and improving efficiency in the long term:

### **Scalability Approaches:**
1. **Mistral 7B and Fine-tuning**:
   - By using the **Mistral 7B model** and fine-tuning it with domain-specific **question-answer datasets**, we can significantly reduce the cost compared to using third-party APIs for every inference request. Mistral 7B offers high performance with fine-tuning for large-scale language models and as mentioned can be optimized for cost-efficiency.
   
2. **Multi LoRA for Parallel Inference**:
   - **Multi LoRA** can be employed to distribute **fine-tuned adapters** across a single GPU, significantly improving scalability. For example, on an AWS instance like **g5.2xlarge**, this setup can enable up to **100x faster inference** without sacrificing performance. This approach dramatically reduces infrastructure costs, enabling better efficiency at scale.

### **Cost Reduction and Efficiency**:
- These techniques would cut costs by up to **tenfold** and allow for **parallel inference** over multiple instances, which has been trialed and proven in various settings.

### **Operational Infrastructure**:
- Robust **unit tests** should be in place to ensure minimal bugs during deployment. We could implement **Slack alerts** for bug tracking this would ensure that the **ML Ops engineer** is immediately notified when backend changes are necessary (e.g., new file formats or schema changes).

---

## **Architecture Proposal**

### **Overview**:
The architecture would be based on a containerized approach (using **Docker** or **AWS Lambda**) to handle different aspects of the RAG system, such as document processing, question retrieval, and answer generation. This modular approach ensures flexibility, scalability, and maintainability of the system.

### **System Components**:

1. **Frontend (User Interface)**:
   - The **frontend** will allow users to upload their documents, submit queries, and receive generated FAQs. It can be built using frameworks like **React** or **Vue.js**, which integrate seamlessly with the backend API for querying documents and displaying results.
   - Users can provide feedback (e.g., thumbs up/down) and suggest improvements to the tool, which will help optimize the model over time.

2. **Backend (Lambda/Containerized Services)**:
   - The **backend** would be containerized (e.g., using **Docker**) or implemented via **AWS Lambda** to handle user queries, process uploaded documents, retrieve relevant chunks, and generate answers using LLMs. It could also include services for **document indexing**, **retrieval**, and **answer generation**.
   - It may be useful to use something like AWS SQS, simple queue service would enable queing of requests to the tool if it served across the entire business for example.
   - When users upload new documents, the system will process and index them into a storage solution (e.g., **S3** or **Azure Blob Storage**) for fast querying later.
   
3. **Document Storage**:
   - **Amazon S3** or **Azure Blob Storage** would be used to store and cache the documents uploaded by users. These services allow for high availability and fast retrieval, ensuring the system can handle a large volume of documents efficiently.
   
4. **Embeddings and Retrieval**:
   - For document retrieval, **Voyage** (or a similar internally hosted embeddings model) will index documents. If we do internally host models of course they would need to be made available as endpoints. When a user submits a query, embeddings of the query are compared with stored document embeddings to find the most relevant context.
   
5. **LLM Answer Generation**:
   - After retrieving the relevant document chunks, the system will use **Claude**, **ChatGPT**, or again an internally hosted LLM say fine-tuned mistral adapter, which could be hosted on a SageMaker Endpoint for example, to generate answers based on the retrieved context. The LLM will form answers tailored to the specific query.

### **Monitoring and Drift Analysis**:
- To ensure the system maintains its accuracy over time, **Evidently** and/or **MLFlow** could be integrated to log key evaluation metrics, model type, prompt used, etc. An initial discovery notebook exists for evidently in this repository with an example of storing sentiment, apology words where LLMs can go off track apologising or hallucinating. However it would be best to invest time in designing custom logging with our decided metrics (e.g., **accuracy**, **precision**, **recall**) for both **retrieval** and **generation** stages. This helps monitor **drift** in performance over time and identify any issues like model **hallucination** or **context loss**.

- **Drift monitoring** can be particularly useful to track any deviations from expected performance and ensure that the model doesn’t start generating inconsistent or incorrect responses. Additionally, the integration of a second LLM (e.g., **ChatGPT** or another model) could provide a **performance review** to cross-check the outputs of the primary model.

### **User Feedback Loop**:
- **User Feedback** will be essential to continuously improve the system. We can use tools like **Nebuly** to automate the feedback collection process. User responses (thumbs up/down, comments, etc.) will be captured and analyzed to adapt the system over time.
  - **A/B testing** will help assess new features or model adjustments to gather user preferences.
  - **Real-time prompt suggestions** will allow the system to adapt dynamically, improving the user experience.
  
### **Architecture Diagram**:
The following components interact within the architecture:
1. **User Interface** (Frontend)
   - Upload document, submit queries, provide feedback.
2. **AWS Lambda/Containers** (Backend)
   - Query handling, document processing, LLM interaction.
3. **Document Storage** (S3/Blob)
   - Cached documents for fast retrieval.
4. **Voyage Embeddings** (Retrieval)
   - Retrieve relevant document chunks using embeddings.
5. **Claude/ChatGPT LLM** (Answer Generation)
   - Generate responses based on retrieved context.

---

## **Deployment Plan**

### **1. Production Architecture**:
- **System Components**: The system will be comprised of the **Frontend**, **Backend**, and **Storage** components mentioned above.
- **Scaling Considerations**: We will use **AWS Lambda** or **containerized services** to scale the backend horizontally. **Auto-scaling** groups will be set up to handle the increased load during high traffic.
- **Performance Optimization**: Caching strategies for commonly used queries, and **indexing** will be implemented to improve retrieval speed. **Model fine-tuning** will ensure efficient use of resources.

### **2. Operational Requirements**:
- **Monitoring and Logging**: Use of **CloudWatch** (AWS) for continuous monitoring of the backend, query performance, and storage health. **Evidently** will be integrated for drift monitoring.
- **Error Handling**: **Graceful error handling** and logging will be implemented to ensure that users are notified of any issues. Alerts for system failures will be sent via **Slack** or other channels.
- **Update Mechanisms**: Updates to the backend, LLM, and frontend will be handled through CI/CD pipelines, ensuring seamless deployment and rollback capabilities.

### **3. Quality Assurance**:
- **Testing Strategy**: A combination of **unit tests** and **integration tests** will be used to ensure the correctness of the code. Additionally, **end-to-end tests** will ensure the system works smoothly across components.
- **Validation Processes**: Performance validation will be carried out periodically to assess model accuracy, retrieval relevance, and overall user satisfaction.
- **Performance Benchmarks**: Latency, response time, and resource utilization benchmarks will be set to ensure that the system can scale effectively.

