# Medical-Chatbot

A retrieval-augmented generation (RAG) medical Q&A chatbot built with LangChain, Pinecone, Flask, and Ollama (Llama 3) for local, low-cost inference.

# How to run?
### STEPS:

Clone the repository
```bash
git clone https://github.com/Scar570/Medical-Chatbot.git
```

### STEP 01 - Create a conda environment after opening the repository
```bash
conda create -n medibot python=3.10 -y
```
```bash
conda activate medibot
```

### STEP 02 - install the requirements
```bash
pip install -r requirements.txt
```

### Create a `.env` file in the root directory and add your Pinecone & OpenAI credentials as follows:
```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

```bash
# In this version we use Ollama running locally. Make sure you have Ollama installed
# and have pulled a model such as llama3:instruct
ollama pull llama3:instruct
```

```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:8080
```

### Techstack Used:
- Python
- LangChain
- Flask
- Ollama (LLM backend)
- Pinecone

---

# Switching Between Ollama and OpenAI
The chatbot supports both Ollama (local models) and OpenAI GPT (cloud API)

## To use Ollama (default in this repo):
```bash
# Install Ollama and pull a model:
ollama pull llama3:instruct
```
- No `OPENAI_API_KEY` is required in `.env`.

## To use OpenAI instead:
```bash
# Add the following line to your .env:
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```
- Update `app.py` to use `ChatOpenAI` from `langchain_openai` instead of `ChatOllama`.
- This is useful if you prefer cloud-hosted inference instead of running locally.

---

## Troubleshooting

If you hit errors like `ModuleNotFoundError: No module named 'langchain_ollama'`,
`langchain_huggingface`, or import errors from `langchain.chains` / `langchain.schema`,
this is due to LangChain's v1.0 breaking changes (the old import paths used in this
codebase were removed). Fix by installing the missing packages and pinning
LangChain to a compatible 0.3.x line:

```bash
pip install langchain_ollama langchain_huggingface
pip install "langchain==0.3.*" "langchain-community==0.3.*" "langchain-core==0.3.*" "langchain-text-splitters==0.3.*"
```

---

## Evaluation

See `/evaluation` for the evaluation harness used to measure system performance:
- `eval_retrieval.py` / `eval_retrieval_advanced.py` — retrieval quality (Recall@3, MRR)
- `eval_test_set_10.csv` — 10-question test set built from real entries in the source PDF
- `eval_answers_template_10_FINAL.csv` — manually graded answer accuracy

**Results:**
- Recall@3: 0.80
- MRR: 0.667
- Answer Accuracy: 90% (9/10, manually graded)

## Known Limitations

- Duplicate vectors currently exist in the Pinecone index from re-running `store_index.py`
  multiple times without clearing the prior index first, which reduces retrieval precision
  (Precision@3 ≈ 0.47). Fix: delete and rebuild the index cleanly.
- Average response latency is ~2 minutes on local CPU-based Llama 3 inference via Ollama.

---

# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment
With specific access:
1. **EC2 access** — virtual machine
2. **ECR** — Elastic Container Registry to save your docker image in AWS

**Description: About the deployment**
1. Build docker image of the source code
2. Push your docker image to ECR
3. Launch your EC2
4. Pull your image from ECR in EC2
5. Launch your docker image in EC2

**Policy:**
1. AmazonEC2ContainerRegistryFullAccess
2. AmazonEC2FullAccess

## 3. Create ECR repo to store/save docker image
- Save the URI: `<your-aws-account-id>.dkr.ecr.<your-region>.amazonaws.com/medibot`

## 4. Create EC2 machine (Ubuntu)

## 5. Open EC2 and Install docker in EC2 Machine:

```bash
# optional
sudo apt-get update -y
sudo apt-get upgrade

# required
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

## 6. Configure EC2 as self-hosted runner:
Settings > Actions > Runner > New self hosted runner > choose OS > then run commands one by one

## 7. Setup GitHub secrets:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`
- `ECR_REPO`
- `PINECONE_API_KEY`
- `OPENAI_API_KEY` (if you use OpenAI instead of Ollama)
