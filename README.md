# Medical-Chatbot

# How to run?
### STEPS:

Start with cloning this repository

### STEP 01- Create a conda environment after opening the repository
```bash
conda create -n medibot python=3.10 -y
```
```bash
conda activate medibot
```

### STEP 02- Install the requirements
```bash
pip install -r requirements.txt
```

### STEP 03- Create a `.env` file in the root directory and add your Pinecone & Gemini credentials as follows:
```ini
PINECONE_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GEMINI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### STEP 04- Store embeddings to Pinecone
```bash
# Run the following command to store embeddings to pinecone
python store_index.py
```

### STEP 05- Run the application
```bash
# Finally run the following command
python app.py
```

### Techstack Used:

- Python
- LangChain
- Flask
- Gemini LLM
- Pinecone
