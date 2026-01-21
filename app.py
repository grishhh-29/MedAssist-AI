from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os

# Langchain imports
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.memory import ConversationBufferWindowMemory

# Local imports
from src.prompt import *
from src.helper import get_embeddings

# Gemini
import google.generativeai as genai

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY


# embeddings = download_hugging_face_embeddings()
print("Initializing embeddings...")
embeddings = get_embeddings()


# index_name = "medical-chatbot" 
index_name = "medical-chatbot-gemini"
namespace = "medical-docs"

print(f"Connecting to Pinecone index: {index_name}")
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name = index_name,
    embedding = embeddings,
    namespace="medical-docs"
)
# retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# With this:
retriever = docsearch.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.5  # Only return docs with similarity > 0.5
    }
)



print("Configuring Gemini model...")
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")


print("Setting up memory...")

memory = ConversationBufferWindowMemory(
    k=5,
    return_messages=True,
    memory_key="chat_history"
)


def gemini_runnable(messages):
    prompt_parts = []
    for m in messages:
        if isinstance(m, tuple):
            role, text = m
            prompt_parts.append(f"{role.upper()}: {text}")
        else:
            prompt_parts.append(m.content)
    prompt = "\n".join(prompt_parts)
    
    try:
        # response = model.generate_content(prompt)
        # return response.text
        response = model.generate_content(prompt)
        response_text = response.text
        
        # Replace the disclaimer with styled HTML version
        disclaimer = "Please consult a healthcare professional for personalized medical advice."
        if disclaimer in response_text:
            styled_disclaimer = f'<span style="font-size: 0.75em; font-style: italic;">{disclaimer}</span>'
            response_text = response_text.replace(disclaimer, styled_disclaimer)
        
        return response_text        
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"I apologize, but I encountered an error: {str(e)}"



chatModel = RunnableLambda(gemini_runnable)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)



print("Creating RAG chain...")
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)



@app.route("/")
def index():
    return render_template("chat.html")



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    # print(input)
    # response = rag_chain.invoke(
    #     {
    #         "input": msg,
    #         "chat_history": memory.load_memory_variables({})["chat_history"]
    #     }
    # )

    # memory.save_context(
    #     {"input": msg},
    #     {"output": response["answer"]}
    # )

    # print("Response : ", response["answer"])
    # return str(response["answer"])
    
    # # return render_template('chat.html')
    try:
        # Get response from RAG chain
        response = rag_chain.invoke(
            {
                "input": msg,
                "chat_history": memory.load_memory_variables({})["chat_history"]
            }
        )
        
        # Save to memory
        memory.save_context(
            {"input": msg},
            {"output": response["answer"]}
        )
        
        # print(f"Response: {response['answer']}")
        return str(response["answer"])
    
    except Exception as e:
        error_msg = f"Sorry, I encountered an error: {str(e)}"
        print(f"Error: {error_msg}")
        return error_msg


@app.route("/health")
def health():
    """Health check endpoint for Render."""
    return jsonify({"status": "healthy"})

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port= 8080, debug= True)

if __name__ == "__main__":
    # port = int(os.environ.get("PORT", 10000))
    # app.run(host="0.0.0.0", port=port)
    print("="*60)
    print("MedAssist AI Starting...")
    print("="*60)
    
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)