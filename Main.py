from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.chat_models.ollama import ChatOllama
from groq import Groq
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Create embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=False)

# Initialize vector store
db = Chroma(persist_directory="./db-mawared",
            embedding_function=embeddings)

# Create retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API")

# Initialize LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Create query rewriting prompt
query_rewrite_template = """
You are an AI assistant helping to improve search queries.
Your task is to rewrite the given question to be more effective for semantic search.
If the input is in English, expand it in English. If it's in Arabic, expand it in both Arabic and English.

Guidelines for rewriting:
1. Expand key terms and concepts
2. Include relevant synonyms
3. Keep the core intent of the question
4. Make it search-friendly

Original question: {question}

Rewritten question (provide only the rewritten question without explanation):
"""

query_rewrite_prompt = ChatPromptTemplate.from_template(query_rewrite_template)

# Create final RAG prompt template
template = """
You are a knowledgeable AI assistant. Your role is to provide comprehensive answers based on the given context.
IMPORTANT: You must ALWAYS respond in formal Arabic (العربية الفصحى), regardless of the language of the question.

Follow these guidelines:
1. ALWAYS write your response in Arabic, using proper Arabic grammar and punctuation
2. Use clear, eloquent Arabic language (الفصحى)
3. Be comprehensive yet concise
4. If context is insufficient, ask for clarification in Arabic
5. Maintain a friendly, professional tone in Arabic
6. If you reference technical terms, provide both Arabic and English terms in your explanation
7. Format numbers and dates in Arabic numerals when appropriate

Context:
{context}

Original Question: {original_question}

Rewritten Question: {question}

Generate your response in Arabic below:
"""

prompt = ChatPromptTemplate.from_template(template)

# Create the enhanced RAG chain with query rewriting
def rag_with_query_rewrite():
    def join_questions(input_dict):
        return {
            "context": input_dict["context"],
            "question": input_dict["rewritten_question"],
            "original_question": input_dict["original_question"]
        }

    return (
        {
            "original_question": RunnablePassthrough(),
            "rewritten_question": query_rewrite_chain,
        }
        | {
            "context": lambda x: retriever.get_relevant_documents(x["rewritten_question"]),
            "rewritten_question": lambda x: x["rewritten_question"],
            "original_question": lambda x: x["original_question"]
        }
        | join_questions
        | prompt
        | llm
        | StrOutputParser()
    )

# Initialize the RAG chain
rag_chain = rag_with_query_rewrite()

# Function to ask questions
def ask_question(question):
    print("\nQuestion:", question)
    print("\nالإجابة:", end=" ", flush=True)
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
    print("\n")

# Example usage
if __name__ == "__main__":
    print("\nWelcome! I'm your AI assistant. I'll understand your questions in English or Arabic and respond in Arabic.")
    while True:
        user_question = input("\nEnter your question (or type 'quit' to exit): ")
        if user_question.lower() == 'quit':
            break
        ask_question(user_question)
    print("\nThank you for using the AI assistant. Good bye!")
