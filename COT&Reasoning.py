
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

# Query Rewriting Prompt with Enhanced Reasoning
query_rewrite_template = """
You are an advanced AI query analysis assistant. Your task is to perform a deep, multi-step reasoning process to improve search queries.

Reasoning Steps:
1. Query Decomposition: Break down the original question into its core components
2. Intent Analysis: Identify the underlying intent and potential implicit information
3. Semantic Expansion: Generate semantically related terms and concepts
4. Query Reformulation: Create a more precise, search-friendly version of the query

Guidelines:
- If the input is in English, expand in English
- If the input is in Arabic, provide expansion in both Arabic and English
- Maintain the original question's core meaning
- Include domain-specific synonyms and related concepts

Original Question: {question}

Reasoning Process Output (provide only the final rewritten question):
"""

query_rewrite_prompt = ChatPromptTemplate.from_template(query_rewrite_template)

# Enhanced RAG Prompt with Chain of Thought (CoT) Reasoning
template = """
You are a sophisticated AI assistant performing a comprehensive reasoning process.

Chain of Thought Reasoning:
1. Context Analysis
   - Carefully examine the retrieved context documents
   - Identify key themes, concepts, and potential relationships
   - Note any potential contextual nuances or implicit information

2. Query Interpretation
   - Break down the original and rewritten questions
   - Identify primary and secondary information needs
   - Determine the specific type of information being sought

3. Reasoning Strategy
   - Select the most appropriate reasoning approach
   - Determine if the answer requires:
     a) Direct information retrieval
     b) Inference and synthesis
     c) Multi-step logical deduction

4. Answer Generation Process
   - Systematically construct the answer using:
     a) Direct contextual information
     b) Logical reasoning and inference
     c) Domain knowledge and conceptual understanding

5. Quality Assurance
   - Cross-verify answer against original context
   - Ensure comprehensiveness and accuracy
   - Maintain formal Arabic language standards

Important Constraints:
- ALWAYS respond in formal Arabic (العربية الفصحى)
- Use clear, eloquent language
- Provide technical terms in both Arabic and English
- Format numbers and dates in Arabic numerals
- Maintain a professional, scholarly tone

Context Documents:
{context}

Original Question: {original_question}

Rewritten Question: {question}

Reasoning-Based Response in Arabic:
"""

prompt = ChatPromptTemplate.from_template(template)

# Create the enhanced RAG chain with query rewriting and CoT reasoning
def rag_with_cot_reasoning():
    def join_questions(input_dict):
        return {
            "context": input_dict["context"],
            "question": input_dict["rewritten_question"],
            "original_question": input_dict["original_question"]
        }

    query_rewrite_chain = (
        query_rewrite_prompt
        | llm
        | StrOutputParser()
    )

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

# Initialize the RAG chain with CoT reasoning
rag_chain = rag_with_cot_reasoning()

# Function to ask questions with enhanced reasoning
def ask_question(question):
    print("\nالسؤال:", question)
    print("\nعملية التفكير والإجابة:", end=" ", flush=True)
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
    print("\n")

# Debugging function to show intermediate steps
def debug_reasoning(question):
    print("\n--- Debug: Reasoning Process ---")
    
    # Query Rewriting Step
    rewrite_chain = (
        query_rewrite_prompt
        | llm
        | StrOutputParser()
    )
    rewritten_query = rewrite_chain.invoke({"question": question})
    print("1. Query Rewriting:")
    print(f"   Original: {question}")
    print(f"   Rewritten: {rewritten_query}\n")
    
    # Context Retrieval Step
    retrieved_docs = retriever.get_relevant_documents(rewritten_query)
    print("2. Context Retrieval:")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"   Document {i} Snippet: {doc.page_content[:200]}...\n")

if __name__ == "__main__":
    print("\nمرحباً! أنا مساعدك الذكي. سأفهم أسئلتك بالعربية أو الإنجليزية وأجيب بالعربية مع عملية تفكير متقدمة.")
    while True:
        user_question = input("\nأدخل سؤالك (أو اكتب 'خروج' للإنهاء): ")
        if user_question.lower() in ['quit', 'خروج']:
            break
        
        # Choose between normal and debug mode
        mode = input("اختر الوضع (1: الإجابة العادية, 2: وضع التصحيح): ")
        if mode == '1':
            ask_question(user_question)
        elif mode == '2':
            debug_reasoning(user_question)
        else:
            print("اختيار غير صحيح. سيتم التنفيذ في الوضع العادي.")
            ask_question(user_question)
    
    print("\nشكراً لاستخدامك المساعد الذكي. إلى اللقاء!")
