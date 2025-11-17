import os
import json
import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Config
api_key = os.getenv("OPENROUTER_API_KEY")
embedding_model = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")
llm_model = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")

# Set up embeddings and LLM with OpenRouter
embeddings = OpenAIEmbeddings(
    model=embedding_model,
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
)
llm = ChatOpenAI(
    model=llm_model,
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
)

# Load vectorstore
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Retriever (k-NN top 5)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)

# Format docs helper
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Prompt for answer generation
prompt_template = """Use ONLY the following chunks of context to answer the question. If you don't know the answer, say so.

Context: {context}

Question: {question}

Answer:"""
PROMPT = PromptTemplate.from_template(prompt_template)

# RAG runnable chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

# Bonus: Evaluator prompt
eval_prompt_template = """Evaluate the answer quality (0-10) based on relevance to chunks, accuracy, and completeness.

User question: {user_question}
System answer: {system_answer}
Chunks: {chunks_related}

Score (0-10): 
Reason: """
EVAL_PROMPT = PromptTemplate.from_template(eval_prompt_template)

eval_runnable = EVAL_PROMPT | llm

def query_rag(question, evaluate=False):
    answer = rag_chain.invoke(question)
    
    sources_with_scores = vectorstore.similarity_search_with_score(question, k=5)
    sources = [doc for doc, score in sources_with_scores]
    chunks_related = [doc.page_content for doc in sources]
    
    output = {
        "user_question": question,
        "system_answer": answer,
        "chunks_related": chunks_related
    }
    
    if evaluate:
        eval_input = {
            "user_question": question,
            "system_answer": answer,
            "chunks_related": "\n".join(chunks_related)
        }
        eval_result = eval_runnable.invoke(eval_input)
        eval_text = eval_result.content.strip()
        
        score = "N/A"
        reason = "N/A"
        
        if "Score" in eval_text:
            score_part = eval_text.split("Score")[1].split("\n")[0].strip()
            if ":" in score_part:
                score = score_part.split(":")[1].strip()
        
        lines = eval_text.split("\n")
        if len(lines) > 1:
            reason = " ".join(lines[1:]).strip()
            if "Reason" in reason and ":" in reason:
                reason = reason.split(":")[1].strip()
        
        output["score"] = score
        output["reason"] = reason
    
    return output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str, help="User question")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate answer")
    args = parser.parse_args()
    
    output = query_rag(args.question, args.evaluate)
    
    printed_output = json.dumps(output, indent=2)
    print(printed_output)
    
    os.makedirs("outputs", exist_ok=True)
    
    samples_path = "outputs/sample_queries.json"
    samples = []
    if os.path.exists(samples_path):
        try:
            with open(samples_path, "r") as f:
                samples = json.load(f)
        except json.JSONDecodeError:
            samples = []
    samples.append(output)
    with open(samples_path, "w") as f:
        json.dump(samples, f, indent=2)
    
    print(f"\nLogged to {samples_path}")