import streamlit as st
import tempfile
import os

# --- 2026 STABLE IMPORTS ---
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain.agents import create_agent 

# ------------------------
# 1. Session State Init (MUST BE FIRST)
# ------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ------------------------
# 2. UI Setup
# ------------------------
st.set_page_config(page_title="Module 5 Project: AI Agent", layout="wide")
st.title("🚀 AI Agent: RAG + Calculator")

with st.sidebar:
    st.header("Authentication")
    google_api_key = st.text_input("Enter Google API Key:", type="password")
    st.info("Get a free key at [aistudio.google.com](https://aistudio.google.com/)")
    
    if st.button("Reset App & Clear Chat"):
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.rerun()

# ------------------------
# 3. Logic: PDF Ingestion
# ------------------------
uploaded_file = st.file_uploader("Upload Project PDF", type="pdf")

if uploaded_file and google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
    if st.session_state.vectorstore is None:
        try:
            with st.spinner("Building Knowledge Base..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                loader = PyPDFLoader(tmp_path)
                docs = loader.load_and_split(
                    RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                )
                
                embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
                st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
                os.remove(tmp_path)
                st.success("✅ PDF Processed for RAG!")
        except Exception as e:
            st.error(f"Upload Error: {e}")

# ------------------------
# 4. Logic: Tools Definition
# ------------------------
@tool
def pdf_search(query: str) -> str:
    """Use this tool to find information inside the uploaded PDF document."""
    if st.session_state.vectorstore is None:
        return "No PDF has been uploaded yet. Please tell the user to upload a document."
    
    docs = st.session_state.vectorstore.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])

@tool
def calculator(expression: str) -> str:
    """Use this tool for math. Input should be a math expression like '144**0.5 + 50'."""
    try:
        return str(eval(expression, {"__builtins__": None}, {}))
    except:
        return "Error: Please provide a valid math expression."

tools = [pdf_search, calculator]

# ------------------------
# 5. Logic: Chat Interface
# ------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask about the PDF or do math...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    if not google_api_key:
        st.warning("⚠️ Please enter an API key in the sidebar.")
    else:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash", 
                temperature=0,
                api_key=google_api_key
            )
            
            agent = create_agent(
                model=llm,
                tools=tools,
                system_prompt="You are a professional assistant. Use tools for math and PDF search."
            )

           
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):

                    response = agent.invoke({"messages": st.session_state.messages})

                    last_message = response["messages"][-1].content

                    # Fix Gemini structured output
                    if isinstance(last_message, list):
                        final_answer = ""
                        for block in last_message:
                            if isinstance(block, dict) and block.get("type") == "text":
                                final_answer += block.get("text", "")
                    else:
                        final_answer = str(last_message)

                    st.write(final_answer)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": final_answer}
                    )

        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                st.error("🛑 **Quota Reached:** You've hit the Gemini free tier limit (20 calls/day). Please wait a minute or try again tomorrow.")
            else:
                st.error(f"Agent Error: {e}")