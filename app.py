import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever

# --- 1. PAGE CONFIGURATION & BUG FIX ---
st.set_page_config(
    page_title="Softora Audit Engine", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. ADVANCED ANIMATED CSS ---
custom_css = """
<style>
    /* Hide watermarks without destroying the sidebar toggle */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent !important;}
    
    /* Custom Sleek Scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #050505; }
    ::-webkit-scrollbar-thumb { background: #3B82F6; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #00D2FF; }

    /* Global Deep Space Background */
    .stApp {
        background: radial-gradient(circle at top right, #0A1128, #050505 60%);
        color: #FFFFFF;
    }
    
    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(15, 20, 30, 0.6) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-right: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    /* Animations */
    @keyframes slideUpFade {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 8px rgba(59, 130, 246, 0.3); }
        50% { box-shadow: 0 0 20px rgba(59, 130, 246, 0.7); }
        100% { box-shadow: 0 0 8px rgba(59, 130, 246, 0.3); }
    }

    /* Base Text Readability */
    .stMarkdown p, .stMarkdown li {
        font-size: 16px !important;
        line-height: 1.7 !important;
        color: #F8FAFC !important;
        letter-spacing: 0.2px;
    }
    
    /* User Chat Bubble (Blue Glass) */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background: rgba(0, 30, 60, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-left: 5px solid #3B82F6;
        animation: slideUpFade 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    
    /* AI Chat Bubble (Emerald Glass) */
    [data-testid="stChatMessage"]:nth-child(even) {
        background: rgba(5, 30, 20, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-left: 5px solid #10B981;
        animation: slideUpFade 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    
    /* High-Visibility Metrics */
    [data-testid="stMetricValue"] {
        color: #00D2FF !important;
        font-weight: 800;
        text-shadow: 0 0 10px rgba(0, 210, 255, 0.3);
    }
    
    /* Input Fields & Dropdowns */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
        background: rgba(20, 25, 40, 0.8) !important;
        color: #FFFFFF !important;
        border: 1px solid #3B82F6 !important;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stTextInput input:focus {
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.5) !important;
    }
    
    /* Glowing Primary Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        color: #FFFFFF !important;
        font-size: 16px;
        font-weight: 700;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0.7rem;
        transition: all 0.3s ease;
        animation: pulseGlow 3s infinite;
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        background: linear-gradient(90deg, #2563EB 0%, #60A5FA 100%);
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.6);
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- 3. INITIALIZATION ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.markdown("<h1 style='text-align: center; color: #FFFFFF; text-shadow: 0 0 15px rgba(59, 130, 246, 0.5);'>⚡ Intelligent Audit Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #10B981; font-weight: 600; letter-spacing: 1px;'>ENTERPRISE HYBRID RETRIEVAL (BM25 + FAISS) | AI CROSS-ENCODER</p>", unsafe_allow_html=True)
st.divider()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0

# --- 4. HYBRID PIPELINE WITH ROBUST ERROR HANDLING ---
def create_vector_embeddings():
    # Session state lock removed to allow dynamic rebuilding upon new uploads
    with st.status("Initializing Quantum Data Pipeline...", expanded=True) as status:
        st.write("📂 Extracting metadata and parsing layouts...")
        
        # PATCH A: Auto-create directory if missing to prevent crashes
        if not os.path.exists("./documents"):
            os.makedirs("./documents")
            status.update(label="Created 'documents' folder. Please add PDFs and try again.", state="error", expanded=False)
            return
        
        docs = []
        for file in os.listdir("./documents"):
            if file.endswith(".pdf"):
                file_path = os.path.join("./documents", file)
                loader = PyMuPDFLoader(file_path)
                docs.extend(loader.load())
        
        if not docs:
            status.update(label="No PDFs found in the 'documents' directory.", state="error", expanded=False)
            return

        st.session_state.doc_count = len(docs)
        st.write("✂️ Chunking documents...")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = st.session_state.text_splitter.split_documents(docs)
        
        st.write("🧠 Building Semantic Engine (FAISS)...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(final_documents, embeddings)
        st.session_state.faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
        
        st.write("🔍 Building Keyword Engine (BM25)...")
        st.session_state.bm25_retriever = BM25Retriever.from_documents(final_documents)
        st.session_state.bm25_retriever.k = 15
        
        st.write("⚖️ Initializing Cross-Encoder Re-ranker...")
        st.session_state.reranker_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        status.update(label="Enterprise Pipeline Online", state="complete", expanded=False)
        st.toast("System is ready for querying.", icon="✅")

# --- 5. SIDEBAR DASHBOARD ---
with st.sidebar:
    st.title("⚙️ Telemetry")
    
    selected_model = st.selectbox(
        "🧠 Core Intelligence Engine",
        ("llama-3.3-70b-versatile", "llama-3.1-8b-instant")
    )
    
    col1, col2 = st.columns(2)
    col1.metric("Status", "Online", "Secure")
    col2.metric("Pages Indexed", f"{st.session_state.doc_count}", "Ready")
    
    st.divider()
    st.write("📂 **Document Upload**")
    
    # Streamlit File Uploader
    uploaded_file = st.file_uploader("Upload a PDF to audit", type=["pdf"])
    
    if uploaded_file:
        # Save the uploaded file temporarily so PyMuPDF can read it
        if not os.path.exists("./documents"):
            os.makedirs("./documents")
        temp_file_path = os.path.join("./documents", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded: {uploaded_file.name}")
        
    st.divider()
    st.write("🚀 **Pipeline Controls**")
    
    if st.button("Initialize Pipeline"):
        # Check if the folder has files or if a new file was uploaded
        if not uploaded_file and (not os.path.exists("./documents") or not [f for f in os.listdir("./documents") if f.endswith(".pdf")]):
            st.error("Please upload a PDF first!")
        else:
            create_vector_embeddings()

    if st.button("Wipe Memory Logs"):
        st.session_state.chat_history = []
        st.rerun()
        
    st.divider()
    st.markdown("### 👨‍💻 System Architect")
    st.markdown("**Abdul Ghafoor** | Lead Data Engineer")
    st.markdown("*Softora Core Engineering Team*")
    st.markdown("[www.softorapk.com](https://www.softorapk.com)")

# Initialize LLM with the stable dropdown selection
llm = ChatGroq(groq_api_key=groq_api_key, model_name=selected_model)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert enterprise document audit agent. 
    Answer questions accurately based ONLY on the following context. 
    The context includes metadata like [Source File] and [Page]. Always cite the source file and page number when providing facts.
    If the answer is not contained in the context, say 'I cannot find the answer in the provided documents.'
    
    Context:
    {context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# --- 6. CHAT INTERFACE ---
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

user_query = st.chat_input("Enter your complex audit query...")

if user_query:
    if "faiss_retriever" not in st.session_state:
        st.error("⚠️ Please initialize the data pipeline from the sidebar first.")
    else:
        with st.chat_message("user"):
            st.markdown(user_query)
            
        with st.chat_message("assistant"):
            with st.spinner("Executing Hybrid Search & AI Cross-Validation..."):
                
                # Hybrid Search + RRF + Cross-Encoder
                faiss_docs = st.session_state.faiss_retriever.invoke(user_query)
                bm25_docs = st.session_state.bm25_retriever.invoke(user_query)
                
                doc_scores = {}
                k = 60
                for rank, doc in enumerate(faiss_docs):
                    doc_scores[doc.page_content] = doc_scores.get(doc.page_content, 0) + 1 / (rank + k)
                for rank, doc in enumerate(bm25_docs):
                    doc_scores[doc.page_content] = doc_scores.get(doc.page_content, 0) + 1 / (rank + k)
                
                unique_docs = {doc.page_content: doc for doc in faiss_docs + bm25_docs}
                fused_docs = [unique_docs[content] for content, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)]
                base_docs = fused_docs[:15]
                
                pairs = [(user_query, doc.page_content) for doc in base_docs]
                scores = st.session_state.reranker_model.score(pairs)
                
                scored_docs = list(zip(base_docs, scores))
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                retrieved_docs = [doc for doc, score in scored_docs[:5]]
                
                context_text = ""
                for doc in retrieved_docs:
                    source_file = os.path.basename(doc.metadata.get('source', 'Unknown File'))
                    page_num = doc.metadata.get('page', 0) + 1
                    context_text += f"\n\n[Source: {source_file} | Page: {page_num}]\n{doc.page_content}"
                
                # PATCH B: Graceful Execution with Try-Except for API Limits
                try:
                    chain = prompt | llm | StrOutputParser()
                    answer = chain.invoke({
                        "context": context_text, 
                        "input": user_query,
                        "chat_history": st.session_state.chat_history
                    })
                    st.markdown(answer)
                except Exception as e:
                    answer = "⚠️ **System Alert:** The AI engine is currently experiencing high latency or API restrictions. Please try your query again in a moment."
                    st.error(f"Internal Diagnostic: {str(e)}")
                    st.markdown(answer)
                
                # Expandable Audit Trail
                with st.expander("🔍 View Verified Source Evidence"):
                    if "System Alert" not in answer:
                        for i, doc in enumerate(retrieved_docs):
                            source_file = os.path.basename(doc.metadata.get('source', 'Unknown File'))
                            page_num = doc.metadata.get('page', 0) + 1
                            st.info(f"**Evidence #{i+1} | Document: `{source_file}` | Page: {page_num}**")
                            st.caption(f'"{doc.page_content[:400]}..."')
                        
        st.session_state.chat_history.extend([
            HumanMessage(content=user_query),
            AIMessage(content=answer)
        ])