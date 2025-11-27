"""
Simplified Multi-Modal Data Intelligence Agent - Web Interface
No matplotlib dependency issues - uses Plotly exclusively!
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, Any, Optional
import io

# Visualization
import plotly.express as px
import plotly.graph_objects as go

# PDF Processing
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("PDF processing not available. Install pymupdf to enable.")

# Google Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Agent System
try:
    from agent_system import run_autonomous_pipeline
    AGENT_SYSTEM_AVAILABLE = True
except ImportError:
    AGENT_SYSTEM_AVAILABLE = False
    st.error("⚠️ agent_system.py not found! Make sure it's in the same folder.")

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Multi-Modal Data Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'data_files' not in st.session_state:
    st.session_state.data_files = {}

if 'pdf_files' not in st.session_state:
    st.session_state.pdf_files = {}

if 'pdf_summaries' not in st.session_state:
    st.session_state.pdf_summaries = {}

if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = {}

if 'llm' not in st.session_state:
    st.session_state.llm = None

# ============================================================================
# PDF PROCESSING FUNCTIONS
# ============================================================================

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file."""
    if not PDF_AVAILABLE:
        return "PDF processing not available"
    
    try:
        pdf_bytes = pdf_file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.get_text()
        
        pdf_document.close()
        return text
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

def summarize_pdf(text: str, llm) -> str:
    """Summarize PDF content using LLM."""
    if not GEMINI_AVAILABLE or llm is None:
        return "LLM not available for summarization"
    
    try:
        # Truncate if too long
        max_chars = 15000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... (truncated)"
        
        prompt = f"""Analyze this document and provide:
1. Main topics covered
2. Key findings or recommendations
3. Important statistics or data points
4. Overall purpose of the document

Document content:
{text}

Provide a clear, structured summary."""

        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error summarizing: {str(e)}"

# ============================================================================
# LLM INITIALIZATION
# ============================================================================
def initialize_llm(api_key: str):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')  # ✅ NEW
        st.session_state.llm = model
        return True
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {str(e)}")
        return False
# ============================================================================
# CHAT FUNCTIONS
# ============================================================================

def chat_with_data(question: str, data_files: Dict[str, pd.DataFrame]) -> str:
    """Chat about the data using LLM."""
    if not st.session_state.llm:
        return "⚠️ Please enter your Gemini API key first!"
    
    try:
        # Create context from data
        context = "Available datasets:\n\n"
        for filename, df in data_files.items():
            context += f"\n**{filename}**:\n"
            context += f"- Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
            context += f"- Columns: {', '.join(df.columns.tolist())}\n"
            context += f"- Sample stats:\n{df.describe().to_string()}\n"
        
        prompt = f"""You are a data analyst. Answer this question about the data:

{context}

Question: {question}

Provide a clear, specific answer based on the data above."""

        response = st.session_state.llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def chat_with_pdfs(question: str, pdf_summaries: Dict[str, str]) -> str:
    """Chat about PDFs using LLM."""
    if not st.session_state.llm:
        return "⚠️ Please enter your Gemini API key first!"
    
    if not pdf_summaries:
        return "No PDF documents have been uploaded yet."
    
    try:
        context = "Document summaries:\n\n"
        for filename, summary in pdf_summaries.items():
            context += f"\n**{filename}**:\n{summary}\n"
        
        prompt = f"""Based on these document summaries, answer this question:

{context}

Question: {question}

Answer based only on the information in the documents above."""

        response = st.session_state.llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def chat_general(question: str) -> str:
    """General chat using LLM."""
    if not st.session_state.llm:
        return "⚠️ Please enter your Gemini API key first!"
    
    try:
        response = st.session_state.llm.generate_content(question)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("🤖 Multi-Modal Agent")
    st.markdown("---")
    
    # API Key Input
    st.subheader("🔑 Gemini API Key")
    
    # Try to get from secrets first
    api_key = None
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if api_key:
            st.success("✅ API key loaded from secrets")
            if st.session_state.llm is None:
                initialize_llm(api_key)
    except:
        pass
    
    # Manual input if not in secrets
    if not api_key:
        api_key_input = st.text_input(
            "Enter your API key:",
            type="password",
            help="Get your key from https://makersuite.google.com/app/apikey"
        )
        if api_key_input:
            if initialize_llm(api_key_input):
                st.success("✅ API key accepted!")
            api_key = api_key_input
    
    st.markdown("---")
    
    # File Upload - Data
    st.subheader("📊 Upload Data Files")
    data_files = st.file_uploader(
        "Upload CSV or Excel files",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        key="data_uploader"
    )
    
    if data_files:
        for file in data_files:
            if file.name not in st.session_state.data_files:
                try:
                    if file.name.endswith('.csv'):
                        df = pd.read_csv(file)
                    else:
                        df = pd.read_excel(file)
                    
                    st.session_state.data_files[file.name] = df
                    st.success(f"✅ {file.name}: {len(df)} rows × {len(df.columns)} cols")
                except Exception as e:
                    st.error(f"❌ {file.name}: {str(e)}")
    
    st.markdown("---")
    
    # File Upload - PDFs
    if PDF_AVAILABLE:
        st.subheader("📄 Upload PDF Documents")
        pdf_files = st.file_uploader(
            "Upload PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        if pdf_files:
            for file in pdf_files:
                if file.name not in st.session_state.pdf_files:
                    with st.spinner(f"Processing {file.name}..."):
                        text = extract_text_from_pdf(file)
                        st.session_state.pdf_files[file.name] = text
                        
                        if st.session_state.llm:
                            summary = summarize_pdf(text, st.session_state.llm)
                            st.session_state.pdf_summaries[file.name] = summary
                            st.success(f"✅ {file.name} processed")
                        else:
                            st.warning(f"⚠️ {file.name} uploaded (add API key for summary)")
    
    st.markdown("---")
    
    # Pipeline Control
    st.subheader("🚀 Pipeline Control")
    
    if st.button("🔄 Run Autonomous Pipeline", use_container_width=True):
        if not st.session_state.data_files:
            st.warning("⚠️ Upload data files first!")
        elif not AGENT_SYSTEM_AVAILABLE:
            st.error("⚠️ agent_system.py not found!")
        else:
            with st.spinner("Running pipeline..."):
                for filename, df in st.session_state.data_files.items():
                    try:
                        results = run_autonomous_pipeline(
                            df,
                            target_score=0.8,
                            max_iterations=3,
                            verbose=False
                        )
                        st.session_state.pipeline_results[filename] = results
                        st.session_state.data_files[filename] = results['cleaned_data']
                        st.success(f"✅ {filename} cleaned!")
                    except Exception as e:
                        st.error(f"❌ {filename}: {str(e)}")
    
    # Clear button
    if st.button("🗑️ Clear All Data", use_container_width=True):
        st.session_state.data_files = {}
        st.session_state.pdf_files = {}
        st.session_state.pdf_summaries = {}
        st.session_state.pipeline_results = {}
        st.session_state.messages = []
        st.rerun()

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("🤖 Multi-Modal Data Intelligence Agent")
st.markdown("Upload data files, run autonomous cleaning pipeline, and chat with your data!")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Data View", "📈 Visualizations", "📄 PDF Summaries"])

# ============================================================================
# TAB 1: CHAT
# ============================================================================

with tab1:
    st.subheader("Chat with Your Data")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your data or documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Simple routing based on keywords
                prompt_lower = prompt.lower()
                
                if any(word in prompt_lower for word in ['column', 'row', 'data', 'dataset', 'table', 'mean', 'median']):
                    response = chat_with_data(prompt, st.session_state.data_files)
                elif any(word in prompt_lower for word in ['pdf', 'document', 'report', 'paper']):
                    response = chat_with_pdfs(prompt, st.session_state.pdf_summaries)
                else:
                    response = chat_general(prompt)
                
                st.markdown(response)
        
        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})

# ============================================================================
# TAB 2: DATA VIEW
# ============================================================================

with tab2:
    st.subheader("Dataset Overview")
    
    if st.session_state.data_files:
        for filename, df in st.session_state.data_files.items():
            with st.expander(f"📊 {filename}", expanded=True):
                col1, col2, col3 = st.columns(3)
                col1.metric("Rows", len(df))
                col2.metric("Columns", len(df.columns))
                col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                
                st.dataframe(df, use_container_width=True, height=400)
                
                # Download button
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "💾 Download Cleaned Data",
                    csv,
                    f"cleaned_{filename}",
                    "text/csv",
                    key=f"download_{filename}"
                )
    else:
        st.info("👆 Upload data files using the sidebar")

# ============================================================================
# TAB 3: VISUALIZATIONS
# ============================================================================

with tab3:
    st.subheader("Data Visualizations")
    
    if st.session_state.data_files:
        for filename, df in st.session_state.data_files.items():
            with st.expander(f"📈 {filename}", expanded=True):
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                if len(numeric_cols) >= 2:
                    # Correlation heatmap
                    st.markdown("**Correlation Heatmap**")
                    corr = df[numeric_cols].corr()
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=corr.values,
                        x=corr.columns,
                        y=corr.columns,
                        colorscale='RdBu',
                        zmid=0
                    ))
                    fig.update_layout(height=500, title="Feature Correlations")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Distribution plots
                    st.markdown("**Distribution Plots**")
                    col1, col2 = st.columns(2)
                    
                    for i, col in enumerate(numeric_cols[:4]):  # First 4 numeric columns
                        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                        if i % 2 == 0:
                            col1.plotly_chart(fig, use_container_width=True)
                        else:
                            col2.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough numeric columns for visualizations")
    else:
        st.info("👆 Upload data files using the sidebar")

# ============================================================================
# TAB 4: PDF SUMMARIES
# ============================================================================

with tab4:
    st.subheader("PDF Document Summaries")
    
    if st.session_state.pdf_summaries:
        for filename, summary in st.session_state.pdf_summaries.items():
            with st.expander(f"📄 {filename}", expanded=True):
                st.markdown(summary)
    elif st.session_state.pdf_files:
        st.info("PDF files uploaded. Add API key to generate summaries.")
    else:
        st.info("👆 Upload PDF files using the sidebar")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🤖 Multi-Modal Data Intelligence Agent | Built with Streamlit & Google Gemini</p>
</div>
""", unsafe_allow_html=True)
