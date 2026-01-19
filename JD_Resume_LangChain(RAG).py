import streamlit as st
import os
import io
import fitz  # PyMuPDF
import docx  # python-docx
from dotenv import load_dotenv

# LangChain / RAG imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema import Document

# -------------------- ENV & MODEL --------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found. Set it in your environment or .env.")

# Initialize LLM (Groq via LangChain)
# You can switch models: "llama-3.1-70b-versatile", "mixtral-8x7b-32768", etc.
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=3000,
)

# -------------------- FILE HELPERS --------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
        text_parts = [page.get_text() for page in document]
        return " ".join(text_parts)
    except Exception as e:
        raise ValueError(f"Failed to open PDF: {e}")

def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        text_parts = [paragraph.text for paragraph in doc.paragraphs]
        return " ".join(text_parts)
    except Exception as e:
        raise ValueError(f"Failed to open DOCX: {e}")

def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8")
    except Exception as e:
        raise ValueError(f"Failed to decode TXT file: {e}")

def process_file(uploaded_file) -> str:
    if uploaded_file is None:
        raise FileNotFoundError("No file uploaded")

    file_bytes = uploaded_file.read()
    if not file_bytes:
        raise ValueError("Uploaded file is empty or unreadable.")

    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif file_extension == "docx":
        return extract_text_from_docx(file_bytes)
    elif file_extension == "doc":
        st.warning("DOC has limited support. Please convert to DOCX or PDF for best results.")
        try:
            return extract_text_from_docx(file_bytes)
        except Exception as e:
            st.error(f"Error processing DOC file: {e}")
            return "Error processing DOC file. Please convert to DOCX or PDF for better results."
    elif file_extension == "txt":
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

# -------------------- RAG INDEX BUILD --------------------
def build_vectorstore(jd_text: str, resume_text: str):
    """
    Builds a single Chroma vectorstore containing both JD and Resume chunks with metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    docs = []

    if jd_text:
        jd_docs = splitter.create_documents([jd_text], metadatas=[{"source": "jd"}])
        docs.extend(jd_docs)
    if resume_text:
        resume_docs = splitter.create_documents([resume_text], metadatas=[{"source": "resume"}])
        docs.extend(resume_docs)

    if not docs:
        return None

    embeddings = FastEmbedEmbeddings()  # lightweight local embeddings
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="jobfit_rag",
        # No persist_directory -> in-memory (ephemeral) store
    )
    return vs

# -------------------- PROMPTS --------------------
PROMPT_RECRUITER = """\
You are an Experienced Technical HR Manager with deep expertise in technical evaluations and recruitment.
Use the provided context from the Job Description and Resume to assess alignment.

Instructions:
- Provide:
  1) Match percentage between the resume and JD (e.g., "80% match").
  2) A professional evaluation (1–2 paragraphs) highlighting strengths and weaknesses.
  3) Suggestions for improvement (e.g., missing skills, certifications to consider).
- Be concise, professional, and grounded strictly in the context.

<context>
{context}
</context>

Task: Perform the full Recruiter Analysis.
"""

PROMPT_TECHNICAL_Q = """\
You are an Advanced AI for Technical Recruitment. Use the JD and Resume context to generate up to 5 questions each under these categories:
- Behavioural Question
- Skill based Technical Question
- Situational Question
- Problem Solving Question

Rules:
- Questions should be tailored to skills/tools noted in the JD and Resume.
- Include a model answer for each question to help recruiters validate responses.
- Balance difficulty (basic, intermediate, advanced).
- Keep generic (no company-specific references).

<context>
{context}
</context>

Task: Generate the categorized questions with answers in a clean bullet/numbered format.
"""

PROMPT_CODING_Q = """\
You are an Advanced AI for Technical Recruitment. Based on the JD and Resume context, generate up to 5 coding questions ordered by project lifecycle (requirements, design, development, testing, deployment).

For each:
- Category: "skill based Coding Question"
- Question: Specific coding task relevant to the JD/Resume stack
- Answer: Provide a reference solution with code and a brief explanation

<context>
{context}
</context>

Task: Generate the coding questions with detailed solutions.
"""

PROMPT_DOMAIN = """\
You are an ATS Scanner with Domain Expertise. Evaluate the Resume against the JD from a domain perspective.

Required Output:
- Match percentage (e.g., "75%") with a brief explanation.
- Missing keywords (comma-separated).
- Objective, thorough evaluation grounded in the context.

<context>
{context}
</context>

Task: Provide the domain-fit analysis.
"""

PROMPT_MANAGER = """\
You are an ATS Scanner with Technical Expertise. Evaluate the Resume against the JD for technical fit.

Required Output:
1) Match percentage
2) Explanation of matches and gaps
3) Missing keywords/skills
4) A table (plain text) of top 5 skills with: Required years (JD), Candidate years (Resume), Relevant projects
5) Final suitability summary

<context>
{context}
</context>

Task: Provide the technical-fit analysis with the requested table and insights.
"""

PROMPT_JD_SUMMARY = """\
You are an AI Assistant. Summarize the JD and provide recruiter recommendations.

Output two sections:
1) JD Summary (3–5 sentences of responsibilities, key skills, and qualifications)
2) Recommendations: Suggest skill combos, keywords, and sourcing strategy

Use only the JD context below.

<context>
{context}
</context>

Task: Provide the summary and recommendations.
"""

PROMPT_JD_CLARIFICATION = """\
You are a Technical Recruitment Consultant. Using ONLY the JD context, generate 5–10 precise questions for the hiring manager to clarify technical requirements, tools, expectations, project scope, and expertise levels.

- Avoid generic questions; tailor to the JD content.
- Output as a clean, numbered list.

<context>
{context}
</context>

Task: Provide the JD clarification questions.
"""

PROMPT_SKILL_ANALYST = """\
You are a Skill Analyst. Using ONLY the Resume context, analyze the provided top_skills list.

For each skill:
- Match Status: Yes/No (explicit or implicit)
- Relevant Projects: roles/projects/experiences from the resume (or "None")
- Years of Experience: best estimate from resume context; if unclear, make a reasonable assumption (e.g., "1 year" junior, "3 years" mid-level)

Output a structured table (plain text) with columns:
Skill | Match Status | Relevant Projects | Years of Experience

<context>
{context}
</context>

Top skills to analyze:
{top_skills}

Task: Provide the table only, followed by a brief note on assumptions if any.
"""

PROMPT_GENERAL_Q = """\
You are an AI Assistant for Recruitment Queries. Use the available context (JD, Resume, or both) to answer the user’s question clearly with useful insights.

<context>
{context}
</context>

User question:
{user_query}

Task: Provide a clear, concise, context-aware answer. If the question is unclear, ask one clarifying question.
"""

# -------------------- RAG CHAIN HELPERS --------------------
def make_retriever(vectorstore, scope="both", k=6):
    """
    scope: "jd" | "resume" | "both"
    """
    if scope == "jd":
        return vectorstore.as_retriever(search_kwargs={"k": k, "filter": {"source": "jd"}})
    elif scope == "resume":
        return vectorstore.as_retriever(search_kwargs={"k": k, "filter": {"source": "resume"}})
    else:
        return vectorstore.as_retriever(search_kwargs={"k": k})

def run_rag_task(retriever, prompt_template: str, **fmt_vars):
    """
    Creates a retrieval-augmented chain with a stuff-combine strategy.
    fmt_vars should include keys used in the prompt (e.g., user_query, top_skills).
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Chain that stuffs retrieved docs into {context}
    doc_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_variable_name="context",
    )
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)

    # Build inference input
    input_vars = {}
    input_vars.update(fmt_vars)  # e.g., user_query=..., top_skills=...

    result = retrieval_chain.invoke(input_vars)
    return result["answer"]

# -------------------- STREAMLIT UI --------------------
st.set_page_config(page_title="Resume Expert (RAG + LangChain)")
st.header("TEKsystems JobFit Analyzer — RAG Edition (LangChain + Groq)")
st.subheader("Understand the JD and evaluate the Resume with grounded retrieval")

uploaded_jd = st.file_uploader(
    "Upload the Job Description (PDF, DOCX, DOC, TXT)...",
    type=["pdf", "docx", "doc", "txt"],
    key="jd_uploader"
)
submit_jd_summarization = st.button("JD Summarization", key="submit_jd_summarization")
submit_jd_clarification = st.button("JD Clarification Questions", key="submit_jd_clarification")

uploaded_resume = st.file_uploader(
    "Upload your Resume (PDF, DOCX, DOC, TXT)...",
    type=["pdf", "docx", "doc", "txt"],
    key="resume_uploader"
)

jd_content = ""
resume_content = ""

if uploaded_jd is not None:
    file_type = uploaded_jd.name.split(".")[-1].upper()
    st.write(f"{file_type} Job Description Uploaded Successfully")
    jd_content = process_file(uploaded_jd)

if uploaded_resume is not None:
    file_type = uploaded_resume.name.split(".")[-1].upper()
    st.write(f"{file_type} Resume Uploaded Successfully")
    resume_content = process_file(uploaded_resume)

# Buttons
col1, col2, col3 = st.columns(3)
with col1:
    submit_recruiter = st.button("Technical Recruiter Analysis", key="submit_recruiter")
    submit_domain = st.button("Domain Expert Analysis", key="submit_domain")
with col2:
    submit_technical_questions = st.button("Technical Questions", key="submit_technical_questions")
    submit_manager = st.button("Technical Manager Analysis", key="submit_manager")
with col3:
    submit_coding_questions = st.button("Coding Questions", key="submit_coding_questions")

top_skills = st.text_input("Top Skills Required for the Job (comma-separated):", key="top_skills_input")
submit_skill_analysis = st.button("Skill Analysis", key="submit_skill_analysis")

input_promp = st.text_input("Queries: Feel Free to Ask here", key="custom_query_input")
submit_general_query = st.button("Answer My Query", key="submit_general_query")

# Build vector store once and cache in session
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if (jd_content or resume_content) and st.session_state.vectorstore is None:
    with st.spinner("Indexing documents for retrieval..."):
        vs = build_vectorstore(jd_content, resume_content)
        st.session_state.vectorstore = vs

# --------------- Actions ---------------
def ensure_vs():
    if st.session_state.vectorstore is None:
        st.warning("Please upload a Job Description and/or a Resume first.")
        return None
    return st.session_state.vectorstore

if submit_recruiter:
    if jd_content and resume_content:
        vs = ensure_vs()
        if vs:
            retriever = make_retriever(vs, scope="both", k=8)
            with st.spinner("Analyzing alignment..."):
                answer = run_rag_task(retriever, PROMPT_RECRUITER)
            st.subheader("Technical Recruiter Analysis")
            st.write(answer)
    else:
        st.info("Please upload both a Job Description and a Resume to proceed.")

elif submit_technical_questions:
    if jd_content and resume_content:
        vs = ensure_vs()
        if vs:
            retriever = make_retriever(vs, scope="both", k=8)
            with st.spinner("Generating technical questions..."):
                answer = run_rag_task(retriever, PROMPT_TECHNICAL_Q)
            st.subheader("Technical Questions")
            st.write(answer)
    else:
        st.info("Please upload both a Job Description and a Resume to proceed.")

elif submit_coding_questions:
    if jd_content and resume_content:
        vs = ensure_vs()
        if vs:
            retriever = make_retriever(vs, scope="both", k=8)
            with st.spinner("Generating coding questions..."):
                answer = run_rag_task(retriever, PROMPT_CODING_Q)
            st.subheader("Coding Questions")
            st.write(answer)
    else:
        st.info("Please upload both a Job Description and a Resume to proceed.")

elif submit_domain:
    if jd_content and resume_content:
        vs = ensure_vs()
        if vs:
            retriever = make_retriever(vs, scope="both", k=8)
            with st.spinner("Running domain-fit analysis..."):
                answer = run_rag_task(retriever, PROMPT_DOMAIN)
            st.subheader("Domain Expert Analysis")
            st.write(answer)
    else:
        st.info("Please upload both a Job Description and a Resume to proceed.")

elif submit_manager:
    if jd_content and resume_content:
        vs = ensure_vs()
        if vs:
            retriever = make_retriever(vs, scope="both", k=8)
            with st.spinner("Running technical-fit analysis..."):
                answer = run_rag_task(retriever, PROMPT_MANAGER)
            st.subheader("Technical Manager Analysis")
            st.write(answer)
    else:
        st.info("Please upload both a Job Description and a Resume to proceed.")

elif submit_jd_summarization:
    if jd_content:
        vs = ensure_vs()
        if vs:
            retriever = make_retriever(vs, scope="jd", k=8)
            with st.spinner("Summarizing JD..."):
                answer = run_rag_task(retriever, PROMPT_JD_SUMMARY)
            st.subheader("Job Description Summary")
            st.write(answer)
    else:
        st.info("Please upload a Job Description to proceed.")

elif submit_jd_clarification:
    if jd_content:
        vs = ensure_vs()
        if vs:
            retriever = make_retriever(vs, scope="jd", k=8)
            with st.spinner("Drafting clarification questions..."):
                answer = run_rag_task(retriever, PROMPT_JD_CLARIFICATION)
            st.subheader("JD Clarification Questions")
            st.write(answer)
    else:
        st.info("Please upload a Job Description to proceed.")

elif submit_skill_analysis:
    if uploaded_resume is not None and top_skills:
        vs = ensure_vs()
        if vs:
            retriever = make_retriever(vs, scope="resume", k=8)
            with st.spinner("Analyzing top skills in the resume..."):
                answer = run_rag_task(retriever, PROMPT_SKILL_ANALYST, top_skills=top_skills)
            st.subheader("Top Skill Analysis")
            st.write(answer)
    else:
        st.info("Please upload a Resume and enter Top Skills to proceed.")

elif submit_general_query:
    if jd_content or resume_content:
        vs = ensure_vs()
        if vs:
            retriever = make_retriever(vs, scope="both", k=8)
            with st.spinner("Answering your query..."):
                answer = run_rag_task(retriever, PROMPT_GENERAL_Q, user_query=input_promp or "Provide insights based on the context.")
            st.subheader("Query Response")
            st.write(answer)
    else:
        st.info("Please upload a Resume or a Job Description to proceed.")
