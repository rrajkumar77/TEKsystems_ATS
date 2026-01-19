
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

# -------------------- ENV & MODEL --------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found. Set it in your environment or .env")

# Initialize LLM (Groq via LangChain)
# You can switch to: "llama-3.1-70b-versatile", "mixtral-8x7b-32768", "gemma2-27b-it"
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
        d = docx.Document(io.BytesIO(file_bytes))
        text_parts = [p.text for p in d.paragraphs]
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

    ext = uploaded_file.name.split(".")[-1].lower()
    if ext == "pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext == "docx":
        return extract_text_from_docx(file_bytes)
    elif ext == "doc":
        st.warning("DOC has limited support. Please convert to DOCX or PDF for best results.")
        try:
            return extract_text_from_docx(file_bytes)
        except Exception as e:
            st.error(f"Error processing DOC file: {e}")
            return "Error processing DOC file. Please convert to DOCX or PDF for better results."
    elif ext == "txt":
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

# -------------------- RAG INDEX BUILD --------------------
def build_vectorstore(jd_text: str, resume_text: str):
    """
    Builds a single Chroma vectorstore containing both JD and Resume chunks with metadata.
    In-memory for simplicity (ephemeral). Add persist_directory for persistence.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )
    docs = []

    from langchain.schema import Document
    if jd_text:
        jd_docs = splitter.create_documents([jd_text], metadatas=[{"source": "jd"}])
        docs.extend(jd_docs)
    if resume_text:
        resume_docs = splitter.create_documents([resume_text], metadatas=[{"source": "resume"}])
        docs.extend(resume_docs)

    if not docs:
        return None

    embeddings = FastEmbedEmbeddings()  # lightweight local embeddings, no API
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="jobfit_rag",
        # persist_directory="./chroma_jobfit"  # uncomment to persist
    )
    return vs

def make_retriever(vectorstore, scope="both", k=8, search_type="mmr"):
    """
    scope: "jd" | "resume" | "both"
    """
    kwargs = {"k": k}
    if scope in ("jd", "resume"):
        kwargs["filter"] = {"source": scope}
    retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=kwargs)
    return retriever

def retrieve_context(vectorstore, scope: str, query: str, k: int = 8) -> str:
    retriever = make_retriever(vectorstore, scope=scope, k=k)
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([d.page_content for d in docs])

def call_llm_with_context(prompt_template: str, context: str, **fmt_vars) -> str:
    """
    Formats a chat prompt with context + variables and calls the LLM.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    # Build messages for the Chat model
    messages = prompt.format_messages(context=context, **fmt_vars)
    response = llm.invoke(messages)
    return response.content

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

# Controls
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

# Build / cache vectorstore
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "indexed_key" not in st.session_state:
    st.session_state.indexed_key = None

def compute_index_key(jd_text, resume_text):
    return f"{hash(jd_text) if jd_text else 0}-{hash(resume_text) if resume_text else 0}"

if jd_content or resume_content:
    key_now = compute_index_key(jd_content, resume_content)
    if st.session_state.vectorstore is None or st.session_state.indexed_key != key_now:
        with st.spinner("Indexing documents for retrieval..."):
            vs = build_vectorstore(jd_content, resume_content)
            st.session_state.vectorstore = vs
            st.session_state.indexed_key = key_now

def ensure_vs():
    if st.session_state.vectorstore is None:
        st.warning("Please upload a Job Description and/or a Resume first.")
        return None
    return st.session_state.vectorstore

# --------------- Actions ---------------
if submit_recruiter:
    if jd_content and resume_content:
        vs = ensure_vs()
        if vs:
            with st.spinner("Analyzing alignment..."):
                # Retrieve from both JD & resume to ground the comparison
                ctx_jd = retrieve_context(vs, "jd", "role requirements, key responsibilities, skills, experience", k=8)
                ctx_cv = retrieve_context(vs, "resume", "candidate skills, projects, responsibilities, experience", k=8)
                context = ctx_jd + "\n\n---\n\n" + ctx_cv
                answer = call_llm_with_context(PROMPT_RECRUITER, context)
            st.subheader("Technical Recruiter Analysis")
            st.write(answer)
    else:
        st.info("Please upload both a Job Description and a Resume to proceed.")

elif submit_technical_questions:
    if jd_content and resume_content:
        vs = ensure_vs()
        if vs:
            with st.spinner("Generating technical questions..."):
                ctx_jd = retrieve_context(vs, "jd", "technical stack, tools, methodologies, domain", k=8)
                ctx_cv = retrieve_context(vs, "resume", "skills, tools, technologies, project details", k=8)
                context = ctx_jd + "\n\n---\n\n" + ctx_cv
                answer = call_llm_with_context(PROMPT_TECHNICAL_Q, context)
            st.subheader("Technical Questions")
            st.write(answer)
    else:
        st.info("Please upload both a Job Description and a Resume to proceed.")

elif submit_coding_questions:
    if jd_content and resume_content:
        vs = ensure_vs()
        if vs:
            with st.spinner("Generating coding questions..."):
                ctx_jd = retrieve_context(vs, "jd", "coding tasks, programming languages, data processing, testing", k=8)
