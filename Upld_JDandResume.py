import streamlit as st
import google.generativeai as genai
import os
import fitz # PyMuPDF for PDF processing
import docx # python-docx for DOCX processing
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input_jd, resume_content, prompt, additional_input=""):
    model = genai.GenerativeModel('gemini-1.5-flash')
    if additional_input:
        response = model.generate_content([input_jd, resume_content, prompt, additional_input])
    else:
        response = model.generate_content([input_jd, resume_content, prompt])
    return response.text

def extract_text_from_pdf(file_bytes):
    """Extract text from PDF file"""
    document = fitz.open(stream=file_bytes, filetype="pdf")
    text_parts = []
    for page in document:
        text_parts.append(page.get_text())
    return " ".join(text_parts)

def extract_text_from_docx(file_bytes):
    """Extract text from DOCX file"""
    doc = docx.Document(io.BytesIO(file_bytes))
    text_parts = []
    for paragraph in doc.paragraphs:
        text_parts.append(paragraph.text)
    return " ".join(text_parts)

def extract_text_from_txt(file_bytes):
    """Extract text from TXT file"""
    return file_bytes.decode('utf-8')

def process_file(uploaded_file):
    """Process different file formats and extract text"""
    if uploaded_file is None:
        raise FileNotFoundError("No file uploaded")
    
    file_bytes = uploaded_file.read()
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        return extract_text_from_pdf(file_bytes)
    elif file_extension == 'docx':
        return extract_text_from_docx(file_bytes)
    elif file_extension == 'doc':
        st.warning("DOC format has limited support. For best results, consider converting to DOCX or PDF.")
        try:
            return extract_text_from_docx(file_bytes)
        except Exception as e:
            st.error(f"Error processing DOC file: {e}")
            return "Error processing DOC file. Please convert to DOCX or PDF for better results."
    elif file_extension == 'txt':
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

# Define all prompts
input_prompt1 = """
Role: Experienced Technical Human Resource Manager with expertise in technical evaluations and Recruitment
Task: Review the provided resume against the job description.
Objective: Evaluate whether the candidate's profile aligns with the job description.
Instructions:
1. Input: A job description (JD), and a resume extracted from a PDF.
2. Process: Compare the resume to the JD to assess alignment.
3. Output: Provide:
 - Match percentage between the resume and JD (e.g., "80% match").
 - A professional evaluation (1-2 paragraphs) highlighting strengths (e.g., relevant skills like Python, SQL) and weaknesses (e.g., missing Machine Learning experience).
 - Suggestions for improvement (e.g., "Candidate should highlight Machine Learning skills or gain certification").
 Ensure the response is concise and professional.
"""

input_prompt_technical = """
Role: Advanced AI for Technical Recruitment
Task: Generate technical questions based on the provided job description and resume.
Objective: Create technical questions tailored to the JD and resume, sequenced from project start to finish.
Instructions:
1. Input: A job description (JD) detailing the role's requirements, skills, and responsibilities, and a resume extracted from a PDF outlining the candidate's skills, experience, and projects.
2. Process: Analyze the JD and resume to generate up to 5 technical questions in project lifecycle order (requirements gathering, design, development, testing, deployment). Base the questions on the skills, tools, and experiences mentioned in the JD and resume.
3. Output: For each question, provide:
 - Category: "Technical Question".
 - Question: A clear, specific question relevant to the JD and resume (e.g., "How do you approach requirements gathering for a data-driven project?").
 - Answer: A detailed answer for recruiters to validate responses, including key concepts, tools, or techniques the candidate should mention.
Additional Notes:
- Ensure questions are tailored to the specific skills, tools, and experiences in the JD and resume (e.g., if Python and SQL are mentioned, include relevant questions).
- Maintain a balance of difficulty (basic, intermediate, advanced) to assess the candidate comprehensively.
- Avoid hardcoding references to a specific role or company; keep the prompt generic and adaptable.
"""

input_prompt_coding = """
Role: Advanced AI for Technical Recruitment
Task: Generate coding questions based on the provided job description and resume.
Objective: Create coding questions tailored to the JD and resume, sequenced from project start to finish.
Instructions:
1. Input: A job description (JD) detailing the role's requirements, skills, and responsibilities, and a resume extracted from a PDF outlining the candidate's skills, experience, and projects.
2. Process: Analyze the JD and resume to generate up to 5 coding questions in project lifecycle order (requirements gathering, design, development, testing, deployment). Base the questions on the skills, tools, and experiences mentioned in the JD and resume.
3. Output: For each question, provide:
 - Category: "Coding Question".
 - Question: A clear, specific coding problem relevant to the JD and resume (e.g., "Write a Python function to aggregate sales data by region from a CSV file.").
 - Answer: A detailed solution with code, explanation, and key concepts the candidate should demonstrate.
Additional Notes:
- Ensure questions are tailored to the specific skills, tools, and experiences in the JD and resume (e.g., if Python and SQL are mentioned, include relevant coding problems).
- Maintain a balance of difficulty (basic, intermediate, advanced) to assess the candidate comprehensively.
- Avoid hardcoding references to a specific role or company; keep the prompt generic and adaptable.
"""

input_prompt3 = """
Role: ATS Scanner with Domain Expertise
Task: Evaluate resume against JD for domain fit (e.g., Business Analytics).
Objective: Assess compatibility from a domain perspective.
Instructions:
1. Input: JD and resume.
2. Output: 
 - Match percentage (e.g., "75%") with explanation.
 - Missing keywords (e.g., "Tableau, PowerBI").
 - Ensure evaluation is thorough and objective.
"""

input_prompt4 = """
Role: ATS Scanner with Technical Expertise
Task: Evaluate resume against JD for technical fit.
Objective: Assess compatibility for the JD and resume.
Instructions:
1. Calculate match percentage.
2. Explain match and gaps.
3. Identify missing keywords/skills.
4. Create a table of top 5 skills with required years (JD), candidate years (resume), and relevant projects.
5. Share final thoughts on suitability.
"""

input_prompt5 = """
Role: AI Assistant
Task: Summarize the JD and provide recruiter recommendations.
Objective: Summarize key details and suggest sourcing strategies.
Instructions:
1. Output two sections:
 - JD Summary: Concise summary (3-5 sentences) of responsibilities, skills (e.g., PySpark, Python), and qualifications.
 - Recommendations: Suggest skill combinations (e.g., "Data Analyst + Statistical Modeler"), keywords (e.g., "SQL, Machine Learning"), and sourcing Strategy.
2. Keep responses professional and actionable.
"""

input_prompt6 = """
Role: Skill Analyst
Task: Perform a Skill Analysis
Objective: Analyze the provided resume to determine the match status of user-specified skills.
Instructions:
1. Input: You will receive two pieces of input:
 - A list of top skills provided as a comma-separated string (e.g., "SQL, Python, Pyspark") passed as additional input.
 - A resume extracted from a file, provided as text content, detailing the candidate's skills, experience, projects, and qualifications.
2. Process: For each skill in the provided top_skills list:
 - Check if the skill is explicitly mentioned or implied in the resume (e.g., through job titles, tools used, projects, certifications, or keywords).
 - Estimate the years of experience for each skill by analyzing the duration of relevant roles, projects, or education in the resume. If no specific duration is provided, estimate based on context (e.g., "recent graduate" = 0-1 year, "senior role" = 3+ years).
 - Identify any relevant projects, roles, or experiences from the resume that demonstrate the skill.
3. Output: Present the results in a clear, structured table format with the following columns:
 - Skill: The specific skill from the top_skills list (use exact wording from the input).
 - Match Status: "Yes" if the skill is present in the resume (explicitly or implicitly), otherwise "No".
 - Relevant Projects: List projects, roles, or experiences from the resume that demonstrate the skill (e.g., "Data Analysis Project"). If none, write "None".
 - Years of Experience: Estimate total years related to the skill (e.g., "2 years"). If no experience, write "0 years".
4. Additional Notes:
 - Use only the skills provided in the top_skills input; do not prompt for additional input or reference the job description (JD).
 - Handle resume text noise (e.g., file extraction artifacts) by focusing on key terms and context.
 - If experience duration is unclear, make reasonable assumptions (e.g., "1 year" for junior roles, "3 years" for mid-level).
 - Ensure output is concise, professional, and suitable for Streamlit display.
"""

input_prompt_query = """
Role: AI Assistant for Recruitment Queries
Task: Answer user queries about JD or resume.
Objective: Provide detailed, context-aware responses.
Instructions:
1. Input: JD, resume, and user query (e.g., "What skills are missing?").
2. Output: Clear response summarizing or comparing JD/resume,

Objective: Provide detailed, context-aware responses.
Instructions:
1. Input: JD, resume, and user query (e.g., "What skills are missing?").
2. Output: Clear response summarizing or comparing JD/resume, with insights (e.g., "Resume lacks Machine Learning; suggest certification").
3. If query is unclear, ask for clarification.
"""

input_prompt_jd_clarification = """
Role: Technical Recruitment Consultant
Task: Generate a list of technical questions to ask the hiring manager to clarify the job description.
Objective: Help the recruiter understand the technical requirements, tools, and expectations of the role to source the right candidate.
Instructions:
1. Input: A job description (JD) detailing the role's requirements, skills, and responsibilities.
2. Process: Analyze the JD to identify ambiguous, technical, or critical aspects that need clarification (e.g., specific tools, expertise levels, project scope). Generate 5–10 technical questions that a recruiter can ask the hiring manager to gain deeper insights into the role’s requirements.
3. Output: 
 - A numbered list of 5–10 questions, each:
 - Focused on technical skills, tools, role responsibilities, or project context.
 - Specific to the JD’s content (e.g., if "cloud computing" is mentioned, ask about preferred platforms like AWS or Azure).
 - Designed to elicit detailed responses (e.g., "What specific cloud platforms should the candidate have experience with, and what types of tasks will they perform?").
 - Ensure questions cover a range of areas (e.g., tools, expertise level, project scope, team dynamics).
4. Additional Notes:
 - Do not reference or require a resume; focus solely on the JD.
 - Avoid generic questions; tailor questions to the JD’s specific skills, tools, or responsibilities.
 - Ensure questions are professional, clear, and suitable for a recruiter to ask a hiring manager.
 - Maintain a balance of question types (e.g., about skills, projects, and expectations).
 - Format the output as a clean, numbered list for display in a Streamlit app.
"""

# Streamlit App
st.set_page_config(page_title="Resume Expert")
st.header("TEKsystems JobFit Analyzer")
st.subheader('This Application helps you to understand the Job Description and evaluate the Resume')

uploaded_jd = st.file_uploader("Upload the Job Description (PDF, DOCX, DOC, TXT)...", type=["pdf", "docx", "doc", "txt"], key="jd_uploader")
submit_jd_summarization = st.button("JD Summarization", key="submit_jd_summarization")
submit_jd_clarification = st.button("JD Clarification Questions", key="submit_jd_clarification")
uploaded_resume = st.file_uploader("Upload your Resume (PDF, DOCX, DOC, TXT)...", type=["pdf", "docx", "doc", "txt"], key="resume_uploader")

jd_content = ""
resume_content = ""

if uploaded_jd is not None:
    file_type = uploaded_jd.name.split('.')[-1].upper()
    st.write(f"{file_type} Job Description Uploaded Successfully")
    jd_content = process_file(uploaded_jd)

if uploaded_resume is not None:
    file_type = uploaded_resume.name.split('.')[-1].upper()
    st.write(f"{file_type} Resume Uploaded Successfully")
    resume_content = process_file(uploaded_resume)

submit_recruiter = st.button("Technical Recruiter Analysis", key="submit_recruiter")
submit_technical_questions = st.button("Technical Questions", key="submit_technical_questions")
submit_coding_questions = st.button("Coding Questions", key="submit_coding_questions")
submit_domain = st.button("Domain Expert Analysis", key="submit_domain")
submit_manager = st.button("Technical Manager Analysis", key="submit_manager")
top_skills = st.text_input("Top Skills Required for the Job (comma-separated):", key="top_skills_input")
submit_skill_analysis = st.button("Skill Analysis", key="submit_skill_analysis")
input_promp = st.text_input("Queries: Feel Free to Ask here", key="custom_query_input")
submit_general_query = st.button("Answer My Query", key="submit_general_query")

if submit_recruiter:
    if jd_content and resume_content:
        try:
            response = get_gemini_response(jd_content, resume_content, input_prompt1)
            st.subheader("Technical Recruiter Analysis")
            st.write(response)
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.write("Please upload both a job description and a resume to proceed.")
elif submit_technical_questions:
    if jd_content and resume_content:
        try:
            response = get_gemini_response(jd_content, resume_content, input_prompt_technical)
            st.subheader("Technical Questions")
            st.write(response)
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.write("Please upload both a job description and a resume to proceed.")
elif submit_coding_questions:
    if jd_content and resume_content:
        try:
            response = get_gemini_response(jd_content, resume_content, input_prompt_coding)
            st.subheader("Coding Questions")
            st.write(response)
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.write("Please upload both a job description and a resume to proceed.")
elif submit_domain:
    if jd_content and resume_content:
        try:
            response = get_gemini_response(jd_content, resume_content, input_prompt3)
            st.subheader("Domain Expert Analysis")
            st.write(response)
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.write("Please upload both a job description and a resume to proceed.")
elif submit_manager:
    if jd_content and resume_content:
        try:
            response = get_gemini_response(jd_content, resume_content, input_prompt4)
            st.subheader("Technical Manager Analysis")
            st.write(response)
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.write("Please upload both a job description and a resume to proceed.")
elif submit_jd_summarization:
    if jd_content:
        try:
            response = get_gemini_response(jd_content, "", input_prompt5)
            st.subheader("Job Description Summary")
            st.write(response)
        except Exception as e:
            st.error(f"Error processing request: {e}")
    else:
        st.write("Please upload a job description to proceed.")
elif submit_jd_clarification:
    if jd_content:
        try:
            response = get_gemini_response(jd_content, "", input_prompt_jd_clarification)
            st.subheader("JD Clarification Questions")
            st.write(response)
        except Exception as e:
            st.error(f"Error processing request: {e}")
    else:
        st.write("Please upload a job description to proceed.")
elif submit_skill_analysis:
    if uploaded_resume is not None and top_skills:
        try:
            resume_content = process_file(uploaded_resume)
            response = get_gemini_response("", resume_content, input_prompt6, top_skills)
            st.subheader("Top Skill Analysis")
            st.write(response)
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.write("Please upload a resume file and enter Top Skills to proceed.")
elif submit_general_query:
    if jd_content or resume_content:
        try:
            resume_content = process_file(uploaded_resume) if uploaded_resume is not None else ""
            response = get_gemini_response(jd_content, resume_content, input_prompt_query, input_promp)
            st.subheader("Query Response")
            st.write(response)
        except Exception as e:
            if "No file uploaded" not in str(e):
                st.error(f"Error processing file: {e}")
            else:
                st.write("Please upload a resume file or enter a Job Description to proceed.")
    else:
        st.write("Please upload a resume file or enter a Job Description to proceed.")
