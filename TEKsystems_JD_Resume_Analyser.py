import streamlit as st
import google.generativeai as genai
import os
import fitz
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input_jd, pdf_content, prompt, additional_input=""):
    model = genai.GenerativeModel('gemini-1.5-flash')
    if additional_input:
        response = model.generate_content([input_jd, pdf_content, prompt, additional_input])
    else:
        response = model.generate_content([input_jd, pdf_content, prompt])
    return response.text

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text_parts = []
        for page in document:
            text_parts.append(page.get_text())
        pdf_text_content = " ".join(text_parts)
        return pdf_text_content
    else:
        raise FileNotFoundError("No file uploaded")

# Define all prompts at the top to ensure they are in scope
input_prompt1 = """
Role: Experienced Technical Human Resource Manager with expertise in technical evaluations and Recruitment
Task: Review the provided resume against the job description.
Objective: Evaluate whether the candidate's profile aligns with the "Brand Analytics – Data Analyst" role.
Instructions:
1. Input: A job description (JD) for a Brand Analytics – Data Analyst role in Chennai, India (3-7 years experience), and a resume extracted from a PDF.
2. Process: Compare the resume to the JD to assess alignment.
3. Output: Provide:
   - Match percentage between the resume and JD (e.g., "80% match").
   - A professional evaluation (1-2 paragraphs) highlighting strengths (e.g., relevant skills like Python, SQL) and weaknesses (e.g., missing Machine Learning experience).
   - Suggestions for improvement (e.g., "Candidate should highlight Machine Learning skills or gain certification").
   Ensure the response is concise and professional.
"""

input_prompt2 = """
Role: Advanced AI for Technical Recruitment
Task: Generate evaluation questions based on the provided job description and resume.
Objective: Create technical and coding questions tailored to the JD and resume, sequenced from project start to finish.
Instructions:
1. Input: A job description (JD) detailing the role’s requirements, skills, and responsibilities, and a resume extracted from a PDF outlining the candidate’s skills, experience, and projects.
2. Process: Analyze the JD and resume to generate up to 10 questions (5 Technical, 5 Coding) in project lifecycle order (requirements gathering, design, development, testing, deployment). Base the questions on the skills, tools, and experiences mentioned in the JD and resume.
3. Output: For each question, provide:
   - Category: "Technical Question" or "Coding Question".
   - Question: A clear, specific question relevant to the JD and resume (e.g., "How do you approach requirements gathering for a data-driven project?").
   - Answer: A detailed answer for recruiters to validate responses, including key concepts, tools, or techniques the candidate should mention.
Example:
Technical Question: "What is the difference between predictive and prescriptive analytics?"
Answer: "Predictive analytics forecasts future outcomes using historical data and statistical models like regression, while prescriptive analytics recommends actions based on those predictions, often using optimization techniques. Look for understanding of tools like Python, R, and business implications."
Additional Notes:
- Ensure questions are tailored to the specific skills, tools, and experiences in the JD and resume (e.g., if Python and SQL are mentioned, include relevant questions).
- Maintain a balance of difficulty (basic, intermediate, advanced) to assess the candidate comprehensively.
- Avoid hardcoding references to a specific role or company; keep the prompt generic and adaptable.
"""

input_prompt3 = """
Role: ATS Scanner with Domain Expertise
Task: Evaluate resume against JD for domain fit (e.g., Business Analytics).
Objective: Assess compatibility from a domain perspective for the "Brand Analytics – Data Analyst" role.
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
Objective: Assess compatibility for the "Brand Analytics – Data Analyst" role.
Instructions:
1. Calculate match percentage.
2. Explain match and gaps.
3. Identify missing keywords/skills.
4. Create a table of top 5 skills with required years (JD), candidate years (resume), and relevant projects.
5. Share final thoughts on suitability.
"""

input_prompt5 = """
Role: AI Assistant
Task: Summarize the "Brand Analytics – Data Analyst" JD and provide recruiter recommendations.
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
   - A resume extracted from a PDF file, provided as text content, detailing the candidate’s skills, experience, projects, and qualifications.

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
   - Handle resume text noise (e.g., PDF extraction artifacts) by focusing on key terms and context.
   - If experience duration is unclear, make reasonable assumptions (e.g., "1 year" for junior roles, "3 years" for mid-level).
   - Ensure output is concise, professional, and suitable for Streamlit display.

Example:
If top_skills are "SQL, Python, Pyspark" and the resume mentions "3 years of Python in Data Science Project" and "1 year of SQL in Database Project," but no Pyspark, the output table should be:

| Skill   | Match Status | Relevant Projects         | Years of Experience |
|---------|--------------|---------------------------|---------------------|
| SQL     | Yes          | Database Project          | 1 year             |
| Python  | Yes          | Data Science Project      | 3 years            |
| Pyspark | No           | None                      | 0 years            |

Now, analyze the provided top_skills and resume content to generate the skill analysis table.
"""

input_prompt_query = """
Role: AI Assistant for Recruitment Queries
Task: Answer user queries about JD or resume.
Objective: Provide detailed, context-aware responses.
Instructions:
1. Input: JD, resume, and user query (e.g., "What skills are missing?").
2. Output: Clear response summarizing or comparing JD/resume, with insights (e.g., "Resume lacks Machine Learning; suggest certification").
3. If query is unclear, ask for clarification.
"""

# Streamlit App
st.set_page_config(page_title="Resume Expert")

st.header("TEKsystems JobFit Analyzer")
st.subheader('This Application helps you to understand the Job Description and evaluate the Resume')

input_text = st.text_input("Job Description: ", key="input_jd")
submit_jd_summarization = st.button("JD Summarization", key="submit_jd_summarization")

uploaded_file = st.file_uploader("Upload your Resume(PDF)...", type=["pdf"], key="pdf_uploader")
pdf_content = ""
if uploaded_file is not None:
    st.write("PDF Uploaded Successfully")

submit_recruiter = st.button("Technical Recruiter Analysis", key="submit_recruiter")
submit_questions = st.button("Technical Questions", key="submit_questions")
submit_domain = st.button("Domain Expert Analysis", key="submit_domain")
submit_manager = st.button("Technical Manager Analysis", key="submit_manager")

top_skills = st.text_input("Top Skills Required for the Job (comma-separated):", key="top_skills_input")
submit_skill_analysis = st.button("Skill Analysis", key="submit_skill_analysis")

input_promp = st.text_input("Queries: Feel Free to Ask here", key="custom_query_input")
submit_general_query = st.button("Answer My Query", key="submit_general_query")

if submit_recruiter:
    if uploaded_file is not None and input_text:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_text, pdf_content, input_prompt1)
        st.subheader("Technical Recruiter Analysis")
        st.write(response)
    else:
        st.write("Please upload a PDF and enter a Job Description to proceed.")

elif submit_questions:
    if uploaded_file is not None and input_text:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_text, pdf_content, input_prompt2)
        st.subheader("Technical Questions")
        st.write(response)
    else:
        st.write("Please upload a PDF and enter a Job Description to proceed.")

elif submit_domain:
    if uploaded_file is not None and input_text:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_text, pdf_content, input_prompt3)
        st.subheader("Domain Expert Analysis")
        st.write(response)
    else:
        st.write("Please upload a PDF and enter a Job Description to proceed.")

elif submit_manager:
    if uploaded_file is not None and input_text:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_text, pdf_content, input_prompt4)
        st.subheader("Technical Manager Analysis")
        st.write(response)
    else:
        st.write("Please upload a PDF and enter a Job Description to proceed.")

elif submit_general_query:
    if uploaded_file is not None or input_text:
        pdf_content = input_pdf_setup(uploaded_file) if uploaded_file is not None else ""
        response = get_gemini_response(input_text, pdf_content, input_prompt_query, input_promp)
        st.subheader("Query Response")
        st.write(response)
    else:
        st.write("Please upload a PDF or enter a Job Description to proceed.")

elif submit_skill_analysis:
    if uploaded_file is not None and top_skills:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response("", pdf_content, input_prompt6, top_skills)
        st.subheader("Top Skill Analysis")
        st.write(response)
    else:
        st.write("Please upload a PDF and enter Top Skills to proceed.")

elif submit_jd_summarization:
    if input_text:
        response = get_gemini_response(input_text, "", input_prompt5)
        st.subheader("Job Description Summary")
        st.write(response)
    else:
        st.write("Please enter a Job Description to proceed.")
else:
        st.write("Please upload a PDF or enter a Job Description to proceed.")
