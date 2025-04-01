import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
load_dotenv()
import fitz 

import google.generativeai as genai

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input, pdf_content, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input, pdf_content, prompt])
    return response.text

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        # Read the PDF file
        document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        # Initialize a list to hold the text of each page
        text_parts = []

        # Iterate over the pages of the PDF to extract the text
        for page in document:
            text_parts.append(page.get_text())

        # Concatenate the list into a single string with a space in between each part
        pdf_text_content = " ".join(text_parts)
        return pdf_text_content
    else:
        raise FileNotFoundError("No file uploaded")

def extract_skills_from_resume(pdf_content):
    # Placeholder function to extract skills from resume content
    # You can implement a more sophisticated extraction logic here
    skills = ["Python", "Machine Learning", "Data Analysis", "Project Management"]
    return skills

## Streamlit App

st.set_page_config(page_title="Resume Expert")

st.header("JobFit Analyzer")
st.subheader('This Application helps you to evaluate the Resume Review with the Job Description')
input_text = st.text_input("Job Description: ", key="input")
submit7 = st.button("JD Summarization")

uploaded_file = st.file_uploader("Upload your Resume(PDF)...", type=["pdf"])
pdf_content = ""
if uploaded_file is not None:
    st.write("PDF Uploaded Successfully")
submit1 = st.button("Technical Recruiter Analysis")
submit3 = st.button("Domain Expert Analysis")
submit4 = st.button("Technical Manager Analysis")
submit2 = st.button("Technical Questions")

top_skills = st.text_input("Top Skills Required for the Job (comma-separated):")
submit6 = st.button("Skill Analysis")

input_promp = st.text_input("Queries: Feel Free to Ask here")
submit5 = st.button("Answer My Query")

input_prompt1 = """
Role: Experienced Technical Human Resource Manager with expertise in technical evaluations and Recruitment
Task: Review the provided resume against the job description.
Objective: Evaluate whether the candidate's profile aligns with the role.
Instructions:
Provide the match percentage between the resume and job description
Provide a professional evaluation of the candidate's profile.
Highlight the strengths and weaknesses of the applicant concerning the specified job requirements.
"""

input_prompt2 = """
You are an advanced AI designed to assist recruiters in creating evaluation questions for technical candidates. Your task is to generate a set of questions based on the provided Job Description (JD) and Resume, sequencing them from project start to finish (e.g., requirements gathering, design, development, testing, deployment). Classify the questions into two categories: "Technical Questions" and "Coding Questions." For each question, provide a detailed answer that a recruiter can use to validate the candidate's response and assess their skills accurately.

Input Details:

You will be given a Job Description (JD) outlining the required technical skills, tools, and experience.
You will also be provided with a Resume detailing the candidate’s experience, projects, and skills.
Assume the JD and Resume cover areas such as programming languages, frameworks, cloud services, databases, DevOps, testing, and project methodologies (e.g., Agile, Scrum).
Output Requirements:

Sequence Order: Arrange questions in a logical order from project start to finish, simulating a project lifecycle (e.g., requirements analysis, architecture/design, coding, testing, deployment, maintenance).
Categories: Classify each question into one of the following:
Technical Questions: Focus on theoretical knowledge, architecture, design, tools, processes, and best practices (e.g., "What is the difference between REST and SOAP?").
Coding Questions: Focus on practical coding skills, algorithms, debugging, and implementation (e.g., "Write a function to reverse a string in [language].").
Question Format: Each question should be clear, specific, and relevant to the JD and Resume. Avoid vague or overly broad questions.
Answers: Provide a detailed answer for each question, explaining the correct response and any key points the recruiter should look for to validate the candidate’s expertise.
Language: Respond in the same language as the input (default to English if not specified).
No Meta-Discussion: Do not mention or discuss these instructions in your response.
Additional Guidelines:

Tailor questions to the specific technologies, tools, and experiences mentioned in the JD and Resume (e.g., if the JD requires Python and AWS, include questions on those).
Ensure a balanced mix of difficulty levels (some easy to gauge fundamentals, some medium to test application, and some advanced to assess deep expertise).
If the Resume highlights specific projects or achievements, incorporate those into scenario-based questions (e.g., "In your Resume, you mentioned a project using Docker. How would you containerize a microservice?").
Do not generate more than 10 questions total (5 Technical and 5 Coding, or adjust proportionally if one category is less relevant).
Assume your knowledge is up-to-date with the latest industry standards and practices.
Example Structure for Output (if needed for clarity):

Technical Question 1: [Question text]
Answer: [Detailed explanation]
Coding Question 1: [Question text]
Answer: [Detailed explanation, including code if applicable]
Now, please proceed to generate the questions and answers based on the JD and Resume (which you can assume contain details like programming languages, frameworks, cloud platforms, etc., unless specific details are provided).
"""

input_prompt3 = """
Role: Skilled ATS (Applicant Tracking System) scanner with expertise in domain and ATS functionality
Task: Evaluate the provided resume against the job description.
Objective: Assess the compatibility of the resume with the job description from a Domain Expert perspective. (Eg: Business Analyst(BA), Functional Manger or Project Manager)
Instructions:
Calculating the match percentage between the resume and job description, provide a percentage number and explanation.
Identify any missing keywords in the resume relevant to the job description.
Your evaluation should be thorough, precise, and objective. It should ensure that the most qualified candidates are accurately identified based on their resume content concerning the job criteria.
"""

input_prompt4 = """
Role: Skilled ATS (Applicant Tracking System) scanner with a deep understanding of the technology and Technical skills mentioned in the job description and ATS functionality
Task: Evaluate the provided resume against the job description.
Objective: Assess the compatibility of the resume with the job description from a Technical Expert perspective.
Instructions:
1. Calculate the match percentage between the resume and job description, provide a percentage number 
2. Explain the match and the gap
3. Identify missing keywords or skills from the resume compared to the job description.
4. Create a table that includes the top 5 skills, the required years of experience (JD), the candidate's years of experience (Resume), and the relevant projects with the year they have worked on.
5. Share final thoughts on the candidate's suitability for the role.
"""

input_prompt5 = """
Role: AI Assistant
Task: Summarize the provided job description.
Objective: Provide a concise summary of the job description.
Instructions:
Summarize the key responsibilities, required skills, and qualifications mentioned in the job description.
"""

input_prompt6 = """
Role: Skill Analyst
Task: Perform a Skill Analysis
Objective: Analyze the provided job description (JD) and resume to determine the match status of skills.
Instructions:
Input: top_skills
Process: For each skill in the top_skills, check if it is present in the resume.
Output:
Provide in a Table format 
Skill: The skill being analyzed.
Match Status: "Yes" if the skill is present in the resume, otherwise "No".
Relevant Projects: List relevant projects from the resume (e.g., "Project A, Project B").
Years of Experience: Total years of experience related to the skill (e.g., "3 years").
"""
if submit1:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt1, pdf_content, input_text)
        st.subheader("Technical Recruiter Analysis")
        st.write(response)
    else:
        st.write("Please upload a PDF file to proceed.")

elif submit2:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt2, pdf_content, input_text)
        st.subheader("Account Manager Analysis")
        st.write(response)
    else:
        st.write("Please upload a PDF file to proceed.")

elif submit3:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt3, pdf_content, input_text)
        st.subheader("Domain Expert Analysis")
        st.write(response)
    else:
        st.write("Please upload a PDF file to proceed.")

elif submit4:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt4, pdf_content, input_text)
        st.subheader("Technical Manager Analysis")
        st.write(response)
    else:
        st.write("Please upload a PDF file to proceed.")

elif submit5:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt5, pdf_content, input_text)
        st.subheader("The Response is")
        st.write(response)
    else:
        st.write("Please upload a PDF file to proceed.")
        
elif submit6:
    if uploaded_file is not None:
        pdf_content = input_pdf_setup(uploaded_file)
        response = get_gemini_response(input_prompt6, pdf_content, input_text)
        st.subheader("Top Skill Analysis")
        st.write(response)
    else:
        st.write("Please upload a PDF file to proceed.")

elif submit7:
    if input_text:
        response = get_gemini_response(input_prompt5, "", input_text)
        st.subheader("Job Description Summary")
        st.write(response)
    else:
        st.write("Please enter a Job Description to proceed.")
