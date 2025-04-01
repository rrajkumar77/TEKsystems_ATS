import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#gemini function
def get_gemini_response(input):
    model=genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(input)
    return response.text

# Convert PDF to text
def input_pdf_text(uploaded_file):
    reader=pdf.PdfReader(uploaded_file)
    text=""
    for page in range(len(reader.pages)):
        page=reader.pages[page]
        text+=str(page.extract_text())
    return text

input_prompt ="""

### As a skilled ATS with advanced technology and domain knowledge, your role is to meticulously evaluate a candidate's resume based on the provided job description by:
Analyzing the resume for relevant skills, experiences, and qualifications that align with the job requirements. Look for key buzzwords and specific criteria outlined in the job description to determine the candidate's suitability for the position.
Calculating the match percentage between the resume and job description, providing a number and explanation.
Identifying missing key keywords from the resume compared to the job description.
Offering specific, actionable tips to enhance the resume and align it with job requirements.
Creating a table listing skills, years of experience, and relevant projects.
Your evaluation should be thorough, precise, and objective. It should ensure that the most qualified candidates are accurately identified based on their resume content concerning the job criteria.
Provide a detailed assessment of how well the resume matches the job requirements, highlighting strengths, weaknesses, and any potential areas of concern. Offer constructive feedback on how the candidate can enhance their resume to better align with the job description and improve their chances of securing the position.
Remember to utilize your expertise in technology and data science to conduct a comprehensive evaluation that optimizes the recruitment process for the hiring company. Your insights will play a crucial role in determining the candidate's compatibility with the job role.

resume={text}
jd={jd}
### Evaluation Output:
1. Calculate the match percentage between the resume and job description, providing a number 
2. Explain the match.
3. Identify missing keywords or skills from the resume compared to the job description.
4. Create a table that includes the top 5 skills, the required years of experience (JD), the candidate's years of experience (Resume), and the relevant projects they have worked on.
"""

##streamlit
st.title("Raj's Smart ATS")
st.text("Imporve your ATS resume score Match")
jd = st.text_area("Paste job description here")
uploaded_file= st.file_uploader("Upload your resume", type="pdf", help= "Please upload the pdf")

submit =  st.button('Check Your Score')
if submit:
    if uploaded_file is not None:
        text =  input_pdf_text(uploaded_file)
        response=get_gemini_response(input_prompt.format(text=text, jd=jd))
        st.subheader(response)
