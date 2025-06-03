
import streamlit as st
from retriever import retrieve_relevant_chunks

# Prompt builder
def build_prompt(student_profile, retrieved_context):
    return f"""
You are a special education expert creating IEP transition goals for high school students with disabilities.

Student Profile:
{student_profile}

Reference Materials:
{retrieved_context}

Instructions:
- Create a clear, measurable postsecondary goal based on the student’s career interest.
- List 2–3 short-term objectives aligned with the goal.
- Suggest specific transition services to support those objectives.
- Include a final note on compliance with IDEA 2004.
- If needed, explain any assumptions you made.

Respond in this format:

Measurable Postsecondary Goals:
[Write two detailed goals here that explain what this student should focus on. One goal should relate to employment. One goal should relate to education and training.]

Short-Term Objectives:
1.
2.
3.

Transition Services:
1.
2.

Measurable annual goal aligned with standards
[Relate the postsecondary goals to soft skills (i.e. listening) that the student should focus on in order to achieve success in their desired career path while aligning it with standards.]

Compliance Note:
[Explain any assumptions or IDEA alignment.]

Response:
"""

from huggingface_hub import InferenceClient
HF_TOKEN = "insert your token"
client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta",
                        token = HF_TOKEN)
def generate_iep_plan(prompt):
    return client.text_generation(prompt, max_new_tokens=700)

# Streamlit UI
st.set_page_config(page_title="IEP Goal Generator", layout="wide")

st.title("🎯 IEP Goal Generator for Transition Planning")
st.markdown("Provide a student's background and needs to generate IEP-aligned goals using AI + educational standards.")

with st.form("iep_form"):
    student_profile = st.text_area(
        "👤 Enter Student Profile",
        placeholder="Clarence, 15-year-old sophomore with a behavior disorder, Completed the O*Net Interest Profiler assessment, Shows strong interest in the Enterprising category, Career interests: retail sales, driver/sales worker, Prefers hands-on learning over academic instruction, Assessment Results: O*Net Interest Profiler indicates strength in Enterprising activities, Career suggestions include retail salesperson or driver/sales worker, Student interview (Vision for the Future) indicates interest in working at Walmart",
        height=200
    )
    submitted = st.form_submit_button("Generate IEP Goals")

if submitted and student_profile.strip():
    with st.spinner("Retrieving educational and career context..."):
        context = retrieve_relevant_chunks(student_profile)
        trimmed_context = context[:2000]  # Trim to avoid overflow
        prompt = build_prompt(student_profile, trimmed_context)

    with st.spinner("Generating IEP goals..."):
        response = generate_iep_plan(prompt)

    st.subheader("📄 Generated IEP Plan")
    st.markdown(f"```text\n{response.strip()}\n```")

elif submitted:
    st.warning("Please enter a valid student profile.")
