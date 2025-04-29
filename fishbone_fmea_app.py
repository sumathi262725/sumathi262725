import streamlit as st
import openai
import graphviz
import pandas as pd
from io import BytesIO
from transformers import pipeline

# ---------- CONFIG ----------
openai.api_key = "your_openai_api_key"  # Replace with your actual API key

nlp = pipeline("text2text-generation", model="google/flan-t5-small")

domain_templates = {
    "Finance": ["People", "Process", "Technology", "Source Systems", "Measurement"],
    "Healthcare": ["People", "Process", "Technology", "Regulations", "Data Standards"],
    "Retail": ["People", "Inventory", "Point of Sale", "ETL", "Vendor Data"]
}

# ---------- HELPERS ----------
def suggest_causes(description):
    prompt = f"List root causes for the following data quality issue: {description}"
    result = nlp(prompt, max_length=100)[0]['generated_text']
    return [cause.strip() for cause in result.split(",") if cause.strip()]

def to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# ---------- UI ----------
st.title("🔍 Data Quality Root Cause & FMEA Analysis")

domain = st.selectbox("Select Data Domain", list(domain_templates.keys()))
main_issue = st.text_input("Enter Main Data Quality Issue", "Missing Customer IDs")
categories = domain_templates.get(domain, [])

# NLP Suggestion
suggested_causes = []
if main_issue:
    with st.expander("💡 Auto-Suggest Causes (NLP)", expanded=True):
        if st.button("Generate Suggested Causes"):
            suggested_causes = suggest_causes(main_issue)
            st.write(suggested_causes)

# Capture Causes and FMEA Ratings
st.subheader("✏️ Enter Causes per Category + FMEA Ratings")

all_causes = []

for cat in categories:
    with st.expander(f"📂 {cat}"):
        default_causes = ", ".join([c for c in suggested_causes if cat.lower() in c.lower()]) if suggested_causes else ""
        raw_input = st.text_area(f"Enter causes for {cat} (comma-separated)", value=default_causes)
        causes = [c.strip() for c in raw_input.split(",") if c.strip()]
        
        for cause in causes:
            col1, col2, col3 = st.columns(3)
            with col1:
                severity = st.slider(f"Severity ({cause})", 1, 10, 5, key=f"{cat}_{cause}_sev")
            with col2:
                occurrence = st.slider(f"Occurrence ({cause})", 1, 10, 5, key=f"{cat}_{cause}_occ")
            with col3:
                detectability = st.slider(f"Detectability ({cause})", 1, 10, 5, key=f"{cat}_{cause}_det")
            rpn = severity * occurrence * detectability
            st.markdown(f"**🧮 RPN for '{cause}'**: `{rpn}` {'🟥 High Risk' if rpn >= 100 else ''}")
            all_causes.append({
                "Category": cat,
                "Cause": cause,
                "Severity": severity,
                "Occurrence": occurrence,
                "Detectability": detectability,
                "RPN": rpn
            })

# ---------- Fishbone Diagram ----------
st.subheader("📈 Fishbone (Ishikawa) Diagram")

dot = graphviz.Digraph()
dot.node("main", main_issue, shape="box", style="filled", color="lightblue")

for cat in categories:
    cat_node = f"cat_{cat}"
    dot.node(cat_node, cat, shape="ellipse", color="gray")
    dot.edge(cat_node, "main")
    for cause in [c for c in all_causes if c["Category"] == cat]:
        cause_node = f"{cat}_{cause['Cause']}"
        dot.node(cause_node, cause["Cause"], shape="note", color="lightgray")
        dot.edge(cause_node, cat_node)

st.graphviz_chart(dot)

# ---------- Table + Download ----------
if all_causes:
    st.subheader("📋 FMEA Table (All Causes)")
    df = pd.DataFrame(all_causes).sort_values(by="RPN", ascending=False)
    st.dataframe(df, use_container_width=True)

    st.download_button(
        label="📥 Download FMEA Table as CSV",
        data=to_csv(df),
        file_name="fmea_analysis.csv",
        mime="text/csv"
    )
