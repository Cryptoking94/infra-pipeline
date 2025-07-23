import os
import json
import boto3
from pathlib import Path

# Replace with your Bedrock config
REGION = "us-east-1"
MODEL_ID = "arn:aws:bedrock:us-east-1:879097804956:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"

PER_FILE_PROMPT = """
You are a senior cloud infrastructure reviewer.

Analyze the following Terraform code and provide a concise review in this structure:

---
### Cost (Short)
- Briefly estimate cost impact of major resources only (no need for precise calculation).
- Mention only resources with significant cost impact.

### Security (Short)
- Briefly note any serious security risks or misconfigurations, if any.

### Best Practices (Short)
- List only the most important missing best practices or improvements, if any.

### Code Quality & Optimization (Short)
- Point out any redundant, inefficient, or overly complex code patterns.
- Mention hardcoded values, inconsistent naming, poor modularization, or underused variables.
- Highlight any readability or maintainability issues.
---

Be very brief, avoid unnecessary details or repetition. If a section does not apply, write "No significant issues found."

Here is the Terraform code:

<CODE>
{code_content}
</CODE>
"""
FINAL_SUMMARY_PROMPT = """
You are a senior cloud and security reviewer.

You are given concise review notes for multiple Terraform files in a project, each with three sections (Cost, Security, Best Practices). Your job is to create a single, comprehensive, non-redundant summary with these sections and a highly professional tone. The final summary should be less that 1000 characters
---


## 1. Security Risks
- Provide a prioritized, grouped list of all unique and critical security issues or vulnerabilities identified.
- For each risk, include a brief description, potential impact, and recommended remediation.
- Focus especially on issues that could impact compliance, data confidentiality, or availability.
- Use clear subheadings or bullet points for each major category of risk.

## 2. Best Practice Recommendations
- Summarize the most important infrastructure, DevOps, and cloud best practices not fully followed across the project.
- Provide actionable, prioritized recommendations for improvement, using clear and concise language.
- Group recommendations by theme (e.g., resource optimization, security posture, maintainability, automation).
- Optionally, acknowledge areas where best practices are already being followed well.

## 3. Code Quality & Optimization
- Highlight inefficiencies, redundant blocks, or overly complex logic in the Terraform codebase.
- Flag inconsistent naming conventions, hard-coded values, and underutilized modules or variables.
- Recommend improvements for readability, maintainability, modularity, and performance.
- Emphasize DRY (Don't Repeat Yourself) principles, input validation, and reusable patterns.

---

**IMPORTANT:**
- Do NOT just aggregate file-level notes; instead, synthesize and deduplicate findings for a true project-wide view.
- Your output should be clear, actionable, and suitable for delivery to engineering leadership.
- Prefer structured output (tables and headings) over free-form text where appropriate.
"""

# === FUNCTIONS ===

def find_tf_files(root_dir):
    """Recursively find all .tf files under root_dir, EXCLUDING .terraform, .git, __pycache__."""
    exclude_dirs = {'.terraform', '.git', '__pycache__'}
    tf_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Remove unwanted directories from the search
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for filename in filenames:
            if filename.endswith('.tf'):
                tf_files.append(os.path.join(dirpath, filename))
    return tf_files

def start_bedrock_session():
    """Create and return a new Bedrock Agent session ID and client."""
    client = boto3.client("bedrock-agent-runtime", region_name=REGION)
    response = client.create_session()
    return response["sessionId"], client

def send_to_bedrock(prompt, model_id=MODEL_ID):
    """Send prompt to Bedrock and get response text."""
    bedrock = boto3.client("bedrock-runtime", region_name=REGION)
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": prompt.strip()}],
        "max_tokens": 2048,
    }
    response = bedrock.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )
    completion = json.loads(response['body'].read())
    if 'content' in completion:
        return completion['content'][0]['text']
    return "No response."

def analyze_file(file_path):
    """Analyze a single .tf file using Bedrock LLM."""
    with open(file_path, "r", encoding="utf-8") as f:
        code_content = f.read()

    prompt = PER_FILE_PROMPT.format(code_content=code_content)
    return send_to_bedrock(prompt)

def chunk_lines(text, chunk_size=100):
    """Split long text into chunks of chunk_size lines."""
    lines = text.splitlines()
    return ['\n'.join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]

def summarize_chunks(chunks):
    """Summarize each chunk, then combine and get a final summary."""
    chunk_summaries = []
    for idx, chunk in enumerate(chunks):
        print(f"Summarizing chunk {idx+1}/{len(chunks)}")
        prompt = FINAL_SUMMARY_PROMPT + "\n" + chunk
        chunk_summary = send_to_bedrock(prompt)
        chunk_summaries.append(chunk_summary)

    if len(chunk_summaries) > 1:
        print("Generating executive summary from chunk summaries...")
        final_prompt = FINAL_SUMMARY_PROMPT + "\n" + "\n\n".join(chunk_summaries)
        return send_to_bedrock(final_prompt)
    else:
        return chunk_summaries[0]

# === MAIN LOGIC ===
def main(root_dir):
    tf_files = find_tf_files(root_dir)
    print(f"Found {len(tf_files)} files for analysis:")
    for tf in tf_files:
        print(tf)

    if not tf_files:
        print("No .tf files found in the specified directory.")
        return

    # Always write the report to the Jenkins workspace directory
    REPORT_FILE = os.path.join(root_dir, "iac_code_review_bedrock.md")

    # Start Bedrock session for demonstration/logging (not strictly needed for stateless API use)
    session_id, agent_client = start_bedrock_session()
    print(f"Started Bedrock session: {session_id}")

    # Collect per-file short summaries in a list (but do not write them to the file)
    per_file_outputs = []
    for tf_file in tf_files:
        print(f"Analyzing: {tf_file}")
        result = analyze_file(tf_file)
        per_file_outputs.append(result)

    # Combine all per-file outputs into a single string for the final summary
    combined_per_file = "\n\n".join(per_file_outputs)
    print("Generating overall summary...")
    chunks = chunk_lines(combined_per_file, chunk_size=100)
    project_summary = summarize_chunks(chunks)

    # Only write the executive summary to the Markdown file in the workspace directory
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("# Project-Level Executive Summary\n\n" + project_summary + "\n")

    print(f"\n✅ Review complete! Only executive summary saved to `{REPORT_FILE}`.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} /path/to/iac/code")
    else:
        main(sys.argv[1])