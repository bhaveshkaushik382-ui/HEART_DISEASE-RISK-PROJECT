from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def generate_advice(probability, age, bmi, cholesterol, bp, glucose, smoker):

    prompt = f"""
You are an AI healthcare assistant.

Patient Information:

Age: {age}
BMI: {bmi}
Total Cholesterol: {cholesterol}
Systolic BP: {bp}
Glucose: {glucose}
Smoking Per Day: {smoker}
Heart Disease Risk: {probability}%

Return ONLY clean HTML.

Use this structure:

<h4>📌 Risk Summary</h4>
<ul>
<li>...</li>
</ul>

<h4>🥗 Diet Recommendations</h4>
<ul>
<li>...</li>
</ul>

<h4>🏃 Exercise Recommendations</h4>
<ul>
<li>...</li>
</ul>

<h4>🌙 Lifestyle Improvements</h4>
<ul>
<li>...</li>
</ul>

<h4>⚠️ Disclaimer</h4>

<p>This information is educational and not a medical diagnosis.</p>

Rules:

- No markdown
- No triple backticks
- Short bullet points
- Maximum 150 words
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b:free",

        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )

    return response.choices[0].message.content