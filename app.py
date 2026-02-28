from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os

app = Flask(__name__)
CORS(app)

# Get OpenAI API key from Railway environment variables
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ------------------------------
# Resume Analysis Endpoint
# ------------------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        text = data.get("text")

        if not text:
            return jsonify({"error": "Resume text required"}), 400

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
You are a professional AI career advisor.
Analyze the resume deeply and provide:

1. Strengths
2. Weaknesses
3. Suitable Career Paths
4. Skill Gaps
5. 6-Month Improvement Plan
                    """
                },
                {"role": "user", "content": text}
            ],
            temperature=0.7
        )

        return jsonify({"result": response.choices[0].message.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------
# Career Plan Endpoint
# ------------------------------
@app.route("/career", methods=["POST"])
def career():
    try:
        data = request.json
        skills = data.get("skills")

        if not skills:
            return jsonify({"error": "Skills input required"}), 400

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
You are a professional career strategist.

Based on skills and interests, generate:

1. Best Career Options
2. Why they fit
3. Step-by-step Roadmap
4. Recommended Certifications
5. Expected Salary Range
                    """
                },
                {"role": "user", "content": skills}
            ],
            temperature=0.7
        )

        return jsonify({"result": response.choices[0].message.content})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------
# Railway Production Config
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
