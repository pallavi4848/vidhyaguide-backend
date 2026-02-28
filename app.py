from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data.get("text")

    if not text:
        return jsonify({"error": "Resume text required"}), 400

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional career advisor. Analyze resumes deeply and provide strengths, weaknesses, career paths, skill gaps, and a 6-month improvement plan."},
            {"role": "user", "content": text}
        ],
        temperature=0.7
    )

    return jsonify({"result": response.choices[0].message.content})


@app.route("/career", methods=["POST"])
def career():
    data = request.json
    skills = data.get("skills")

    if not skills:
        return jsonify({"error": "Skills input required"}), 400

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional career strategist. Based on skills and interests, suggest best career options, roadmap, certifications, and salary range."},
            {"role": "user", "content": skills}
        ],
        temperature=0.7
    )

    return jsonify({"result": response.choices[0].message.content})


if __name__ == "__main__":
    app.run()
