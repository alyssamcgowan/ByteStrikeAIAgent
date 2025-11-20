from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import langchain
import os
from dotenv import load_dotenv
import json

load_dotenv()

langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False

app = Flask(__name__)
CORS(app)  # Enable CORS

llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

systemPromptStr = ""
humanPromptStr = "Output only the structured JSON output classifying this product: {data}"
apiUrl = "https://xffd-cx0m-wnb0.n7d.xano.io/api:Yjo3M2L3/filtered_products"


prompt = ChatPromptTemplate.from_messages(
    [("system", systemPromptStr), ("human", humanPromptStr)]
)
chain = prompt | llm



@app.route('/query', methods=['POST'])
def prompt_llm():
    try:
        
        response = requests.post(
            apiUrl,
            json={"page": ""}
        )
        
        if response.status_code != 200:
            return jsonify({
                "error": f"API request failed: {response.status_code} - {response.text}"
            }), 400
        
        data = response.json()
        output = []
        
        for d in data:
            classificationJson = chain.invoke({"data":d})
            jsonStr = classificationJson.content
            jsonStr = jsonStr.lstrip("```json")
            jsonStr = jsonStr.rstrip("```")

            output.append(json.loads(jsonStr))

        
        return jsonify({
            "success": True,
            "data": output
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)