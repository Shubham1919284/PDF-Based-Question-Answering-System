from flask import Flask, request, jsonify, render_template

from pdf_processor import extract_all_pdfs
from chunker import create_chunks
from vector_store import build_vector_store, save_vector_store, load_vector_store
from rag_pipeline import answer_question, API_KEY

app=Flask(__name__)

if not API_KEY:
    raise ValueError("OPENROUTER_API_KEY is not set. Please set it in your environment variables or .env file.")

# First, check whether I already have a saved FAISS index
index, chunks = load_vector_store()

if index is None:
    # If no saved index exists, process PDFs and create one
    pages = extract_all_pdfs()
    chunks = create_chunks(pages)

    # Check whether any usable text was extracted
    if not chunks:
        raise ValueError("No text chunks found in the provided PDFs.")

    index, embeddings = build_vector_store(chunks)

    # Save the index and chunks for future use
    save_vector_store(index, chunks)

    print("New vector store created and saved.")

else:
    # Reuse the previously saved index and chunks
    print("Existing vector store loaded successfully.")

@app.route("/", methods=["GET"])
def index_page():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "API is running successfully"}), 200

@app.route("/ask", methods=["POST"])
def ask():

    # Get the JSON data sent by the user
    data = request.get_json(silent=True)

    # Check whether the request contains valid JSON
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON request"}), 400

    # Extract the question first
    question = data.get("question")

    # Validate the question
    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Invalid question format"}), 400

    # Send the question to my existing RAG pipeline
    try:
        result = answer_question(question.strip(), chunks, index)
        # Return the answer and source details
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": f"Failed to generate answer: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
