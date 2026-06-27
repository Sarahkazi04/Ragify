import json
import sys
from docx import Document
import os
import faiss
import numpy as np
import openai
from openai import OpenAI
import pandas as pd
import PyPDF2

# Set your OpenAI API key here
OPENAI_API_KEY = "xyz"

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Function to load embeddings from a JSON file
def load_embeddings(json_path):
    if not os.path.exists(json_path):
        print(f"Error: Embeddings file not found at {json_path}")
        sys.exit(1)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list) and all(isinstance(item, list) for item in data):
            print(f"Successfully loaded embeddings from '{json_path}'.")
            print(f"Number of embeddings: {len(data)}")
            print(f"Dimension of each embedding: {len(data[0])} (assuming all embeddings have the same dimension)")
            return data
        else:
            if isinstance(data, list):
                print(f"Loaded a single embedding vector from '{json_path}'.")
                print(f"Embedding dimension: {len(data)}")
                return [data]
            else:
                print(f"Error: Unexpected JSON structure in '{json_path}'. Expected a list of lists or a single list.")
                sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{json_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading embeddings: {e}")
        sys.exit(1)

# Function to load text from a Word document
def load_word_document(docx_path):
    if not os.path.exists(docx_path):
        print(f"Error: Word document not found at {docx_path}")
        sys.exit(1)
    try:
        doc = Document(docx_path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        print(f"Successfully loaded Word document from '{docx_path}'.")
        print(f"Number of non-empty paragraphs: {len(paragraphs)}")
        return paragraphs
    except Exception as e:
        print(f"An error occurred while loading the Word document: {e}")
        sys.exit(1)

# Function to load text from an Excel file
def load_excel_document(xlsx_path):
    if not os.path.exists(xlsx_path):
        print(f"Error: Excel document not found at {xlsx_path}")
        sys.exit(1)
    try:
        excel_data = pd.read_excel(xlsx_path, sheet_name=None)  # Read all sheets
        texts = []
        for sheet_name, df in excel_data.items():
            # Convert each cell to string and concatenate
            sheet_text = df.astype(str).apply(' | '.join, axis=1).tolist()
            texts.extend([cell for cell in sheet_text if cell.strip()])
        print(f"Successfully loaded Excel document from '{xlsx_path}'.")
        print(f"Number of non-empty cells: {len(texts)}")
        return texts
    except Exception as e:
        print(f"An error occurred while loading the Excel document: {e}")
        sys.exit(1)

# Function to load text from a PDF document
def load_pdf_document(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"Error: PDF document not found at {pdf_path}")
        sys.exit(1)
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        texts = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text:
                paragraphs = [para.strip() for para in text.split('\n') if para.strip()]
                texts.extend(paragraphs)
        print(f"Successfully loaded PDF document from '{pdf_path}'.")
        print(f"Number of non-empty paragraphs: {len(texts)}")
        return texts
    except Exception as e:
        print(f"An error occurred while loading the PDF document: {e}")
        sys.exit(1)

# Function to create a FAISS index and add embeddings to it
def create_faiss_index(embeddings):
    if not embeddings:
        print("Error: No embeddings to add to the FAISS index.")
        sys.exit(1)
    try:
        embedding_dim = len(embeddings[0])
        np_embeddings = np.array(embeddings).astype('float32')
        print(f"Converted embeddings to NumPy array with shape: {np_embeddings.shape}")
    except Exception as e:
        print(f"Error converting embeddings to NumPy array: {e}")
        sys.exit(1)
    try:
        index = faiss.IndexFlatL2(embedding_dim)
        print(f"Created FAISS IndexFlatL2 with dimension: {embedding_dim}")
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        sys.exit(1)
    try:
        index.add(np_embeddings)
        print(f"Added {index.ntotal} embeddings to the FAISS index.")
    except Exception as e:
        print(f"Error adding embeddings to FAISS index: {e}")
        sys.exit(1)
    return index

# Function to get embeddings from OpenAI
def get_embeddings(text, model="text-embedding-ada-002"):
    try:
        response = client.embeddings.create(
            input=[text],
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"An error occurred while fetching embeddings from OpenAI: {e}")
        sys.exit(1)

# Function to generate LLM response
def generate_llm_response(question, retrieved_paragraphs):
    context = "\n\n".join(retrieved_paragraphs)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"An error occurred while generating LLM response: {e}")
        return "I'm sorry, I couldn't generate a response at this time."

def main():
    folder_path = input("Enter folder path containing DOCX, XLSX, and PDF files: ").strip()
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory.")
        sys.exit(1)

    # Supported file extensions and their loaders
    supported_extensions = {
        '.docx': load_word_document,
        '.xlsx': load_excel_document,
        '.xls': load_excel_document,
        '.pdf': load_pdf_document
    }

    # Gather all supported files in the specified folder
    all_files = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in supported_extensions
    ]

    if not all_files:
        print("No supported files (DOCX, XLSX, PDF) found in the folder.")
        sys.exit(1)

    print("Found the following supported files:")
    for file in all_files:
        print(file)

    all_paragraphs = []
    all_embeddings = []

    for file_path in all_files:
        print(f"\nProcessing file: {file_path}")
        file_extension = os.path.splitext(file_path)[1].lower()
        loader_function = supported_extensions[file_extension]

        # Determine embeddings file path
        embeddings_path = f"{file_path}.embeddings.json"

        # Load or generate text content
        if file_extension == '.docx':
            paragraphs = loader_function(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            paragraphs = loader_function(file_path)
        elif file_extension == '.pdf':
            paragraphs = loader_function(file_path)
        else:
            print(f"Unsupported file type: {file_extension}. Skipping.")
            continue

        # Load or generate embeddings for the current file
        if os.path.exists(embeddings_path):
            print(f"Loading embeddings from {embeddings_path}")
            embeddings = load_embeddings(embeddings_path)
            if len(embeddings) != len(paragraphs):
                print(f"Mismatch in embeddings and paragraphs for {file_path}. Regenerating embeddings...")
                embeddings = []
                for idx, para in enumerate(paragraphs):
                    print(f"Generating embedding for text {idx + 1}/{len(paragraphs)}...")
                    embedding = get_embeddings(para)
                    embeddings.append(embedding)
                try:
                    with open(embeddings_path, 'w', encoding='utf-8') as f:
                        json.dump(embeddings, f)
                    print(f"Regenerated embeddings saved to '{embeddings_path}'.")
                except Exception as e:
                    print(f"Error saving embeddings to JSON: {e}")
                    sys.exit(1)
        else:
            print(f"Embeddings not found for {file_path}. Generating new embeddings.")
            embeddings = []
            for idx, para in enumerate(paragraphs):
                print(f"Generating embedding for text {idx + 1}/{len(paragraphs)}...")
                embedding = get_embeddings(para)
                embeddings.append(embedding)
            try:
                with open(embeddings_path, 'w', encoding='utf-8') as f:
                    json.dump(embeddings, f)
                print(f"Embeddings saved to '{embeddings_path}'.")
            except Exception as e:
                print(f"Error saving embeddings to JSON: {e}")
                sys.exit(1)

        all_paragraphs.extend(paragraphs)
        all_embeddings.extend(embeddings)

    print("\nCreating FAISS index with embeddings from all documents...")
    index = create_faiss_index(all_embeddings)

    print("\nYou can now ask questions. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nEnter your question: ")
            if question.lower() in ['exit', 'quit']:
                print("Exiting the program. Goodbye!")
                break

            print("Generating embedding for your question...")
            question_embedding = get_embeddings(question)
            question_embedding_np = np.array([question_embedding]).astype('float32')

            k = 3  # Number of nearest neighbors
            distances, indices = index.search(question_embedding_np, k)

            print(f"\nTop {k} results:")
            retrieved_paragraphs = []
            for rank, (distance, idx) in enumerate(zip(distances[0], indices[0]), start=1):
                if idx < len(all_paragraphs):
                    paragraph = all_paragraphs[idx]
                    retrieved_paragraphs.append(paragraph)
                    print(f"\nResult {rank}:")
                    print(f"Distance: {distance}")
                    print(f"Text: {paragraph}")
                else:
                    print(f"\nResult {rank}:")
                    print("Invalid index retrieved from FAISS.")

            print("\nGenerating response using GPT-4o...")
            llm_response = generate_llm_response(question, retrieved_paragraphs)
            print("\nAssistant's Response:")
            print(llm_response)

        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()