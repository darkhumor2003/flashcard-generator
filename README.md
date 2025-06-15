# Flashcard Generator (LLM-Powered, Offline)

This is a Streamlit web app that generates flashcards from any uploaded PDF using a local language model (TinyLlama).  
No API keys or internet access required after the first model download.
I have also attached the screenshots of the working model in 5 images.

## Features

- Upload any PDF notes
- Uses `TinyLlama-1.1B-Chat-v1.0` model locally
- No API needed
- Generates Q&A flashcards from educational content
- Built with Streamlit + Hugging Face Transformers

## 📁 File Structure

flashcard-generator/
├── app.py # Main Streamlit app
├── utils.py # PDF text extraction
├── requirements.txt # All Python dependencies
├── README.md # Project overview
├── sample.pdf # Optional test file
└──screenshots of the model/ # 5 images of the working model

## Installation

1. **Clone the repo** or download the files

```bash
git clone https://github.com/your-username/flashcard-generator.git
cd flashcard-generator

2. Create virtual environment

python -m venv venv
venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Run the app

streamlit run app.py
