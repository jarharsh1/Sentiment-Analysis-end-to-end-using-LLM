# Sentiment-Analysis-end-to-end-using-LLM

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/LLM-GPT-008080?style=for-the-badge&logo=openai" alt="LLM GPT">
  <img src="https://img.shields.io/badge/Transformers-Hugging%20Face-FFD700?style=for-the-badge&logo=huggingface" alt="Hugging Face Transformers">
  <img src="https://img.shields.io/badge/Flask-2.3.3-000000?style=for-the-badge&logo=flask" alt="Flask Version">
</p>

This repository contains an end-to-end sentiment analysis project leveraging Large Language Models (LLMs), specifically **GPT models**, and the **Hugging Face Transformers library** to perform sentiment classification on textual data. The project includes a robust backend API built with Flask for inference and a user-friendly frontend developed with Streamlit for interaction.

## ✨ Features

* **GPT-Powered Sentiment Analysis**: Utilizes powerful GPT models for highly accurate sentiment detection.
* **Hugging Face Transformers Integration**: Leverages the Transformers library for easy access to state-of-the-art NLP models.
* **Flexible Backend API**: Built with **Flask** for a lightweight, robust, and efficient sentiment prediction service.
* **Interactive Frontend**: Developed with Streamlit for a simple and intuitive user interface.
* **Dockerized Deployment**: Easily deployable using Docker for consistent environments across different machines.
* **End-to-End Workflow**: Demonstrates a complete machine learning project lifecycle from model inference to serving.

## 🧠 Model Details & Sentiment Logic

This project leverages the power of Large Language Models (LLMs) and the Hugging Face Transformers library for performing sentiment analysis.

* **Core LLM**: The primary sentiment classification is performed using a **GPT model** (e.g., OpenAI's `gpt-3.5-turbo` or a similar variant). The model is prompted to analyze the input text and classify its sentiment (e.g., positive, negative, neutral) or provide a sentiment score.
* **Hugging Face Transformers**: The `transformers` library from Hugging Face is used to streamline the interaction with these models. Depending on the exact implementation, this might involve:
    * **Direct API Calls**: If using services like OpenAI, `transformers` can wrap around the API calls for easier integration.
    * **Zero-Shot / Few-Shot Classification**: For models directly available through Hugging Face (e.g., `distilbert-base-uncased-finetuned-sst-2-english`), the library is used to load the pre-trained model and tokenizer for immediate sentiment inference without requiring specific fine-tuning.
    * **Fine-tuned Models**: In more advanced scenarios, a model might have been fine-tuned on a custom sentiment dataset, and the `transformers` library facilitates loading and using such fine-tuned checkpoints.

The `app.py` (Flask backend) handles the input text, passes it to the chosen LLM/model via the `transformers` library (or direct API call), processes the model's output, and returns the inferred sentiment.

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

* Python 3.9+
* Docker (optional, for containerized deployment)
* Git

```bash
git clone [https://github.com/jarharsh1/Sentiment-Analysis-end-to-end-using-LLM.git](https://github.com/jarharsh1/Sentiment-Analysis-end-to-end-using-LLM.git)
cd Sentiment-Analysis-end-to-end-using-LLM
