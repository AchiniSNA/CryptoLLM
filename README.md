# Crypto Price Forecasting using Large Language Models (LLMs)
This project explores the use of **Large Language Models (LLMs)** for **univariate time series forecasting** of different types of cryptocurrency prices. The approach leverages structured prompting and the multimodal architecture to improve forecasting accuracy using historical price data and prompts which contains contextual and statistical details about data.

---

## 📌 Project Objectives

- Predict future cryptocurrency prices.
- Explore the capability of LLMs in time series forecasting tasks.
- Utilize the multimodal embedding capabilities for temporal data.

---

## Requirements

- numpy>=1.26.0
- pandas>=2.2.0
- matplotlib>=3.8.0
- yfinance
- plotly
- neuralforecast
- torch>=2.2.0
- transformers>=4.37.0
- tokenizers
- sentencepiece
- streamlit

---

## ⚙️ How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AchiniSNA/CrptoLLM.git
  
2. **To install dependencies:**
```bash
pip install -r requirements.txt
```
3. **Run the code**
   ```bash
   streamlit run app.py
