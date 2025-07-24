# 🚨 GBVTrackAI

Welcome to **GBVTrackAI** – a machine learning-powered tool designed to help detect and predict the likelihood of **Gender-Based Violence (GBV)** based on socio-economic factors.

Built with ❤️ using Python, Streamlit, and XGBoost.

---

## 📦 Project Structure

gbvtrackai/
├── data/ # Raw and cleaned datasets

├── notebooks/ # Jupyter notebooks for EDA and model development

├── UI/UX/ # streamlit Deployment

├── reports/ # PDF or markdown reports and TEST VIDEO

├── requirements.txt # All required Python packages

---

## 🧠 How It Works

1. **User inputs**: Age, Education level, Employment status, Income, Marital status.
2. **Model processes data** with a trained XGBoost classifier.
3. **Prediction displayed**: High or Low likelihood of experiencing GBV, plus a confidence score.

---

## 🚀 Running the App

### 🔧 Step 1: Clone the repo

```bash
git clone https://github.com/your-username/gbvtrackai.git](https://github.com/Lukembogo-dot/GBV_Prediction.git
cd gbvtrackai
```
## 🐍 Step 2: Install dependencies

```bash
pip install -r requirements.txt
```
## ▶️ Step 3: Launch the app

```bash
streamlit run app.py
```

## 🧪 Sample Input

- Age: 28

- Education: Tertiary

- Children: 3

- Employment: Employed

- Marital Status: Single

➡️ Result: Prediction: No Violence

## 🔐 Ethical Note

This app is only a supportive tool. It should never replace professional social, legal, or medical support. Always approach GBV-related predictions with empathy and discretion.

🛠️ Built With

🧠 Random-Forest – Handles imbalanced data like a champ.

📊 Pandas, Scikit-learn – For data wrangling and modeling.

🎈 Streamlit – Super simple UI for users.

## 🙌 Contributing
Pull requests are welcome. Let’s use tech to make a difference 💪.

## 📫 Contact
Made with purpose by Luke Mbogo.
Reach out: Lukembog5@gmail.com
