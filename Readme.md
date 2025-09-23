# YouTube Content Monetization Prediction  

This project predicts **YouTube Content Monetization revenue** based on video metadata, engagement metrics, and channel information. It combines data preprocessing, machine learning, and an interactive Streamlit web app.  

---

## 🚀 Features
- **Manual Input Mode**: Enter video stats (views, likes, comments, etc.) manually to predict revenue.  
- **YouTube Link Mode**: Paste a YouTube video URL, and the app fetches video metadata (views, likes, comments, subscribers, length, etc.) using the YouTube Data API.  
- **Revenue Prediction**: Uses a trained regression model (`model_LR.pkl`) to estimate monetization potential.  
- **Eligibility Check**: Automatically flags videos as "Not Eligible for Monetization" if the channel has <10,000 subscribers.  

---

## 📂 Project Structure
```
.
├── app.py                 # Streamlit app for manual input + YouTube link mode
├── DataCleaning.py        # Data preprocessing & feature engineering
├── scaler.pkl             # Saved StandardScaler for feature scaling
├── model_LR.pkl           # Trained regression model
├── X_feature              # Processed feature dataset (inputs)
├── Y_feature              # Processed target dataset (ad revenue)
└── README.md              # Documentation
```

---

## 🧹 Data Preprocessing (DataCleaning.py)
- Handles **missing values** for likes, comments, and watch time logically by grouping country and category and then 
taking mean from that.
- Creates derived features such as:  
  - `watch_fraction = (watch_time_minutes / views) / video_length_minutes`  
- Applies **standard scaling** to numeric features.  
- Applies **one-hot encoding** to categorical features (`country`, `device`, `category`).  
- Saves preprocessed datasets (`X_feature`, `Y_feature`) and scaler for inference.  

---

## 🖥️ Streamlit Application (app.py)
The app offers **two modes**:  

### 1. Manual Input Mode
Users can enter:  
- Views, Likes, Comments  
- Watch Time (minutes), Video Length (minutes)  
- Subscribers, Country, Device, Category  

The model scales inputs, applies one-hot encoding, and predicts expected revenue.  

### 2. YouTube Link Mode
- Accepts a **YouTube video link**.  
- Extracts metadata via **YouTube Data API v3**:
  - Views, Likes, Comments  
  - Video length  
  - Channel subscriber count & country  
- Preprocesses features and runs the model.  
- Returns **revenue prediction range**.  

---

## ⚙️ Installation & Setup

### 1. Clone repository
```bash
git clone https://github.com/harishgg13/Youtube-Content-Monetization-predict-model.git
cd youtube-revenue-predictor
```

### 2. Install dependencies
Make sure `pip` is installed:
```bash
python -m ensurepip --upgrade
```
Then install required libraries:
```bash
pip install -r requirements.txt
```

**requirements.txt** should include:
```
streamlit
pandas
numpy
scikit-learn
joblib
google-api-python-client
isodate
matplotlib
seaborn
```

### 3. Run preprocessing (optional)
```bash
python DataCleaning.py
```

### 4. Start Streamlit app
```bash
streamlit run app.py
```

---

## 🔑 API Setup
To use the **YouTube Link Mode**, you need a YouTube Data API key:
1. Go to [Google Cloud Console](https://console.cloud.google.com/).  
2. Create a new project and enable **YouTube Data API v3**.  
3. Generate an API key.  
4. Replace the placeholder in `app.py`:
   ```python
   api_key = "YOUR_API_KEY"
   ```

---

## 📊 Model
- **Algorithm**: Linear Regression (saved as `model_LR.pkl`)  
- **Features**:
  - Scaled: `views`, `likes`, `comments`, `watch_time_minutes`, `video_length_minutes`, `subscribers`, `watch_fraction`  
  - Encoded: `device_*`, `country_*`, `category_*`  
- **Target**: `ad_revenue_usd`  

---

## 📈 Example Usage
### Manual Mode
1. Select **Manual**.  
2. Enter video statistics.  
3. Click **Predict Revenue** to view estimated earnings.  

### YouTube Link Mode
1. Select **Using YouTube Link**.  
2. Paste a YouTube URL.  
3. Click **Predict Revenue**.  

---

## ✅ Monetization Eligibility Rules
- YouTube requires **≥10,000 subscribers** to be eligible.  
- If a channel has fewer subscribers, the app returns:
  ```
  ❌ Not Eligible for Monetization
  ```

---

## 📌 Future Improvements
- Integrate YouTube Analytics API for actual watch-time data.  
- Support more countries, devices, and categories.  
- Add deep learning models for improved accuracy.  
