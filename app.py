import pandas as pd
import streamlit as st
import joblib
import numpy as np
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
import isodate
from HTML_CSS import page1_home,page1_footer
import time 

page1_home()

pd.set_option("display.max_columns",None)
X=pd.read_csv("/Users/ggharish13/Data Science/Capstone Project/Content Monetization/X_feature")
y=pd.read_csv("/Users/ggharish13/Data Science/Capstone Project/Content Monetization/Y_feature")


# ------------------------------------------------------------------------------
#----------------------------------app------------------------------------------
# ------------------------------------------------------------------------------
# --- Streamlit UI ---
st.title("Youtube Revenue Prediction")
option = st.radio("", ["Manual","Using Youtube Link"],horizontal=True)

if option == "Manual":
    st.write("Fill the Values")
    col1, col2 = st.columns(2)

    with col1:
        UI_views = st.number_input("Views", min_value=1)
        UI_likes = st.number_input("Likes", min_value=1,max_value=UI_views)
        UI_Comments = st.number_input("Comments", min_value=1,max_value=UI_views)
        UI_Device = st.selectbox("Select Device:", ["Mobile", "TV", "Tablet"])
        UI_Category = st.selectbox("Select Category:", ["Entertainment",
                                                       "Gaming", "Lifestyle",
                                                       "Music", "Tech"])

    with col2:
        UI_video_length_minutes = st.number_input("video_length_minutes", min_value=1.0)
        UI_watch_time_minutes = st.number_input("watch_time_minutes", min_value=1.0,max_value=UI_video_length_minutes*UI_views)
        UI_subscribers = st.number_input("subscribers", min_value=10000)
        UI_Country = st.selectbox("Select Country:", ["CA", "DE", "IN", "UK", "US"])

    # Derived feature
    watch_fraction = (UI_watch_time_minutes / UI_views) / UI_video_length_minutes if UI_views > 0 else 0
    engagement_rate=UI_views/(UI_likes+UI_Comments)


    # Match training features
    Scalar_key_list = ["views", "likes", "comments",
                       "watch_time_minutes", "video_length_minutes",
                       "subscribers", "watch_fraction","engagement_rate"]

    Scalar_Value_list = [UI_views, UI_likes, UI_Comments,
                         UI_watch_time_minutes, UI_video_length_minutes,
                         UI_subscribers, watch_fraction, engagement_rate]

    # Build single-row DataFrame
    input_df = pd.DataFrame([dict(zip(Scalar_key_list, Scalar_Value_list))])

    scaler=joblib.load("scaler.pkl")
    # Scale using previously fitted scaler
    scaled_input = scaler.transform(input_df)
    scaled_df=pd.DataFrame()
    scaled_df[X.keys()[:8]]=scaled_input

    ohe_data = pd.DataFrame(np.zeros((1, len(X.keys()[8:]))), columns=X.keys()[8:])
    ohe_data[f"country_{UI_Country}"]=1
    ohe_data[f"device_{UI_Device}"]=1
    ohe_data[f"category_{UI_Category}"]=1
    predict_data=pd.concat([scaled_df, ohe_data], axis=1)

    # Predict with trained model
    if st.button("Predict Revenue"):
        model=joblib.load("model_LR.pkl")
        prediction = model.predict(predict_data)
        if prediction<0:
            with st.spinner("Calculating... Please wait ⏳"):
                time.sleep(1)
            st.error(f"❌ Estimated Revenue: $0")
        else:
            if prediction[0][0]>10000:
                with st.spinner("Crunching the numbers... Hang tight ⏳"):
                    time.sleep(3)
                with st.spinner("Think so, the Amount is Huge... Please wait ⏳"):
                    time.sleep(3)
                with st.spinner("Almost there! Preparing your big surprise ✨"):
                    time.sleep(3)
                if UI_Country=="CA":
                    st.success(f"✅ Estimated Revenue: CA$ {round(prediction[0][0]*1.39,2)}")
                elif UI_Country=="DE":
                    st.success(f"✅ Estimated Revenue: € {round(prediction[0][0]*0.85,2)}")
                elif UI_Country=="IN":
                    st.success(f"✅ Estimated Revenue: ₹ {round(prediction[0][0]*88.77,2)}")
                elif UI_Country=="UK":
                    st.success(f"✅ Estimated Revenue: £ {round(prediction[0][0]*0.74,2)}")
                else:
                    st.success(f"✅ Estimated Revenue: $ {round(prediction[0][0],2)}")

            else:
                with st.spinner("Calculating... Please wait ⏳"):
                    time.sleep(3)
                if UI_Country=="CA":
                    st.success(f"✅ Estimated Revenue: CA$ {round(prediction[0][0]*1.39,2)}")
                elif UI_Country=="DE":
                    st.success(f"✅ Estimated Revenue: € {round(prediction[0][0]*0.85,2)}")
                elif UI_Country=="IN":
                    st.success(f"✅ Estimated Revenue: ₹ {round(prediction[0][0]*88.77,2)}")
                elif UI_Country=="UK":
                    st.success(f"✅ Estimated Revenue: £ {round(prediction[0][0]*0.74,2)}")
                else:
                    st.success(f"✅ Estimated Revenue: $ {round(prediction[0][0],2)}")

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

X=pd.read_csv("/Users/ggharish13/Data Science/Capstone Project/Content Monetization/X_feature")

def get_youtube_video_data(youtube_url, api_key):
    # Extract video ID from URL
    parsed_url = urlparse(youtube_url)
    if parsed_url.hostname in ["youtu.be"]:
        video_id = parsed_url.path[1:]
    else:
        video_id = parse_qs(parsed_url.query).get("v")
        if video_id:
            video_id = video_id[0]
        else:
            raise ValueError("Invalid YouTube URL")

    # Build API client
    youtube = build("youtube", "v3", developerKey=api_key)

    # Request video details
    response = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    ).execute()

    if not response['items']:
        raise ValueError("Video not found or invalid ID")

    video_info = response['items'][0]

    channel_id = video_info['snippet']['channelId']
    channel_info = youtube.channels().list(
        part="statistics",
        id=channel_id
    ).execute()

    subscribers = int(channel_info['items'][0]['statistics'].get('subscriberCount', 0))
    video_response = youtube.videos().list(part="contentDetails", id=video_id).execute()
    duration = video_response['items'][0]['contentDetails']['duration'] 

    # Convert to minutes
    video_length_minutes = isodate.parse_duration(duration).total_seconds() / 60

    channel_id = video_info['snippet']['channelId']
    channel_info = youtube.channels().list(part="snippet", id=channel_id).execute()
    channel_location = channel_info['items'][0]['snippet'].get('country', "Unknown")

    # Extract data
    data = {
        "views": int(video_info['statistics'].get('viewCount', 0)),
        "likes": int(video_info['statistics'].get('likeCount', 0)),
        "comments": int(video_info['statistics'].get('commentCount', 0)),"watch_time_minutes":0,
        "video_length_minutes":video_length_minutes,
        "subscribers": subscribers,"watch_fraction":0
    }

    return data,channel_location

def predit(input_url,api_key):
    video_data,Loc = get_youtube_video_data(input_url, api_key)
    video_data["watch_fraction"] = (video_data["watch_time_minutes"] / video_data["views"]) / video_data["video_length_minutes"] if video_data["views"] > 0 else 0
    video_data["engagement_rate"]=video_data["views"]/(video_data["likes"]+video_data["comments"])
    video_data["watch_time_minutes"]=((video_data["views"]*video_data["video_length_minutes"])*0.90)*0.66
    video_data_p1=pd.DataFrame([video_data])
    video_data["watch_time_minutes"]=((video_data["views"]*video_data["video_length_minutes"])*0.33)*0.66
    video_data_p2=pd.DataFrame([video_data])

    scaler=joblib.load("scaler.pkl")
    # Scale using previously fitted scaler
    scaled_input_p1= scaler.transform(video_data_p1)
    scaled_df1=pd.DataFrame()
    scaled_df1[video_data_p1.keys()]=scaled_input_p1

    # Scale using previously fitted scaler
    scaled_input_p2= scaler.transform(video_data_p2)
    scaled_df2=pd.DataFrame()
    scaled_df2[video_data_p2.keys()]=scaled_input_p2

    ohe_data = pd.DataFrame(np.zeros((1, len(X.keys()[8:]))), columns=X.keys()[8:])
    ohe_data[f"country_{Loc}"]=1
    predict_data=pd.concat([scaled_df1, ohe_data], axis=1)

    model=joblib.load("model_LR.pkl")
    prediction1= model.predict(predict_data)

    ohe_data = pd.DataFrame(np.zeros((1, len(X.keys()[8:]))), columns=X.keys()[8:])
    ohe_data[f"country_{Loc}"]=1
    predict_data=pd.concat([scaled_df2, ohe_data], axis=1)

    prediction2= model.predict(predict_data)
    return prediction1,prediction2,video_data

if option == "Using Youtube Link":
    input_url=st.text_input("Paste the URL","https://youtu.be/bFwI1SJcMo0?si=bN2UVEN4LJQ-izx-")
    api_key = "YOUR_API_KEY"
    prediction1,prediction2,video_data=predit(input_url,api_key)
    VIDEO_URL = input_url
    st.video(VIDEO_URL)

    if st.button("Predict Revenue"):
        if video_data["subscribers"]>9999:
            if prediction2[0][0]>10000:
                with st.spinner("Crunching the numbers... Hang tight ⏳"):
                    time.sleep(3)
                with st.spinner("Think so, the Amount is Huge... Please wait ⏳"):
                    time.sleep(3)
                with st.spinner("Almost there! Preparing your big surprise ✨"):
                    time.sleep(3)
                st.success(f"✅ Estimated Revenue: {round(prediction2[0][0],2)} to {round(prediction1[0][0],2)} USD")

            else:
                with st.spinner("Calculating... Please wait ⏳"):
                    time.sleep(3)
                st.success(f"✅ Estimated Revenue: {round(prediction2[0][0],2)} to {round(prediction1[0][0],2)} USD")
        else:
            with st.spinner("Calculating... Please wait ⏳"):
                time.sleep(1)
            st.error(f"❌ Not Eligible for Monetization")

page1_footer()
