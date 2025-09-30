# notes
# for country and category one hot encoding is used.

# ------------------------------------------------------------------------------

# import the necessary library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import joblib
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


# settings to show whole data set columns
pd.set_option("display.max_columns", None)

# loading the dataset
data_original = pd.read_csv("/Users/ggharish13/Data Science/Capstone Project/Content Monetization/youtube_ad_revenue_dataset.csv")
data=data_original.copy()

# ------------------------------------------------------------------------------
# Exploratory Data Analysis

print(data.info())

# Outliers checking
fig=plt.hist(x=data["likes"]) #checking he distribution of datas and for outliers
# plt.show()
fig=plt.hist(x=data["views"]) #checking the distribution of datas and for outliers
# plt.show()
fig=plt.hist(x=data["comments"]) #checking the distribution of datas and for outliers
# plt.show()

for i in range (0,len(data.keys())):
    plt.scatter(x=data.iloc[:,i],y=data["ad_revenue_usd"])
    keys=data.keys()
    plt.xlabel(f"{keys[i]}")
    plt.ylabel("Ad Revenue")
    plt.show()

# ------------------------------------------------------------------------------

data["date"] = pd.to_datetime(data["date"])
data_grp=data.groupby(["country","category"])["likes"].mean().reset_index().rename(columns={"likes":"Avg_likes"})
data=data.merge(data_grp,how="left",on=["country","category"])
data_grp=data.groupby(["country","category"])["views"].mean().reset_index().rename(columns={"views":"Avg_views"})
data=data.merge(data_grp,how="left",on=["country","category"])
data_grp=data.groupby(["country","category"])["comments"].median().reset_index().sort_values("comments").rename(columns={"comments":"Avg_comments"})
data=data.merge(data_grp,how="left",on=["country","category"])

data["Avg_like_per_view"]=data["Avg_likes"]/data["Avg_views"]
data["Avg_comments_per_view"]=data["Avg_comments"]/data["Avg_views"]

data["Like_derived"]=round(data["Avg_like_per_view"]*data["views"],0)
data["comments_derived"]=round(data["Avg_comments_per_view"]*data["views"],0)
data["likes"]=data["likes"].fillna(data["Like_derived"])

data_new=data.dropna(subset="watch_time_minutes").reset_index(drop=True) # since, the watch time missing value is less than 5%, decided to drop.
data_new["comments"]=data_new["comments"].fillna(data["comments_derived"]) # comments manipulation is aggregate. 
# so we manupulate it after removing the watch time. this is less than 5% manipulation. (116295-110483)/116295

print(data_new[data_new["subscribers"]<1000]) # checking whether subscriber less than 1000 is there. since YT need minimum 1000 Subs

# ------------------------------------------------------------------------------

# print(data_new[["date", "ad_revenue_usd"]].corr()) # Checked correlation. corr_value= 0.002587
# adding additional date information have less correlation with target feature and doesn't affect the data accuracy.

# data_new["date"]=pd.to_datetime(data_new["date"])
# data_new["Year"]=data_new["date"].dt.year
# data_new["Month"]=data_new["date"].dt.month

# ------------------------------------------------------------------------------
# removing 

data_new=data_new.drop(columns=["video_id","date","Avg_likes","Avg_views","Avg_comments","Avg_like_per_view","Avg_comments_per_view",
                                "Like_derived","comments_derived"]) # since video_id is unique and weak correlation, we dont need it. so we drop it

# ------------------------------------------------------------------------------

data_new["comments"]=data_new["comments"].astype(int)
data_new["likes"]=data_new["likes"].astype(int)
data_new=data_new.drop_duplicates().reset_index(drop=True)

# ------------------------------------------------------------------------------
# new columns 
data_new["average_watchtime_per_view"]=data_new["watch_time_minutes"]/data_new["views"]
data_new["watch_fraction"]=data_new["average_watchtime_per_view"]/data_new["video_length_minutes"]
data_new["engagement_rate_per_view"]=(data_new["likes"]+data_new["comments"])/data_new["views"] #vif is more. so dont include 
data_new["engagement_rate"]=data_new["views"]/(data_new["likes"]+data_new["comments"])
data_new=data_new.drop(columns=["average_watchtime_per_view","engagement_rate_per_view"]) # removing due to high VIF factor

# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Encoding Device to One_Hot Encoding
data_new_enc=data_new[["category","device","country"]]
y=data_new[["ad_revenue_usd"]]
X_data=data_new.drop(columns=["category","device","country","ad_revenue_usd"],)
keys=X_data.keys()
scalar=StandardScaler()
x_scaled=scalar.fit_transform(X_data)
X_data[keys]=x_scaled

# ------------------------------------------------------------------------------
# Encoding Device to One_Hot Encoding
# tuned this, according to VIF, since we get inf in all One Hot Encoded columns. after checking the LOC-23 from ML.py
OHE=OneHotEncoder(sparse_output=False,drop="first")
OHE_enc=OHE.fit_transform(data_new_enc[["device"]])
OHE_df=pd.DataFrame(OHE_enc,columns=OHE.get_feature_names_out(["device"])).astype(int)
X_data[OHE.get_feature_names_out(["device"])]=OHE_enc.astype(int)

OHE_enc=OHE.fit_transform(data_new_enc[["country"]])
OHE_df=pd.DataFrame(OHE_enc,columns=OHE.get_feature_names_out(["country"])).astype(int)
X_data[OHE.get_feature_names_out(["country"])]=OHE_enc.astype(int)

OHE_enc=OHE.fit_transform(data_new_enc[["category"]])
OHE_df=pd.DataFrame(OHE_enc,columns=OHE.get_feature_names_out(["category"])).astype(int)
X_data[OHE.get_feature_names_out(["category"])]=OHE_enc.astype(int)

# fig=sns.pairplot(X_data)
# plt.show()
# print(X_data.corr(method="pearson")) # to find the correlation

X=X_data.copy()
# print(X)

# ------------------------------------------------------------------------------
# Checking Correlation
# print(X.corr()) # there the no strong correlation between input features

# Checking VIF
X_Vif=pd.DataFrame()
X_Vif=X
X_Vif=sm.add_constant(X_Vif)
vif=pd.DataFrame()
vif["Features"]=X_Vif.keys()
vif["values"]=[variance_inflation_factor(X_Vif.values,i) for i in range(X_Vif.shape[1])]
print(vif)
# print(X)
# print(y)

# ------------------------------------------------------------------------------

X.to_csv("X_feature",index=False)
y.to_csv("Y_feature",index=False)
joblib.dump(scalar, "scaler.pkl")
