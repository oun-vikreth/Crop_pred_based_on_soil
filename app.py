import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

df_soil = pd.read_csv('CleanBy_NormalRange(Soil Fertility).csv')
df_prod = pd.read_csv('apy_new3.csv')

X3_test, X3_train, y_test, y3_train = train_test_split(df_soil.drop(['DistrictId', 'DistrictName', 'BlockId', 'BlockName', 'SampleNo'], axis=1), df_soil['DistrictName'], test_size=0.2, random_state=42)


# Load the trained model from the pickle file
with open('xgb_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# Function to predict DistrictName based on soil data input
def predict_district(rf_model, label_encoder, soil_data):
    prediction = rf_model.predict(soil_data)
    predicted_district = label_encoder.inverse_transform(prediction)
    return predicted_district[0]

# Load the label encoder used for encoding DistrictName
label_encoder = LabelEncoder()
# Assuming df_soil is the original DataFrame used for training the model
label_encoder.fit(df_soil['DistrictName'])

st.title('Crops yield prediction based on soil')

st.write('Adjust Soil Data for Prediction:')
# Adjusted input sliders for soil data based on provided min and max values
soil_ph = st.slider('Soil pH', min_value=0.0, max_value=14.0, step=0.01)
electrical_conductivity = st.slider('Electrical Conductivity', min_value=0.0, max_value=10.0, step=0.01)
organic_carbon = st.slider('Organic Carbon', min_value=0.0, max_value=100.0, step=0.1)
nitrogen = st.slider('Nitrogen', min_value=0.0, max_value=2000.0, step=1.0)
phosphorous = st.slider('Phosphorous', min_value=0.0, max_value=2000.0, step=1.0)
potassium = st.slider('Potassium', min_value=0.0, max_value=5000.0, step=1.0)
sulphur = st.slider('Sulphur', min_value=0.0, max_value=2000.0, step=0.1)
zinc = st.slider('Zinc', min_value=0.0, max_value=2000.0, step=1.0)
iron = st.slider('Iron', min_value=0.0, max_value=200.0, step=0.01)
magnesium = st.slider('Magnesium', min_value=0.0, max_value=200.0, step=0.01)

# Make prediction based on user input
if st.button('Predict'):
    user_input = [[soil_ph, electrical_conductivity, organic_carbon, nitrogen,
                   phosphorous, potassium, sulphur, zinc, iron, magnesium]]
    new_soil_df = pd.DataFrame(user_input, columns=X3_train.columns)
    predicted_district = predict_district(rf_model, label_encoder, new_soil_df)

    st.write('Top 5 Crops and Production per Area according to your Soil Fertility:')
    top_5_crops = df_prod[df_prod['District_Name'] == predicted_district].groupby('Crop')['ProdPerArea'].mean().sort_values(ascending=False).head(5).to_frame()
    st.write(top_5_crops)

    st.write('Historical Production of Top 5 Crops in soil similar to yours:')
    plt.figure(figsize=(20, 10))
    line_plot = sns.lineplot(x='Crop_Year', y='Production', hue='Crop', data=df_prod[df_prod['District_Name'] == predicted_district])
    plt.xticks(rotation=90)
    st.pyplot(line_plot.figure)
