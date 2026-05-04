import streamlit as st
import requests




st.title('irish potato leaf disease detection')
st.title('Simuating real-time U/P computing edge AI analysis')
img_file = st.file_uploader("upload an image of a potato leaf") 
if img_file is not None:
    files = {'file': img_file.getvalue()}
    response =requests.post('http://127.0.0.1:8000/predict', files = files)
    if response.status_code == 200:
        result = response.json()
        st.success(f'Results: {result['Diagnosis']}   Confidence: {result['confidence']}')

        st.write(f'Recommendation measures for {result["Diagnosis"]}')
        if result["Diagnosis"] == 'Healthy':
            st.write('the potato leaf is healthy')
        elif result["Diagnosis"] == 'earlyblt':
            st.write('early blight is a common fungal disease that affects potato plants, causing dark spots on the leaves. It can lead to reduced yield and quality of the potatoes if not managed properly.')
            st.write('To stop early blight, spray the plants with fungicides and pick off any sick leaves right away. Avoid getting water on the leaves by watering only the soil, and make sure the plants have enough fertilizer to stay strong.')
        else:
            st.write('late blight is a serious fungal disease that affects potato plants, causing dark lesions on the leaves, stems, and tubers. It can lead to significant crop loss if not managed properly.'
            )
            st.write('To stop late blight, you must act fast by spraying strong fungicides and destroying infected plants immediately by burning or burying them. If the disease is spreading quickly near harvest time, kill the green tops of the plants to protect the potatoes underground. You should also stop all watering to keep the leaves dry and check with local farming experts, as this disease can easily spread to neighboring fields.')
        
    else:
        st.error('Failed to connect to the prediction server.')

        