import os
import pickle
import streamlit as st  # type: ignore
from streamlit_option_menu import option_menu  # type: ignore


st.set_page_config(page_title='Disease Prediction System',
                   layout='wide', page_icon="🏥")


try:
    diabetes_model = pickle.load(open(r"MODELS/diabetes_model.sav", 'rb'))
    heart_disease_model = pickle.load(open(r"MODELS/heart_model.sav", 'rb'))
    parkinsons_model = pickle.load(open(r"MODELS/parkinsons_model.sav", 'rb'))
    alzheimers_model = pickle.load(open(r"MODELS/alzheimers_model.sav", 'rb'))
    breastcancer_model = pickle.load(open(r"MODELS/breastcancer_model.sav", 'rb'))
except Exception as e:

    st.error(f"Error loading models: {e}")
    st.stop()

with st.sidebar:
    selected = option_menu("Disease Prediction System",
                           ["Diabetes Prediction", "Heart Disease Prediction",
                               "Parkinson's Prediction", "Alzheimer's Prediction","Breast Cancer Prediction"],
                           menu_icon="hospital-fill", icons=["activity", "heart", "person", "file-medical","capsule"], default_index=0)


Pregnancies, Glucose, Bloodpressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age = None, None, None, None, None, None, None, None
Age_heart, Sex, ChestPainType, RestingBloodPressure, Cholesterol, FastingBloodSugar, RestECG, MaxHeartRate, ExerciseInducedAngina, StDepression, SlopeOfStSegement, vessels, Thalassemia = None, None, None, None, None, None, None, None, None, None, None, None, None
mdvpfo, mdvpfhi, mdvpflo, mdvpjitter, mdvpjitterabs, mdvprap, mdvpppq, jitterddp, mdvpshimmer, mdvpshimmerdb, shimmerapq3, shimmerapq5, mdvpapq, shimmerdda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe = None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
Age, Gender, CognitiveTestScore, BrainScanResult, FamilyHistory, Outcome = None, None, None, None, None, None
Age,RadiusMean,TextureMean,PerimeterMean,AreaMean,SmoothnessMean,CompactnessMean,ConcavityMean,ConcavePointsMean,SymmetryMean,FractalDimensionMean,RadiusSE,TextureSE,PerimeterSE,AreaSE,SmoothnessSE,CompactnessSE,ConcavitySE,ConcavePointsSE,SymmetrySE,FractalDimensionSE,RadiusWorst,TextureWorst,PerimeterWorst,AreaWorst,SmoothnessWorst,CompactnessWorst,ConcavityWorst,ConcavePointsWorst,SymmetryWorst,FractalDimensionWorst = None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None


if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using Machine Learning')
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        Bloodpressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input(
            'Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the person')

    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        try:
            user_input = [Pregnancies, Glucose, Bloodpressure,
                          SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            user_input = [float(x) for x in user_input]
            diab_prediction = diabetes_model.predict([user_input])
            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'The person is not diabetic'
            st.success(diab_diagnosis)
        except Exception as e:
            st.error(f"Error during prediction: {e}")


elif selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using Machine Learning')
    col1, col2, col3 = st.columns(3)

    with col1:
        Age_heart = st.text_input('Age of the person')
    with col2:
        Sex = st.text_input('Sex of the person (0 - Female/1 - Male)')
    with col3:
        ChestPainType = st.text_input('Chest pain Type')
    with col1:
        RestingBloodPressure = st.text_input('Resting blood pressure')
    with col2:
        Cholesterol = st.text_input('Cholesterol')
    with col3:
        FastingBloodSugar = st.text_input('Fasting blood sugar')
    with col1:
        RestECG = st.text_input('Resting ECG')
    with col2:
        MaxHeartRate = st.text_input('Maximum Heart rate')
    with col3:
        ExerciseInducedAngina = st.text_input(
            'Exercise Induced Angina of person')
    with col1:
        StDepression = st.text_input('ST Depression of person')
    with col2:
        SlopeOfStSegement = st.text_input('Slope of ST segment of person')
    with col3:
        vessels = st.text_input('Number of major vessels')
    with col1:
        Thalassemia = st.text_input('Thalassemia of person')

    heart_diagnosis = ''
    if st.button('Heart Test Result'):
        try:
            user_input = [Age_heart, Sex, ChestPainType, RestingBloodPressure, Cholesterol, FastingBloodSugar,
                          RestECG, MaxHeartRate, ExerciseInducedAngina, StDepression, SlopeOfStSegement, vessels, Thalassemia]
            user_input = [float(x) for x in user_input]
            heart_prediction = heart_disease_model.predict([user_input])
            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person is attacked by Heart Disease'
            else:
                heart_diagnosis = 'The person is not attacked by Heart Disease'
            st.success(heart_diagnosis)
        except Exception as e:
            st.error(f"Error during prediction: {e}")


elif selected == "Parkinson's Prediction":
    st.title('Parkinson Disease Prediction using Machine Learning')
    col1, col2, col3 = st.columns(3)
    with col1:
        mdvpfo = st.text_input('MDVP:Fo(HZ)')
    with col2:
        mdvpfhi = st.text_input('MDVP:Fhi(HZ)')
    with col3:
        mdvpflo = st.text_input('MDVP:Flo(HZ)')
    with col1:
        mdvpjitter = st.text_input('MDVP:Jitter(%)')
    with col2:
        mdvpjitterabs = st.text_input('MDVP:Jitter(Abs)')
    with col3:
        mdvprap = st.text_input('MDVP:RAP')
    with col1:
        mdvpppq = st.text_input('MDVP:PPQ')
    with col2:
        jitterddp = st.text_input('Jitter:DDP')
    with col3:
        mdvpshimmer = st.text_input('MDVP:Shimmer')
    with col1:
        mdvpshimmerdb = st.text_input('MDVP:Shimmer(db)')
    with col2:
        shimmerapq3 = st.text_input('Shimmer:APQ3')
    with col3:
        shimmerapq5 = st.text_input('Shimmer:APQ5')
    with col1:
        mdvpapq = st.text_input('MDVP:APQ')
    with col2:
        shimmerdda = st.text_input('Shimmer: DDA')
    with col3:
        nhr = st.text_input('NHR')
    with col1:
        hnr = st.text_input('HNR')
    with col2:
        rpde = st.text_input('RPDE')
    with col3:
        dfa = st.text_input('DFA')
    with col1:
        spread1 = st.text_input('Spread1')
    with col2:
        spread2 = st.text_input('Spread2')
    with col3:
        d2 = st.text_input('D2')
    with col1:
        ppe = st.text_input('PPE')

    parkinson_diagnosis = ''
    if st.button('Parkinson Test Result'):
        try:
            user_input = [mdvpfo, mdvpfhi, mdvpflo, mdvpjitter, mdvpjitterabs, mdvprap, mdvpppq, jitterddp, mdvpshimmer,
                          mdvpshimmerdb, shimmerapq3, shimmerapq5, mdvpapq, shimmerdda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]
            user_input = [float(x) for x in user_input]
            parkinson_prediction = parkinsons_model.predict([user_input])
            if parkinson_prediction[0] == 1:
                parkinson_diagnosis = 'The person is attacked by Parkinson Disease'
            else:
                parkinson_diagnosis = 'The person is not attacked by Parkinson Disease'
            st.success(parkinson_diagnosis)
        except Exception as e:
            st.error(f"Error during prediction: {e}")

elif selected == "Alzheimer's Prediction":
    st.title("Alzheimer's Disease Prediction using Machine Learning")
    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.text_input('Patient Age')
    with col2:
        Gender = st.text_input('Patient gender')
    with col3:
        CognitiveTestScore = st.text_input('Cognitive Test Score')
    with col1:
        BrainScanResult = st.text_input('Brain Scan Result')
    with col2:
        FamilyHistory = st.text_input('Family History')

    alzheimers_diagnosis = ''
    if st.button("Alzheimer's Test Result"):
        try:
            user_input = [Age, Gender, CognitiveTestScore,
                          BrainScanResult, FamilyHistory]
            user_input = [float(x) for x in user_input]
            alzheimers_prediction = alzheimers_model.predict([user_input])
            if alzheimers_prediction[0] == 1:
                alzheimers_diagnosis = "The person is attacked by alzheimer's Disease"
            else:
                alzheimers_diagnosis = "The person is not attacked by alzheimer's Disease"
            st.success(alzheimers_diagnosis)
        except Exception as e:
            st.error(f"Error during prediction: {e}")

elif selected == "Breast Cancer Prediction":
    st.title("Breast Cancer Prediction using Machine Learning")
    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.text_input('Patient Age')
    with col2:
        RadiusMean = st.text_input('Radius Mean')
    with col3:
        TextureMean = st.text_input('Texture Mean')
    with col1:
        PerimeterMean = st.text_input('Perimeter Mean')
    with col2:
        AreaMean = st.text_input('Area Mean')
    with col3:
        SmoothnessMean = st.text_input('Smoothness Mean')
    with col1:
        CompactnessMean = st.text_input('Compactness Mean')
    with col2:
        ConcavityMean = st.text_input('Concavity Mean')
    with col3:
        ConcavePointsMean = st.text_input('Concave Points Mean')
    with col1:
        SymmetryMean = st.text_input('Symmetry Mean')
    with col2:
        FractalDimensionMean = st.text_input('Fractal Dimension Mean')
    with col3:
        RadiusSE = st.text_input('Radius Standard Error')
    with col1:
        TextureSE = st.text_input('Texture Standard Error')
    with col2:
        PerimeterSE = st.text_input('Perimeter Standard Error')
    with col3:
        AreaSE = st.text_input('Area Standard Error')
    with col1:
        SmoothnessSE = st.text_input('Smoothness Standard Error')
    with col2:
        CompactnessSE = st.text_input('Compactness Standard')
    with col3:
        ConcavitySE = st.text_input('Concavity Standard')
    with col1:
        ConcavePointsSE = st.text_input('Concave Points Standard')
    with col2:
        SymmetrySE = st.text_input('Symmetry Standard')
    with col3:
        FractalDimensionSE = st.text_input('Fractal Dimension Standard')
    with col1:
        RadiusWorst = st.text_input('Radius Worst')
    with col2:
        TextureWorst = st.text_input('Texture Worst')
    with col3:
        PerimeterWorst = st.text_input('Perimeter Worst')
    with col1:
        AreaWorst = st.text_input('Area Worst')
    with col2:
        SmoothnessWorst = st.text_input('Smoothness Worst')
    with col3:
        CompactnessWorst = st.text_input('Compact Duration Worst')
    with col1:
        ConcavityWorst = st.text_input('Concavity Worst')
    with col2:
        ConcavePointsWorst = st.text_input('Concave Points Worst')
    with col3:
        SymmetryWorst = st.text_input('Symmetry Worst')
    with col1:
        FractalDimensionWorst = st.text_input('Fractal Dimension Worst')

    breastcancer_diagnosis = ''
    if st.button("Breast Cancer Test Result"):
        try:
            user_input = [Age,RadiusMean,TextureMean,PerimeterMean,AreaMean,SmoothnessMean,CompactnessMean,ConcavityMean,ConcavePointsMean,SymmetryMean,FractalDimensionMean,RadiusSE,TextureSE,PerimeterSE,AreaSE,SmoothnessSE,CompactnessSE,ConcavitySE,ConcavePointsSE,SymmetrySE,FractalDimensionSE,RadiusWorst,TextureWorst,PerimeterWorst,AreaWorst,SmoothnessWorst,CompactnessWorst,ConcavityWorst,ConcavePointsWorst,SymmetryWorst,FractalDimensionWorst]
            user_input = [float(x) for x in user_input]
            breastcancer_prediction = breastcancer_model.predict([user_input])
            if breastcancer_prediction[0] == 1:
                breastcancer_diagnosis = "The person is attacked by Breast Cancer Disease"
            else:
                breastcancer_diagnosis = "The person is not attacked by Breast Cancer Disease"
            st.success(breastcancer_diagnosis)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
