import streamlit as st 
import pandas as pd 
import pickle
import numpy as np

# Memuat model
with open('perceptron_fruit.pkl', 'rb') as f:
    model_perceptron = pickle.load(f)

with open('svm_fruit.pkl', 'rb') as f:
   model_svm = pickle.load(f)

with open('random_forest_fruit.pkl', 'rb') as f:
    model_random_forest = pickle.load(f)


st.title("Prediksi Buah") 
st.markdown("Prediksi/Label adalah Output utama dari model adalah prediksi untuk variabel dependen berdasarkan variabel independen yang diberikan.") 
st.sidebar.title("Inputkan data Anda di sini")  

# Pilih model
model_choice = st.sidebar.selectbox('Pilih Model untuk Prediksi:', 
                                    ('Perceptron', 'SVM', 'random forest')) 

# Inisialisasi atau reset hasil jika model berubah
if 'selected_model' not in st.session_state:
    st.session_state['selected_model'] = model_choice
    st.session_state['results'] = []
elif st.session_state['selected_model'] != model_choice:
    st.session_state['selected_model'] = model_choice
    st.session_state['results'] = []  # Hapus hasil jika model berubah

diameter = st.sidebar.slider("Ukuran buah(diameter):", 0, 100, 0)
weight = st.sidebar.number_input('Berat buah(weight):', min_value=0.0)
red = st.sidebar.number_input('Intensitas warna merah dalam buah (red):', min_value=0.0)
green = st.sidebar.number_input('Intensitas warna hijau dalam buah (green):', min_value=0.0)
blue = st.sidebar.number_input('Intensitas warna biru dalam buah  (blue):', min_value=0.0)

# Tombol untuk memprediksi spesies ikan
if st.sidebar.button('Prediksi Buah'):
    features = np.array([[diameter, weight, red, green, blue]])
    
    # Memilih model berdasarkan pilihan pengguna
    if model_choice == 'Perceptron':
        model = model_perceptron
        with open('label_encoder_fruit_Perseptron.pkl', 'rb') as encoder_file:
            encoder = pickle.load(encoder_file)
    elif model_choice == 'SVM':
        model = model_svm
        with open('label_encoder_fruit_SVM.pkl', 'rb') as encoder_file:
            encoder = pickle.load(encoder_file)
    else:
        model = model_random_forest
        with open('label_encoder_fruit_forest.pkl', 'rb') as encoder_file:
            encoder = pickle.load(encoder_file)

    
    predicted_species_encoded = model.predict(features)[0]
    
    # Dekode hasil prediksi menggunakan encoder
    predicted_species = encoder.inverse_transform([predicted_species_encoded])[0]
    
    # Menyimpan hasil ke dalam session_state
    st.session_state['results'].append({
        'diameter': diameter,
        'Weight': weight,
        'red': red,
        'green': green,
        'blue': blue,
        'Model': model_choice,
        'Predicted Buah': predicted_species
    })

# Menampilkan semua hasil prediksi dalam tabel
if st.session_state['results']:
    result_df = pd.DataFrame(st.session_state['results'])
    st.subheader('Tabel Hasil Prediksi')
    st.dataframe(result_df)
