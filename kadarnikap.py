import streamlit as st
import pandas as pd
import joblib
import time

# Konfigurasi halaman
st.set_page_config(page_title='ML Portfolio', page_icon='📊', layout='wide')
st.write('Welcome to the ML Portfolio App!')

# Opsi sidebar
select_var = st.sidebar.selectbox("Want to open about?", ("Home", "Estimasi Kadar Ni"))

# Batas input berdasarkan data pelatihan
X_MIN, X_MAX = 9786009, 9786129
Y_MIN, Y_MAX = 307438, 307547
DEPTH_MIN, DEPTH_MAX = 0, 186.0

def estimasini():
    st.title("Estimasi Kadar Nikel (Total Ni)")
    st.write("Aplikasi ini memprediksi kadar Ni berdasarkan data spasial lubang bor di area eksplorasi X.")

    st.sidebar.header('User Input Features:')

    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            st.sidebar.header('Input Manual')
            koor_x = st.sidebar.number_input('Koordinat X', min_value=9780000, max_value=9800000, value=9786050)
            koor_y = st.sidebar.number_input('Koordinat Y', min_value=307000, max_value=308000, value=307500)
            depth = st.sidebar.number_input('Koordinat Z (mdpl)', min_value=0.0, max_value=200.0, value=160.0)
            data = {'koor_x': koor_x, 'koor_y': koor_y, 'Depth': depth}
            return pd.DataFrame(data, index=[0])
        input_df = user_input_features()

    st.image("https://www.antam.com/uploads/ngc_global_images/5dcbd0025fced_20191113164226-2.jpg", width=500, caption='Ilustrasi Nikel')

    button_var = st.sidebar.button('Prediksi!')

    if button_var:
        df = input_df
        st.write("Data Input:")
        st.write(df)

        # Validasi jangkauan input
        in_range = (
            X_MIN <= df['koor_x'][0] <= X_MAX and
            Y_MIN <= df['koor_y'][0] <= Y_MAX and
            DEPTH_MIN <= df['Depth'][0] <= DEPTH_MAX
        )

        with st.spinner('Tunggu sebentar...'):
            time.sleep(2)
            try:
                model = joblib.load("best_model_random_forest.pkl")
                prediction = model.predict(df)[0]
                if in_range:
                    st.success(f"✅ Perkiraan kadar Total Ni: {prediction:.4f}")
                else:
                    st.warning(f"⚠️ Hasil prediksi: {prediction:.4f}, namun nilai input berada di luar jangkauan data pelatihan. Akurasi bisa menurun.")
            except Exception as e:
                st.error(f"❌ Gagal memuat model: {e}")

if select_var == "Estimasi Kadar Ni":
    estimasini()