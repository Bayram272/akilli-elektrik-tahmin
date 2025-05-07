import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Yapay Zeka Tabanlı Elektrik Tüketimi Tahmini", layout="wide")

# Şık ve sade stil
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .main {
        padding: 20px;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #0c4b75;
    }
    .subtitle {
        font-size: 20px;
        color: #444;
    }
    .stButton > button {
        background-color: #0c4b75;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Başlık
st.markdown("<div class='title'>🚀 Elektrik Tüketimi Tahmin Uygulaması (LSTM)</div>", unsafe_allow_html=True)

# Dosya yükleme
st.markdown("---")
st.subheader("📁 Excel (.xlsx) Dosyası Yükleyin")
file = st.file_uploader("Excel dosyası yükleyin (Tarih ve Tüketim sütunlarıyla)", type="xlsx")

if file is not None:
    try:
        df = pd.read_excel(file)

        # Veriyi kontrol et
        if "Tarih" not in df.columns or "Tüketim" not in df.columns:
            st.error("Excel dosyanızda 'Tarih' ve 'Tüketim' adlı sütunlar bulunmalı.")
            st.stop()

        df["Tarih"] = pd.to_datetime(df["Tarih"])
        df = df.sort_values("Tarih")
        df.set_index("Tarih", inplace=True)

        st.line_chart(df["Tüketim"], height=300)

        # Normalizasyon
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[["Tüketim"]])

        time_step = 5
        if len(scaled_data) <= time_step:
            st.error(f"⚠️ Tahmin için en az {time_step + 1} satır veri gerekir. Lütfen daha uzun veri yükleyin.")
            st.stop()

        def create_dataset(data, time_step):
            X, y = [], []
            for i in range(len(data) - time_step):
                X.append(data[i:(i + time_step), 0])
                y.append(data[i + time_step, 0])
            return np.array(X), np.array(y)

        X, y = create_dataset(scaled_data, time_step)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, return_sequences=False, input_shape=(time_step, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=10, batch_size=16, verbose=0)

        # 7 gün tahmin
        input_seq = scaled_data[-time_step:].reshape(1, time_step, 1)
        predictions = []

        for _ in range(7):
            pred = model.predict(input_seq, verbose=0)[0][0]
            predictions.append(pred)
            input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

        future_predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=7)

        df_future = pd.DataFrame({"Tarih": future_dates, "Tahmin": future_predictions.flatten()})
        df_future.set_index("Tarih", inplace=True)

        st.markdown("---")
        st.subheader("🔍 Tahmin Grafiği")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index, df["Tüketim"], label="Geçmiş Verisi", color="#1f77b4")
        ax.plot(df_future.index, df_future["Tahmin"], label="Tahmin", color="#ff7f0e")
        ax.legend()
        ax.set_ylabel("Tüketim (kWh)")
        ax.set_xlabel("Tarih")
        st.pyplot(fig)

        st.success("🌟 Tahmin başarıyla tamamlandı ve görüntülendi.")

    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")
else:
    st.info("Lütfen .xlsx uzantılı bir dosya yükleyin. 'Tarih' ve 'Tüketim' sütunları zorunludur.")
