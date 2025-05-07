import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Akıllı Elektrik Tüketimi Tahmini", layout="wide")

# Sayfa Başlığı ve Stil
st.markdown("""
    <style>
    body {
        background-color: #f5f5f5;
    }
    .main {
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    .title {
        color: #2c3e50;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🔌 Yapay Zeka Destekli Elektrik Tüketimi Tahmini</div>", unsafe_allow_html=True)

# Veri Yükleme
st.subheader("📁 CSV Verisi Yükleyin")
file = st.file_uploader("Tarih ve Tüketim içeren bir .csv dosyası yükleyin", type="csv")

if file is not None:
    df = pd.read_csv(file)
    df["Tarih"] = pd.to_datetime(df["Tarih"])
    df = df.sort_values("Tarih")
    df.set_index("Tarih", inplace=True)

    st.markdown("---")
    st.subheader("📊 Tüketim Verisi Grafiği")
    st.line_chart(df["Tüketim"])

    # Veri ölçeklendirme
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[["Tüketim"]])
if len(scaled_data) <= time_step:
    st.error(f"⚠️ Tahmin için en az {time_step + 1} günlük veri gerekir. Lütfen daha uzun bir veri dosyası yükleyin.")
    st.stop()


    # LSTM için veri hazırlama
    def create_dataset(data, time_step=5):
        X, y = [], []
        for i in range(len(data)-time_step):
            X.append(data[i:(i+time_step), 0])
            y.append(data[i+time_step, 0])
        return np.array(X), np.array(y)

    time_step = 5
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # LSTM Modeli
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)

    # Gelecek tahmin (7 gün)
    inputs = scaled_data[-time_step:].reshape(1, time_step, 1)
    predictions = []
    for _ in range(7):
        pred = model.predict(inputs, verbose=0)
        predictions.append(pred[0][0])
        inputs = np.append(inputs[:, 1:, :], [[pred]], axis=1)

    # Tahmini veriyi ters ölçeklendir
    future = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=7)
    df_future = pd.DataFrame({"Tarih": future_dates, "Tahmin": future.flatten()})
    df_future.set_index("Tarih", inplace=True)

    # Grafik
    st.markdown("---")
    st.subheader("🔍 7 Günlük Tahmin")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["Tüketim"], label="Geçmiş Verisi", color="blue")
    ax.plot(df_future.index, df_future["Tahmin"], label="Tahmin", color="green")
    ax.legend()
    ax.set_ylabel("kWh")
    ax.set_xlabel("Tarih")
    st.pyplot(fig)

else:
    st.info("📅 Tahmin için CSV dosyası yükleyiniz.")
