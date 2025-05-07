import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Akıllı Elektrik Tüketimi Tahmini", layout="wide")

# Şık stil
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
    }
    .title {
        color: #1f77b4;
        font-size: 36px;
        font-weight: bold;
    }
    .subtitle {
        font-size: 20px;
        color: #333;
        margin-bottom: 10px;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🌌 Yapay Zeka Destekli Elektrik Tüketimi Tahmini</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div class='subtitle'>📁 Excel (.xlsx) Dosyası Yükleyin</div>", unsafe_allow_html=True)
file = st.file_uploader("Excel dosyası (.xlsx) yükleyin (Tarih ve Tüketim sütunlarıyla)", type=["xlsx"])

if file is not None:
    try:
        df = pd.read_excel(file)
        df["Tarih"] = pd.to_datetime(df["Tarih"])
        df = df.sort_values("Tarih")
        df.set_index("Tarih", inplace=True)

        st.markdown("---")
        st.subheader("📊 Gerçek Tüketim Grafiği")
        st.line_chart(df["Tüketim"])

        # LSTM model için hazırlık
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df[["Tüketim"]])

        time_step = 5
        if len(scaled_data) <= time_step:
            st.error(f"⚠️ Tahmin için en az {time_step + 1} günlük veri gerekir. Daha uzun veri yükleyin.")
            st.stop()

        # X, y oluştur
        def create_dataset(data, time_step=1):
            X, y = [], []
            for i in range(len(data) - time_step):
                X.append(data[i:(i + time_step), 0])
                y.append(data[i + time_step, 0])
            return np.array(X), np.array(y)

        X, y = create_dataset(scaled_data, time_step)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=10, batch_size=16, verbose=0)

        # Tahmin
        inputs = scaled_data[-time_step:].reshape(1, time_step, 1)
        predictions = []
        for _ in range(7):
            pred = model.predict(inputs, verbose=0)
            predictions.append(pred[0][0])
            inputs = np.append(inputs[:, 1:, :], [[pred]], axis=1)

        future = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=7)
        df_future = pd.DataFrame({"Tarih": future_dates, "Tahmin": future.flatten()})
        df_future.set_index("Tarih", inplace=True)

        st.markdown("---")
        st.subheader("🔍 7 Günlük LSTM Tahmini")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df["Tüketim"], label="Gerçek", color="#1f77b4")
        ax.plot(df_future.index, df_future["Tahmin"], label="Tahmin", color="#2ca02c")
        ax.legend()
        ax.set_xlabel("Tarih")
        ax.set_ylabel("Tüketim (kWh)")
        st.pyplot(fig)

        st.success("✅ Tahmin başarıyla tamamlandı.")

    except Exception as e:
        st.error(f"Hata oluştu: {str(e)}")
else:
    st.info("Lütfen bir Excel dosyası yükleyiniz.")
