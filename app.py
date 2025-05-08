import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Elektrik Tahmini", layout="wide")

st.markdown("""
<style>
    .main {padding: 2rem;}
    .title {font-size: 32px; color: #0c4b75; font-weight: bold;}
    .stButton > button {
        background-color: #0c4b75;
        color: white;
        padding: 0.6em 1.2em;
        border-radius: 8px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🔌 LSTM Tabanlı Elektrik Tüketimi Tahmin Uygulaması</div>", unsafe_allow_html=True)

st.markdown("## 📁 Excel Dosyası Yükleyin (.xlsx) - 'Tarih' ve 'Tüketim' sütunları zorunludur")
file = st.file_uploader("Dosyanızı yükleyin", type="xlsx")

if file:
    try:
        df = pd.read_excel(file)

        # Sütun kontrolü
        if not {"Tarih", "Tüketim"}.issubset(df.columns):
            st.error("❌ Excel dosyanızda 'Tarih' ve 'Tüketim' adında iki sütun bulunmalı.")
            st.stop()

        df["Tarih"] = pd.to_datetime(df["Tarih"])
        df = df.sort_values("Tarih")
        df.set_index("Tarih", inplace=True)

        st.line_chart(df["Tüketim"], height=300)

        # Normalize et
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df[["Tüketim"]])

        time_step = 5
        if len(scaled) <= time_step:
            st.warning(f"⚠️ En az {time_step + 1} satırlık veri gerekir.")
            st.stop()

        def create_dataset(data, step):
            X, y = [], []
            for i in range(len(data) - step):
                X.append(data[i:i + step, 0])
                y.append(data[i + step, 0])
            return np.array(X), np.array(y)

        X, y = create_dataset(scaled, time_step)
        X = X.reshape((X.shape[0], time_step, 1))

        model = Sequential()
        model.add(LSTM(50, input_shape=(time_step, 1)))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=10, batch_size=16, verbose=0)

        input_seq = scaled[-time_step:].reshape(1, time_step, 1)
        predictions = []
        for _ in range(7):
            next_val = model.predict(input_seq, verbose=0)[0][0]
            predictions.append(next_val)
            input_seq = np.append(input_seq[:, 1:, :], [[[next_val]]], axis=1)

        future = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=7)
        df_pred = pd.DataFrame({"Tahmin": future.flatten()}, index=future_dates)

        st.markdown("## 📊 7 Günlük Tahmin Grafiği")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index, df["Tüketim"], label="Geçmiş", color="blue")
        ax.plot(df_pred.index, df_pred["Tahmin"], label="Tahmin", color="green")
        ax.legend()
        st.pyplot(fig)

        st.success("✅ Tahmin başarıyla yapıldı!")

    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")
else:
    st.info("⬆️ Lütfen yukarıdan bir .xlsx dosyası yükleyin.")