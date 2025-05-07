import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib

st.set_page_config(page_title="Elektrik Tüketimi Tahmini", layout="wide")

st.title("⚡ Yapay Zeka ile Elektrik Tüketimi Tahmini")
st.write("Bu uygulama, geçmiş verilere göre gelecekteki elektrik tüketimini LSTM modeli ile tahmin eder.")

uploaded_file = st.file_uploader("📁 Lütfen veri dosyanızı (CSV) yükleyin", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("🔍 Veri Önizlemesi")
    st.dataframe(df.head())

    st.write("Veri setiniz şu sütunlara sahip olmalı: `Tarih` ve `Tüketim`")

    # Tarihi datetime formatına çevir
    df['Tarih'] = pd.to_datetime(df['Tarih'])
    df = df.sort_values('Tarih')
    df.set_index('Tarih', inplace=True)

    # Anomali tespiti (z-score yöntemi)
    st.subheader("📊 Anomali Tespiti")
    df['Z-Score'] = (df['Tüketim'] - df['Tüketim'].mean()) / df['Tüketim'].std()
    df['Anomali'] = df['Z-Score'].abs() > 2.5
    st.write(f"Toplam {df['Anomali'].sum()} anomali tespit edildi.")

    # Grafik
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df['Tüketim'], label="Tüketim", color='blue')
    ax.scatter(df[df['Anomali']].index, df[df['Anomali']]['Tüketim'], color='red', label="Anomali")
    ax.legend()
    st.pyplot(fig)

    # Normalizasyon
    scaler = MinMaxScaler()
    df['Tüketim_norm'] = scaler.fit_transform(df[['Tüketim']])

    # LSTM için veri hazırlığı
    def create_sequences(data, step=30):
        X, y = [], []
        for i in range(step, len(data)):
            X.append(data[i-step:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    sequence_length = 30
    data = df['Tüketim_norm'].values
    X, y = create_sequences(data, sequence_length)

    # Model (burada örnek model eğitimi yapılıyor)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense

    st.subheader("🤖 Model Eğitimi")
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(sequence_length, 1)),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X.reshape(X.shape[0], X.shape[1], 1), y, epochs=10, batch_size=16, verbose=0)

    st.success("Model başarıyla eğitildi!")

    # Tahmin
    st.subheader("📈 Gelecek Tüketim Tahmini")
    last_sequence = data[-sequence_length:]
    predictions = []

    for _ in range(7):
        seq_input = last_sequence[-sequence_length:].reshape(1, sequence_length, 1)
        pred = model.predict(seq_input)[0][0]
        predictions.append(pred)
        last_sequence = np.append(last_sequence, pred)

    future_values = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=7)

    result_df = pd.DataFrame({
        'Tarih': future_dates,
        'Tahmin Tüketim': future_values
    })

    st.dataframe(result_df)

    # Grafik
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(df.index, df['Tüketim'], label="Geçmiş Tüketim", color='gray')
    ax2.plot(result_df['Tarih'], result_df['Tahmin Tüketim'], label="Tahmin", color='green')
    ax2.legend()
    st.pyplot(fig2)
