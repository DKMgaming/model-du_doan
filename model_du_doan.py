import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import folium
from streamlit_folium import folium_static

# Đọc dữ liệu từ file Excel
def load_data_from_excel(file):
    data = pd.read_excel(file)
    X = data[["Kinh_do_tram_thu", "Vi_do_tram_thu", "Do_cao_anten_thu", 
              "Muc_tin_hieu", "Azimuth", "Chat_luong_phep_do", "Chieu_cao_anten_tram_phat", "Tan_so"]].values
    y = data[["Kinh_do_tram_phat", "Vi_do_tram_phat"]].values
    return X, y

# Huấn luyện hoặc tải mô hình và scaler
def train_or_load_model(file):
    model_path = 'model_dnn_vo_tuyen_dien.h5'
    scaler_path = 'scaler_params.npy'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        scaler_params = np.load(scaler_path, allow_pickle=True).item()
        scaler = StandardScaler()
        scaler.mean_ = scaler_params['mean']
        scaler.scale_ = scaler_params['scale']
    else:
        X, y = load_data_from_excel(file)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        scaler_params = {'mean': scaler.mean_, 'scale': scaler.scale_}
        np.save(scaler_path, scaler_params)
        
        model = Sequential([
            Dense(64, input_shape=(8,), activation='relu'),  # input_shape cập nhật với 8 đặc trưng
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(2)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, callbacks=[early_stopping], verbose=1)
        model.save(model_path)
    
    return model, scaler

# Hàm dự đoán tọa độ
def predict_coordinates(model, scaler, X):
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    return y_pred

# Hàm hiển thị bản đồ
def display_map(predictions, actuals=None):
    map_center = [np.mean(predictions[:, 1]), np.mean(predictions[:, 0])]
    m = folium.Map(location=map_center, zoom_start=5)

    for pred in predictions:
        folium.Marker([pred[1], pred[0]], icon=folium.Icon(color="blue"), popup="Predicted").add_to(m)

    if actuals is not None:
        for actual in actuals:
            folium.Marker([actual[1], actual[0]], icon=folium.Icon(color="red"), popup="Actual").add_to(m)
    
    folium_static(m)

# Streamlit App
st.title("Dự đoán tọa độ trạm phát")

uploaded_file = st.file_uploader("Tải lên file dữ liệu huấn luyện hoặc dự đoán (Excel)", type="xlsx")

if uploaded_file is not None:
    X, y = load_data_from_excel(uploaded_file)
    
    if st.button("Huấn luyện mô hình"):
        model, scaler = train_or_load_model(uploaded_file)
        st.success("Huấn luyện mô hình thành công!")
        
    if st.button("Dự đoán"):
        model, scaler = train_or_load_model(uploaded_file)
        predictions = predict_coordinates(model, scaler, X)
        
        # Xuất kết quả ra file Excel
        pred_df = pd.DataFrame(predictions, columns=["Kinh_do_tram_phat", "Vi_do_tram_phat"])
        pred_df.to_excel("du_doan_toa_do.xlsx", index=False)
        st.success("Dự đoán thành công! Kết quả đã được lưu vào file du_doan_toa_do.xlsx")
        
        # Hiển thị bản đồ
        display_map(predictions, actuals=y)
