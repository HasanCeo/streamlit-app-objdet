import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Başlık
st.title("YOLO Nesne Tespiti Uygulaması")

# Modeli yükle
@st.cache_resource  # Model yüklemesini önbelleğe alır
def load_model():
    return YOLO("yolo11s.pt")

model = load_model()

# Kullanıcıdan PNG dosyası yüklemesini iste
uploaded_file = st.file_uploader("Bir PNG dosyası yükleyin", type=["png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Yüklenen Görüntü', use_column_width=True)
    
    # Nesne tespiti yap
    st.write("### Nesne Tespiti Yapılıyor...")
    results = model.predict(image)
    
    # Tespit edilen nesneleri göster
    result_image = results[0].plot()
    st.image(result_image, caption='Tespit Sonucu', use_column_width=True)
