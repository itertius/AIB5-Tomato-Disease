import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np

class TomatoDiseaseModel(nn.Module):
    def __init__(self, num_classes=10):
        super(TomatoDiseaseModel, self).__init__()
        self.features = models.mobilenet_v2(pretrained=False).features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

@st.cache_resource
def load_model():
    model = TomatoDiseaseModel()
    model.load_state_dict(torch.load('Models/Week8/Week8v3.pth', map_location=torch.device('mps')))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['Bacterial Spot - ใบเกิดจุดแบคทีเรีย', 'Early Blight - โรคใบไหม้ระยะเริ่มต้น', 'Late Blight - โรคใบไหม้ระยะสุดท้าย',
               'Leaf Mold - เชื้อราในใบไม้', 'Septoria Leaf Spot - โรคใบจุดเซปโทเรีย', 'Two Spotted Spider Mite - ไรสองจุด',
               'Target Spot - โรคใบจุดเป้ากระสุน', 'Yellow Leaf Curl Virus - ไวรัสใบม้วนเหลืองในมะเขือเทศ',
               'Mosaic Virus - ไวรัสโมเสก', 'Healthy - ใบสุขภาพดี']

st.title("AIB5-Tomato-Disease-Classification 🍅")

model = load_model()

uploaded_file = st.file_uploader("Choose a tomato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)    

    st.markdown("<h4 style='text-align: center;'>Uploaded Image</h4>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, use_container_width=True)

    with col2:
        gray_image = image.convert("L")
        gray_array = np.array(gray_image)
        gray_array = (gray_array - gray_array.min()) / (gray_array.max() - gray_array.min() + 1e-8)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)
        ax.imshow(gray_array, cmap='Reds', alpha=0.5)
        ax.axis('off')
        # ax.set_title('Heatmap Overlay', fontsize=12)
        st.pyplot(fig)
    
    if st.button('Predict'):
        img_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        pred_idx = torch.argmax(probabilities).item()
        pred_class = class_names[pred_idx]
        confidence = probabilities[pred_idx].item()
        
        st.write(f"Prediction: {pred_class}")
        st.write(f"Confidence: {confidence:.2%}")
        
        st.write("All probabilities:")
        for i, prob in enumerate(probabilities):
            st.write(f"{class_names[i]}: {prob.item():.2%}")
        
