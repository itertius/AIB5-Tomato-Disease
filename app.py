import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import os

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
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.load_state_dict(torch.load('Models/Week8/Week8v2.pth', map_location=device))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = [
    'Bacterial Spot - ใบเกิดจุดแบคทีเรีย',
    'Early Blight - โรคใบไหม้ระยะเริ่มต้น',
    'Late Blight - โรคใบไหม้ระยะสุดท้าย',
    'Leaf Mold - เชื้อราในใบไม้',
    'Septoria Leaf Spot - โรคใบจุดเซปโทเรีย',
    'Two Spotted Spider Mite - ไรสองจุด',
    'Target Spot - โรคใบจุดเป้ากระสุน',
    'Yellow Leaf Curl Virus - ไวรัสใบม้วนเหลืองในมะเขือเทศ',
    'Mosaic Virus - ไวรัสโมเสก',
    'Healthy - ใบสุขภาพดี'
]

example_images_dir = "Data/Sample"
example_dict = {}

for class_folder in os.listdir(example_images_dir):
    folder_path = os.path.join(example_images_dir, class_folder)
    if os.path.isdir(folder_path):
        images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            example_dict[class_folder] = images



st.title("🍅 AIB5-Tomato-Disease-Classification")

model = load_model()

mode = st.selectbox("เลือกโหมดการใช้งาน", ["อัพโหลดรูป", "เลือกรูปตัวอย่าง"])

image = None

if mode == "อัพโหลดรูป":
    uploaded_file = st.file_uploader("อัพโหลดภาพใบมะเขือเทศ", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif mode == "เลือกรูปตัวอย่าง":
    selected_class = st.selectbox("ตัวอย่างรูปภาพ", list(example_dict.keys()))
    
    image_paths = example_dict[selected_class]

    image_filenames = [os.path.basename(path) for path in image_paths]

    selected_filename = st.selectbox("เลือกรูปภาพ", image_filenames)

    selected_image_path = None
    for path in image_paths:
        if os.path.basename(path) == selected_filename:
            selected_image_path = path
            break

    if os.path.exists(selected_image_path):
        image = Image.open(selected_image_path).convert("RGB")
        uploaded_file = selected_image_path
    else:
        st.error("ไม่พบภาพตัวอย่างสำหรับคลาสนี้")


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("<h4 style='text-align: center;'>ภาพที่อัพโหลด</h4>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, use_container_width=True)

        with col2:

            model.eval()
            img_tensor = transform(image).unsqueeze(0).requires_grad_()

            def get_last_conv_layer(model):
                return model.features[-1]  # Last conv layer in MobileNetV2

            target_layer = get_last_conv_layer(model)
            gradients = []
            activations = []

            def backward_hook(module, grad_input, grad_output):
                gradients.append(grad_output[0])

            def forward_hook(module, input, output):
                activations.append(output)

            handle_fw = target_layer.register_forward_hook(forward_hook)
            handle_bw = target_layer.register_backward_hook(backward_hook)

            output = model(img_tensor)
            pred_class_idx = output.argmax(dim=1).item()

            model.zero_grad()
            class_score = output[0, pred_class_idx]
            class_score.backward()

            activ = activations[0].squeeze(0)
            grad = gradients[0].squeeze(0)

            weights = grad.mean(dim=(1, 2), keepdim=True)
            cam = (weights * activ).sum(dim=0)
            cam = torch.relu(cam).detach().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam = np.uint8(255 * cam)
            cam = Image.fromarray(cam).resize(image.size, resample=Image.BILINEAR)
            cam = np.array(cam)

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(image)
            ax.imshow(cam, cmap='jet', alpha=0.5)
            ax.axis('off')
            st.pyplot(fig)

            handle_fw.remove()
            handle_bw.remove()

    
    confidence_threshold = st.select_slider("ความมั่นใจของโมเดล", range(0, 101), value=50)

    if st.button('ทำนาย'):
        img_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        pred_idx = torch.argmax(probabilities).item()
        pred_class = class_names[pred_idx]
        confidence = probabilities[pred_idx].item()
        
        if confidence >= confidence_threshold / 100:
            st.success(f"Prediction: {pred_class} ({confidence:.2%})")
        else:
            st.warning("The image might not be a tomato leaf.")
            st.warning(f"Prediction: {pred_class} ({confidence:.2%})")

        
        st.write("All probabilities:")
        for i, prob in enumerate(probabilities):
            st.write(f"{class_names[i]}: {prob.item():.2%}")
        
