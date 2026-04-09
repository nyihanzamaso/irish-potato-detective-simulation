import streamlit as st
from PIL import Image
import torch 
import torchvision.transforms as transforms


st.title('irish potato leaf disease detection')
st.title('Simuating real-time U/P computing edge AI analysis')
img_file = st.file_uploader("upload an image of a potato leaf")

import torch.nn as nn
import torch.nn.functional as F

class MyCNN(nn.Module):
  def __init__(self, num_classes):
    super(MyCNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding= 1)
    self.pool = nn.MaxPool2d(kernel_size=2, stride= 2)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

    self.fc1 = nn.Linear(32*56*56, 128)
    self.fc2 = nn.Linear(128, num_classes)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x
  
model = MyCNN(num_classes=3)

if img_file:
    st.write("Processing image...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    img = Image.open(img_file).convert('RGB')
    img_t=transform(img)
    batch_t = torch.unsqueeze(img_t, 0)

    classes = ['Healthy', 'earlyblt', 'lateblt']
    model.load_state_dict(torch.load("C:\\irish potato detective\\venv\\model\\potato_1.pth"))
    st.write('model loaded successfully')


    with torch.no_grad():
        output = model(batch_t)
        _, predicted = torch.max(output, 1)
        st.write(f"predicted class: {classes[predicted]}")
        if classes[predicted] == 'Healthy':
           st.write('the potato leaf is healthy')
        elif classes[predicted] == 'earlyblt':
           st.write('the potato leaf is affected by early blight')
           st.write('early blight is a common fungal disease that affects potato plants, causing dark spots on the leaves. It can lead to reduced yield and quality of the potatoes if not managed properly.')
           st.write('To stop early blight, spray the plants with fungicides and pick off any sick leaves right away. Avoid getting water on the leaves by watering only the soil, and make sure the plants have enough fertilizer to stay strong.')
        else:
           st.write('the potato leaf is affected by late blight')
           st.write('late blight is a serious fungal disease that affects potato plants, causing dark lesions on the leaves, stems, and tubers. It can lead to significant crop loss if not managed properly.'
           )
           st.write('To stop late blight, you must act fast by spraying strong fungicides and destroying infected plants immediately by burning or burying them. If the disease is spreading quickly near harvest time, kill the green tops of the plants to protect the potatoes underground. You should also stop all watering to keep the leaves dry and check with local farming experts, as this disease can easily spread to neighboring fields.')
