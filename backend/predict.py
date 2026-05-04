from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch 
import torch.nn as nn
import torch.nn.functional as F
import io
from torchvision import transforms

app = FastAPI()



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
model.load_state_dict(torch.load("C:\\irish potato detective\\model\\potato_1.pth"))
model.eval()

def process(image_bytes):
  image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
  transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
  image = transform(image).unsqueeze(0)
  return image


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
  image_bytes = await file.read()
  input_tensor = process(image_bytes)
  with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()
  classes = ['Healthy', 'earlyblt', 'lateblt']
  result = classes[prediction]
  return {'Diagnosis': result, 'confidence': torch.softmax(output, dim=1)[0][prediction].item(), 'status': 'success', }

