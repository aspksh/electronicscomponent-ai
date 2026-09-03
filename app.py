import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# same model architecture
class ComponentCNN(nn.Module):
    def __init__(self, num_classes):
        super(ComponentCNN, self).__init__()
        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64*16*16,128)
        self.fc2 = nn.Linear(128,num_classes)

    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# load model
device = torch.device("cpu")
model = ComponentCNN(num_classes=15)
model.load_state_dict(torch.load("electronics_cnn.pth", map_location=device))
model.eval()

# classes (same order as training)
classes = ['Buzzer','Capasitor','Diode','IC','Induktor','LDR','LED',
          'Potensiometer','Relay','Resistor','Saklar','Sensor IR',
          'Sensor LM35','Sensor Ultrasonik','Transistor']

transform = transforms.Compose([
  transforms.Resize((128,128)),
  transforms.ToTensor()
])

st.title(" Electronics Component Classifier")
st.write("It can classify Buzzer, Capacitor, Diode, IC, Inductor, LDR, LED,"
          "Potentiometer,Relay, Resistor, Switch, Sensor LM35,"
          "Sensor Ultrasonik,")

uploaded_files = st.file_uploader("Upload Image", type=["jpg","png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:                                
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image")
    
        img = transform(img).unsqueeze(0)
    
        with torch.no_grad():
          output = model(img)
          probs = torch.softmax(output, dim=1)  
          _, pred = torch.max(output, 1)
        
          confidence = probs[0][pred.item()].item()*100  
        st.success(f"{uploaded_file.name} → {classes[pred.item()]} ({confidence:.2f}%)")
