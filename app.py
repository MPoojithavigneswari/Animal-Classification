import os
import uuid
import urllib
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model class
class CustomMobileNetV2(nn.Module):
    def __init__(self, base_model, num_classes=10):
        super(CustomMobileNetV2, self).__init__()
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(base_model.last_channel, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Loading the MobileNetV2 model pre-trained on ImageNet
base_model = models.mobilenet_v2(pretrained=True)

# Freeze the parameters of the base model
for param in base_model.parameters():
    param.requires_grad = False

num_classes = 10
model = CustomMobileNetV2(base_model,num_classes)
model.load_state_dict(torch.load('animal_classification_model.pth', map_location=device))
model.to(device)
model.eval()

app = Flask(__name__)


class_names = ['dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel']
# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def allowed_file(filename):
    allowed_extensions = {'jpg', 'jpeg', 'png', 'jfif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def predict(filename, model):
    image = Image.open(filename).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)[0]
        top_probs, top_indices = torch.topk(probabilities, 3)
        
        top_classes = [class_names[i] for i in top_indices.cpu().numpy()]
        top_probs = top_probs.cpu().numpy()
    
    return top_classes, top_probs

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    error = ''
    target_img = os.path.join(os.getcwd(), 'static/uploaded_images')
    
    # If it is in a link
    if request.method == 'POST':
        if(request.form):
            link = request.form.get('link')
            try:
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename + ".jpg"
                img_path = os.path.join(target_img, filename)
                output = open(img_path , "wb")
                output.write(resource.read())
                output.close()
                img = filename

                class_result, prob_result = predict(img_path, model)

                predictions = {
                    "class1": class_result[0],
                    "class2": class_result[1],
                    "class3": class_result[2],
                    "prob1": (prob_result[0]*100).round(2),
                    "prob2": (prob_result[1]*100).round(2),
                    "prob3": (prob_result[2]*100).round(2),
                }

            except Exception as e:
                print(str(e))
                error = 'Unable to access image from the provided link'

            if not error:
                return render_template('prediction.html', img=img, predictions=predictions)
            else:
                return render_template('index.html', error=error)

        # If it is a file
        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img, file.filename))
                img_path = os.path.join(target_img, file.filename)
                img = file.filename

                class_result, prob_result = predict(img_path, model)

                predictions = {
                    "class1": class_result[0],
                    "class2": class_result[1],
                    "class3": class_result[2],
                    "prob1": (prob_result[0]*100).round(2),
                    "prob2": (prob_result[1]*100).round(2),
                    "prob3": (prob_result[2]*100).round(2),
                }
            else:
                error = "Please upload images of jpg, jpeg, png, or jfif extension only"

            if(len(error) == 0):
                return render_template('prediction.html', img=img, predictions=predictions)
            else:
                return render_template('index.html', error=error)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
