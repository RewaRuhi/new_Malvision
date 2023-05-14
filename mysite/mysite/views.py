from django.http import HttpResponse
from django.shortcuts import render
import cv2
import numpy as np
from PIL import Image
# from .services import process_image
from django.templatetags.static import static
from tensorflow.keras.models import load_model
import os


def homePage(request):
   # data={
    # 'title':'my home page',
    # 'bdata': 'welcome Rewa Ruhi at mysite',
    # 'clist':['php','java','python'],
    # 'numbers':[10,20,30,40,50],
    # 'student_details':[
    #  {'name': 'rewa','phone':9905528922},
    #   {'name': 'ruhi','phone':79005528922}
    # ]
    # }
    return render(request, "malaria-home.html")
    # return render(request,"after-predict.html")


def aboutUs(request):
    return render(request, "after-predict.html")


def about(request):
    return render(request, "about.html")


def contact(request):
    return render(request, "contact.html")


def predict(request):

    # def process_image:
    if request.method == "POST":
        sample_image = request.FILES["sample_image"]
        print(sample_image)
        # img= cv2.imread(sample_image)
        img = cv2.imdecode(np.frombuffer(
            sample_image.read(), np.uint8), cv2.IMREAD_COLOR)

        # Resize the image to (64, 64) using OpenCV
        img = cv2.resize(np.array(img), (64, 64))

        # Convert the image to array format
        test_image = np.asarray(img)

        # Expand the dimensions of the image to match the expected input shape
        test_image = np.expand_dims(test_image, axis=0)

        # Normalize the pixel values
        test_image = test_image / 255.0

        model_path = static('model/malaria.h5')
        model = load_model(model_path.lstrip('/'))
        img = Image.open(sample_image)
        save_path = os.path.join('static', 'uploaded_img.png')
        img.save(save_path)
        result = model.predict(test_image)
        if result[0][0] >= 0.5:
            prediction = 'Uninfected'
            message = "The person expected to be not infected with malaria"
        else:
            prediction = 'Parasitized'
            message = f"The person is expected to be infected with malaria"

    return render(request, "output.html", {'img_file': 'static/uploaded_img.png', 'prediction': prediction, 'message': message})
