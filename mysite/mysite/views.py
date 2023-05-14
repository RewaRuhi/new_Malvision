from django.http import HttpResponse
from django.shortcuts import render
import cv2
import numpy as np
from .services import process_image

def homePage(request): 
   # data={
       # 'title':'my home page',
       # 'bdata': 'welcome Rewa Ruhi at mysite',
       # 'clist':['php','java','python'],
        #'numbers':[10,20,30,40,50],
       # 'student_details':[
      #  {'name': 'rewa','phone':9905528922},
      #   {'name': 'ruhi','phone':79005528922}
       # ]
    #}
    return render(request,"malaria-home.html")
    #return render(request,"after-predict.html")

def aboutUs(request):
    return render(request,"after-predict.html")
def about(request):
    return render(request,"about.html")
def contact(request):
    return render(request,"contact.html")

def predict(request):

    #def process_image:
    if request.method == "POST":
        sample_image = request.FILES["sample_image"]
        # img= cv2.imread(sample_image)
        img = cv2.imdecode(np.frombuffer(sample_image.read(),np.uint8),cv2.IMREAD_COLOR)
        prcoess_image(img)


    

    
        

    return render(request,"output.html")

