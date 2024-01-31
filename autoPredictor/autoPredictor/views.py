import pandas as pd
from django.shortcuts import render
import numpy as np
import joblib
from django.conf import settings

def predictor(request):
    payload ={}
    if request.method=='POST':
       data1 = {'highway-mpg': 0,
        'city-mpg': 0,
        'peak-rpm': 5099.740932642487,
        'horsepower': 0,
        'compression-ratio': 10.14362694300518,
        'stroke': 0,
        'bore': 0,
        'engine-size': 0,
        'num-of-cylinders': 0,
        'curb-weight': 0,
        'height': 53.869948186528504,
        'width': 0,
        'length': 0,
        'wheel-base': 0,
        'num-of-doors': 4,
        'fuel-system_encoded': 0,
        'engine-type_encoded': 0,
        'eng-location_front': 0.9844559585492227,
        'eng-location_rear': 0.015544041450777202,
        'body-style_convertible': 0,
        'body-style_hardtop': 0,
        'body-style_hatchback': 0,
        'body-style_sedan': 0,
        'body-style_wagon': 0,
        'fuel-type_diesel': 0,
        'fuel-type_gas': 0,
        'make_alfa-romero': 0,
        'make_audi': 0,
        'make_bmw': 0,
        'make_chevrolet': 0,
        'make_dodge': 0,
        'make_honda': 0,
        'make_isuzu': 0,
        'make_jaguar': 0,
        'make_mazda': 0,
        'make_mercedes-benz': 0,
        'make_mercury': 0,
        'make_mitsubishi': 0,
        'make_nissan': 0,
        'make_peugot': 0,
        'make_plymouth': 0,
        'make_porsche': 0,
        'make_saab': 0,
        'make_subaru': 0,
        'make_toyota': 0,
        'make_volkswagen': 0,
        'make_volvo': 0}
    
       formData = dict(request.POST.items())
       del formData['csrfmiddlewaretoken']
       data1[formData['make']]=1
       del formData['make']
       data1[formData['body-style']]=1
       del formData['body-style']
       data1[formData['fuel-type']]=1
       del formData['fuel-type']
       for key in formData.keys():
        data1[key] = float(formData[key])


       file = (str(settings.BASE_DIR)+'/autoPredictor/lassoModelAutomobile.pkl')
       La = joblib.load(file) 
       values = []
       for x in data1:
         values.append(data1[x])


       your_input_2d = np.array([values])
       predicted = La.predict(your_input_2d)
       print(predicted)
       payload['predicted'] = predicted[0]

       return render(request, "index.html", payload)
    return render(request, "index.html", payload)
