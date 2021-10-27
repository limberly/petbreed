from django.http.response import HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from .models import Question, Choice
from django import forms
from fastai.vision.all import *
from django.urls import reverse
from django.utils.datastructures import MultiValueDictKeyError
# Create your views here.


def index(request, result=null):
    try:
        request.FILES['image']
        pred_image = PILImage.create(request.FILES['image'])
        learner = load_learner(request.META['PWD'] + '/polls/model.pkl')
        pred, pred_idx, probs = learner.predict(pred_image)
        result = f"I am {probs[pred_idx]* 100:.01f}% sure this is {pred}"
        context = {'result': result, }
        return render(request, 'polls/index.html', context)
    except MultiValueDictKeyError:
        return render(request, 'polls/index.html')


def model(request):
    # image = PILImage.create(request.FILES['image'])
    # learner = load_learner(request.META['PWD'] + '/polls/model.pkl')
    # pred, pred_idx, probs = learner.predict(image)
    # result = f"I am {probs[pred_idx]* 100:.04f}% sure this is {pred}"
    # context = {'result': result, }
    return "ss"

# https://stackoverflow.com/questions/50777849/from-conda-create-requirements-txt-for-pip3
