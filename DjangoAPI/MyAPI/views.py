from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.decorators import api_view
from django.core import serializers
from rest_framework.response import Response
from rest_framework import status
from . forms import AssessmentForm
from . models import assessment
from . serializers import assessmentSerializers
from django.contrib import messages
from django.http import HttpResponse
import pickle
from sklearn.externals import joblib
from sklearn import preprocessing
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from . apps import MyapiConfig


class AssessmentView(viewsets.ModelViewSet):
	queryset = assessment.objects.all()
	serializer_class = assessmentSerializers

def estimation(unit):
	try:
		comment = [unit]
		tokenizer = MyapiConfig.tokenizer
		mdl = MyapiConfig.myModel
		test_token = tokenizer.texts_to_sequences(comment)
		test_token_pad = pad_sequences(test_token, maxlen = 150)
		y_pred = mdl.predict(x=test_token_pad)
		y_pred = (y_pred>0.58)
		return(y_pred[0][0])
	except ValueError as e:
		return Response(e.args[0], status.HTTP_400_BAD_REQUEST)

def cxcontact(request):
	if request.method == 'POST':
		form = AssessmentForm(request.POST)
		if form.is_valid():
			comment = form.cleaned_data['comment']
			print(comment)
			answer = estimation(comment)
			if answer == True:
				messageAnswer = "positive"
			else:
				messageAnswer = "negative"
			messages.success(request, "Comment = "+ comment)
			messages.success(request, "Your comment is {}".format(messageAnswer))
	form = AssessmentForm()

	return render(request, 'myform/cxform.html', {'form':form})
