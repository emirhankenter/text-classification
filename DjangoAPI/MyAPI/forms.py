from django import forms
from django.forms import ModelForm
from . models import assessment

class AssessmentForm(forms.Form):
	comment = forms.CharField(max_length=1000, widget = forms.Textarea(attrs={'placeholder': 'Enter your comment'}))