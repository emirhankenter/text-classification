from rest_framework import serializers
from . models import assessment

class assessmentSerializers(serializers.ModelSerializer):
	class Meta:
		model=assessment
		fields='__all__'