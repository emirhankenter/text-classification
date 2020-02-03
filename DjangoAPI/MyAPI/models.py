from django.db import models

# Create your models here.
class assessment(models.Model):
	comment = models.CharField(max_length=1000)

	def __str__(self):
		return '{}'.format(self.comment)
