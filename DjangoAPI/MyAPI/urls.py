from django.urls import path, include
from . import views
from rest_framework import routers
from django.conf import settings
from django.conf.urls.static import static
from rest_framework.urlpatterns import format_suffix_patterns

router = routers.DefaultRouter()
router.register('MyAPI', views.AssessmentView)
urlpatterns = [
    path('api/', include(router.urls)),
    path('form/', views.cxcontact, name='cxform')
]
