from django.shortcuts import render
from django.http import JsonResponse, response
from rest_framework.views import APIView
from . import model_builder_function as mb

# Create your views here.
class ModelBuilder(APIView):
    def post(self, request):
        model = mb.build_model(request.data['data'], verbose=False)
        #print(model)
        return JsonResponse(model)