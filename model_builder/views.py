from django.shortcuts import render
from django.http import JsonResponse, response
from rest_framework.views import APIView

# Create your views here.
class ModelBuilder(APIView):
    def post(self, request):
        print(request.data)
        return JsonResponse({"data": [1,5,4]})