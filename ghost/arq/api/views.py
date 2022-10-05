from rest_framework.response import Response
from rest_framework.decorators import api_view
from base.models import Asset, Algorithm, HitratioHistory
from .serializers import AssetSerializer, AlgorithmSerializer

@api_view(['GET'])
def getRoutes(request):
    routes = [
        {'GET':'api/assets'},
        {'GET':'api/assets/identifier'}
    ]
    return Response(routes)

@api_view(['GET'])
def getAssets(request):
    assets = Asset.objects.filter(isactive=True).order_by('name') 
    serializer = AssetSerializer(assets, many=True)
    return Response(serializer.data)

@api_view(['GET']) 
def getAsset(request, identifier):
    assets = Asset.objects.get(identifier=identifier)
    serializer = AssetSerializer(assets, many=False)
    return Response(serializer.data)

@api_view(['GET']) 
def getModels(request):
    models = Algorithm.objects.all()
    serializer = AlgorithmSerializer(models, many=True)
    return Response(serializer.data)

@api_view(['GET']) 
def getModel(request, identifier):
    asset = Asset.objects.get(identifier=identifier)
    models = Algorithm.objects.filter(asset=asset)
    hitratios = HitratioHistory.objects.filter(asset=asset)
    serializer = AlgorithmSerializer(models, many=True)
    return Response(serializer.data)

    

