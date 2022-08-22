from rest_framework.response import Response
from rest_framework.decorators import api_view
from base.models import Asset
from .serializers import AssetSerializer

@api_view(['GET'])
def getRoutes(request):
    routes = [
        {'GET':'api/assets'},
        {'GET':'api/assets/identifier'}
    ]
    return Response(routes)

@api_view(['GET'])
def getAssets(request):
    assets = Asset.objects.all() 
    serializer = AssetSerializer(assets, many=True)
    return Response(serializer.data)

@api_view(['GET']) 
def getAsset(request, identifier):
    assets = Asset.objects.get(identifier=identifier)
    serializer = AssetSerializer(assets, many=False)
    return Response(serializer.data)

'''
@api_view(['POST'])
def createAccount(request):
    data = request.data
    user = User(email = data['email'])
    user.save()
    account = Account(user = user, email = data['email'], mex_pk = data['mex_pk'], mex_sk = data['mex_sk'])
    account.save()
    serializer = AccountSerializer(account, many=False)
    return Response(serializer.data)

@api_view(['GET'])
def getAccounts(request):
    accounts = Account.objects.all()
    serializer = AccountSerializer(accounts, many=True)
    return Response(serializer.data)

@api_view(['GET'])
def getAccount(request, pk):
    account = Account.objects.get(id=pk)
    serializer = AccountSerializer(account, many=False)
    return Response(serializer.data)
'''
