from rest_framework import serializers
from base.models import Asset, Algorithm, Strategy , Tag

class TagSerializer(serializers.ModelSerializer):
    class Meta:
        model = Tag
        fields = '__all__'

class AlgorithmSerializer(serializers.ModelSerializer):
    class Meta:
        model = Algorithm
        fields = '__all__'

class AssetSerializer(serializers.ModelSerializer):
    tags = TagSerializer(many=True)
    models = AlgorithmSerializer(many=True)
    class Meta:
        model = Asset
        fields = '__all__'

class StrategySerializer(serializers.ModelSerializer):
    Asset = AssetSerializer(many=False)
    class Meta:
        model = Strategy
        fields = '__all__'
