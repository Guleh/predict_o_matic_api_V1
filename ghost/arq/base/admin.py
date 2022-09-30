from django.contrib import admin

class AssetAdmin(admin.ModelAdmin):
    list_display = ('platformsymbol','name','timeframe','last_updated')

    
class AlgorithmAdmin(admin.ModelAdmin):
    list_display = ('asset', 'name','last_updated')
# Register your models here.
from .models import Asset, Algorithm, Strategy, Tag, HitratioHistory
admin.site.register(Asset, AssetAdmin)
admin.site.register(Algorithm, AlgorithmAdmin)
admin.site.register(Strategy)
admin.site.register(Tag)
admin.site.register(HitratioHistory)