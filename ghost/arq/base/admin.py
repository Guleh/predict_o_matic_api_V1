from django.contrib import admin

# Register your models here.
from .models import Asset, Algorithm, Strategy, Tag
admin.site.register(Asset)
admin.site.register(Algorithm)
admin.site.register(Strategy)
admin.site.register(Tag)