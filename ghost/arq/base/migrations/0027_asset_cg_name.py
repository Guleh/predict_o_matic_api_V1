# Generated by Django 3.2.13 on 2022-09-23 08:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0026_asset_prediction_term'),
    ]

    operations = [
        migrations.AddField(
            model_name='asset',
            name='cg_name',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
    ]
