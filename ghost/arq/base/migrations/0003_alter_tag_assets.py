# Generated by Django 4.0.4 on 2022-05-27 11:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0002_asset_confidence_asset_current_prediction_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='tag',
            name='assets',
            field=models.ManyToManyField(to='base.asset'),
        ),
    ]
