# Generated by Django 4.0.4 on 2022-05-27 11:42

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0004_remove_strategy_asset_strategy_algorithm'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='tag',
            name='assets',
        ),
        migrations.AddField(
            model_name='asset',
            name='tags',
            field=models.ManyToManyField(to='base.tag'),
        ),
    ]
