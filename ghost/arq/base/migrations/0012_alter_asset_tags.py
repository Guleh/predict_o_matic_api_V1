# Generated by Django 4.0.4 on 2022-06-01 09:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0011_alter_asset_identifier'),
    ]

    operations = [
        migrations.AlterField(
            model_name='asset',
            name='tags',
            field=models.ManyToManyField(blank=True, null=True, to='base.tag'),
        ),
    ]
