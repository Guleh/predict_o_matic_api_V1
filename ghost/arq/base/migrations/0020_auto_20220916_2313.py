# Generated by Django 3.2.13 on 2022-09-16 21:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0019_auto_20220916_1503'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='algorithm',
            name='asset',
        ),
        migrations.AddField(
            model_name='asset',
            name='algorithm',
            field=models.ManyToManyField(to='base.Algorithm'),
        ),
    ]
