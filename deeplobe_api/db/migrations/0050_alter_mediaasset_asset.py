# Generated by Django 3.2 on 2022-12-13 02:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('db', '0049_alter_apikey_table'),
    ]

    operations = [
        migrations.AlterField(
            model_name='mediaasset',
            name='asset',
            field=models.FileField(max_length=255, upload_to='media_assets'),
        ),
    ]
