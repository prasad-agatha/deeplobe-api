# Generated by Django 3.2 on 2022-12-20 10:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('db', '0051_apicount_chargebeemodel'),
    ]

    operations = [
        migrations.AddField(
            model_name='apilogger',
            name='file',
            field=models.FileField(blank=True, max_length=255, null=True, upload_to='media_file'),
        ),
    ]
