# Generated by Django 3.2 on 2021-10-04 11:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('db', '0014_apikey_expire_date'),
    ]

    operations = [
        migrations.AddField(
            model_name='apikey',
            name='active',
            field=models.BooleanField(default=True),
        ),
    ]
