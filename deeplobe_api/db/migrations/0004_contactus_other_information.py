# Generated by Django 3.2 on 2021-05-21 08:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('db', '0003_contactus'),
    ]

    operations = [
        migrations.AddField(
            model_name='contactus',
            name='other_information',
            field=models.TextField(default='Empty'),
            preserve_default=False,
        ),
    ]
