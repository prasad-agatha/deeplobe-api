# Generated by Django 3.2 on 2022-02-03 08:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('db', '0032_alter_user_profile_pic'),
    ]

    operations = [
        migrations.AddField(
            model_name='contactus',
            name='title',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
