# Generated by Django 3.2 on 2023-04-10 11:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('db', '0089_apilogger_response_code'),
    ]

    operations = [
        migrations.AddField(
            model_name='userinvitation',
            name='role',
            field=models.CharField(blank=True, choices=[('collaborator', 'collaborator'), ('annotator', 'annotator')], max_length=255, null=True),
        ),
        migrations.AlterField(
            model_name='subscription',
            name='subscription_plan',
            field=models.CharField(default='DeepLobe-Free-Plan-USD-Monthly', max_length=255),
        ),
    ]
