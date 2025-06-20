# Generated by Django 3.2 on 2023-02-08 06:47

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('db', '0079_alter_contactus_requested_by'),
    ]

    operations = [
        migrations.AddField(
            model_name='annotationhireexpertrequest',
            name='type',
            field=models.CharField(default='annotation', max_length=255),
        ),
        migrations.AlterField(
            model_name='annotationhireexpertrequest',
            name='requested_by',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
    ]
