# Generated by Django 3.2 on 2022-06-06 09:12

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('db', '0045_auto_20220330_1252'),
    ]

    operations = [
        migrations.CreateModel(
            name='AutoAnnotatePredictionModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created', models.DateTimeField(auto_now_add=True, verbose_name='Created At')),
                ('updated', models.DateTimeField(auto_now=True, verbose_name='Last Modified At')),
                ('input_images', models.JSONField()),
                ('result', models.JSONField(null=True)),
                ('extra', models.JSONField(blank=True, null=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'verbose_name': 'AutoAnnotatePredictionModel',
                'verbose_name_plural': ' AutoAnnotatePredictionModels',
                'db_table': 'autoannotatepredictionmodel',
            },
        ),
    ]
