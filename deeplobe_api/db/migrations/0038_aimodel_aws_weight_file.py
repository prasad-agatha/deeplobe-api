# Generated by Django 3.2 on 2022-02-28 05:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('db', '0037_aimodel_job'),
    ]

    operations = [
        migrations.AddField(
            model_name='aimodel',
            name='aws_weight_file',
            field=models.URLField(blank=True, null=True),
        ),
    ]
