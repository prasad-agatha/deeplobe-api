# Generated by Django 3.2 on 2021-11-10 10:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('db', '0019_auto_20211103_1612'),
    ]

    operations = [
        migrations.CreateModel(
            name='MediaAsset',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created', models.DateTimeField(auto_now_add=True, verbose_name='Created At')),
                ('updated', models.DateTimeField(auto_now=True, verbose_name='Last Modified At')),
                ('name', models.CharField(max_length=255)),
                ('asset', models.FileField(upload_to='media_assets')),
            ],
            options={
                'abstract': False,
            },
        ),
    ]
