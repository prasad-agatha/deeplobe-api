# Generated by Django 3.2 on 2023-04-27 09:31

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('db', '0093_auto_20230427_1457'),
    ]

    operations = [
        migrations.RenameField(
            model_name='aimodel',
            old_name='preproceesing',
            new_name='preprocessing',
        ),
    ]
