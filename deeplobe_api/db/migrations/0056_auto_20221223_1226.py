# Generated by Django 3.2 on 2022-12-23 06:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('db', '0055_alter_subscription_subscriber'),
    ]

    operations = [
        migrations.AddField(
            model_name='subscription',
            name='subscription_plan',
            field=models.CharField(default='Free-INR-Monthly', max_length=255),
        ),
        migrations.AddField(
            model_name='subscriptioncollobarator',
            name='is_collobarator',
            field=models.BooleanField(default=False),
        ),
    ]
