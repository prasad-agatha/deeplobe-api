# Generated by Django 3.2 on 2022-12-23 06:24

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('db', '0054_subscription_subscriptioncollobarator'),
    ]

    operations = [
        migrations.AlterField(
            model_name='subscription',
            name='subscriber',
            field=models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
    ]
