# Generated by Django 3.2.13 on 2024-10-16 06:21

import datetime
import django.contrib.auth.models
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('auth', '0012_alter_user_first_name_max_length'),
    ]

    operations = [
        migrations.CreateModel(
            name='MAIAUser',
            fields=[
                ('user_ptr', models.OneToOneField(auto_created=True, on_delete=django.db.models.deletion.CASCADE, parent_link=True, primary_key=True, serialize=False, to='auth.user')),
                ('namespace', models.CharField(blank=True, max_length=150, verbose_name='namespace')),
                ('gpu', models.CharField(blank=True, max_length=150, null=True, verbose_name='gpu')),
                ('date', models.DateField(default=datetime.date.today, verbose_name='date')),
                ('conda', models.TextField(default='N/A', null=True, verbose_name='conda')),
                ('cluster', models.TextField(default='N/A', null=True, verbose_name='cluster')),
            ],
            bases=('auth.user',),
            managers=[
                ('objects', django.contrib.auth.models.UserManager()),
            ],
        ),
    ]
