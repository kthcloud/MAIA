# Generated by Django 3.2.13 on 2024-10-17 14:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('authentication', '0002_maiauser_minimal_env'),
    ]

    operations = [
        migrations.AddField(
            model_name='maiauser',
            name='cpu_limit',
            field=models.TextField(default='1.0', null=True, verbose_name='memory_limit'),
        ),
        migrations.AddField(
            model_name='maiauser',
            name='memory_limit',
            field=models.TextField(default='1G', null=True, verbose_name='memory_limit'),
        ),
        migrations.AlterField(
            model_name='maiauser',
            name='minimal_env',
            field=models.BooleanField(default=True, null=True, verbose_name='minimal_env'),
        ),
    ]