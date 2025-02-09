from django import forms

from .models import GPUBooking
from django.conf import settings
from MAIA.dashboard_utils import get_groups_in_keycloak, get_pending_projects


class GPUBookingForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super(GPUBookingForm, self).__init__(*args, **kwargs)
        maia_groups = get_groups_in_keycloak(settings= settings)
        pending_projects = get_pending_projects(settings=settings)

        for pending_project in pending_projects:
            maia_groups[pending_project] = pending_project + " (Pending)"

        self.fields['namespace'] = forms.ChoiceField(
            choices=[(maia_group,maia_group) for maia_group in maia_groups.values()],
            widget=forms.Select(attrs={
            'class': "form-select text-center fw-bold",
                'style': 'max-width: auto;',
            }
        )
        )
        self.fields['user_email'] = forms.EmailField(
            widget=forms.EmailInput(
                attrs={
                    "placeholder": "Your Email.",
                    "class": "form-control"
                }
            ))
        
        self.fields['gpu'] = forms.ChoiceField(
            choices=[(gpu,gpu) for gpu in settings.GPU_LIST],
            widget=forms.Select(attrs={
                'class': "form-select text-center fw-bold",
                'style': 'max-width: auto;',
            })
        )

        self.fields['start_date'] = forms.DateField(widget=forms.TextInput(attrs={'class': 'form-control', 'type': 'date'}))
        
        self.fields['end_date'] = forms.DateField(widget=forms.TextInput(attrs={'class': 'form-control', 'type': 'date'}))
    
    def clean(self):
        cleaned_data = super().clean()
        start_date = cleaned_data.get("start_date")
        end_date = cleaned_data.get("end_date")
        user_email = cleaned_data.get("user_email")

        if start_date and end_date:
            if end_date <= start_date:
                self.add_error('end_date', "End date must be after start date.")
            elif (end_date - start_date).days > 60:
                self.add_error('end_date', "The maximum booking duration is 60 days.")
            else:
                # Check existing bookings for the same user
                existing_bookings = GPUBooking.objects.filter(user_email=user_email)
                total_days = 0
                for booking in existing_bookings:
                    total_days += (booking.end_date - booking.start_date).days

                if total_days > 60:
                    self.add_error('end_date', "The total booking duration for this user exceeds 60 days.")
        
        return cleaned_data

    

    class Meta:
        model = GPUBooking
        fields = ('namespace','gpu', 'user_email','start_date', 'end_date')