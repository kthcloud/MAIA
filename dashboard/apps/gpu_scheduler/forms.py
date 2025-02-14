from django import forms

from .models import GPUBooking
from django.conf import settings
from MAIA.dashboard_utils import get_groups_in_keycloak, get_pending_projects, verify_gpu_booking_policy

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
                "class": "form-control",
                "readonly": "readonly"
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
            else:
                existing_bookings = GPUBooking.objects.filter(user_email=user_email)
                booking_data = {
                    "starting_time": start_date.strftime('%Y-%m-%d  %H:%M:%S'),
                    "ending_time": end_date.strftime('%Y-%m-%d  %H:%M:%S')
                }
                is_bookable = verify_gpu_booking_policy(existing_bookings, booking_data)
                # Check existing bookings for the same user

                if not is_bookable:
                    self.add_error('end_date', "The booking is not allowed due to the booking policy.")
        
        return cleaned_data

    class Meta:
        model = GPUBooking
        fields = ('namespace','gpu', 'user_email','start_date', 'end_date')