from django.shortcuts import render
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from .models import GPUBooking
from django.conf import settings
from datetime import datetime, timezone
from .forms import GPUBookingForm
from MAIA.dashboard_utils import get_namespaces


@method_decorator(csrf_exempt, name='dispatch')  # 🚀 This disables CSRF for this API
class GPUSchedulabilityAPIView(APIView):
    permission_classes = [AllowAny]  # 🚀 Allow requests without authentication or CSRF

    def post(self, request, *args, **kwargs):
        try:
            user_email = request.data.get("user_email")
            namespace = request.data.get("namespace")
            if not user_email:
                return Response({"error": "Missing user_email"}, status=400)
            if not namespace:
                return Response({"error": "Missing namespace"}, status=400)

            secret_token = request.data.get("token")
            if not secret_token or secret_token != settings.SECRET_KEY:
                return Response({"error": "Invalid or missing secret token"}, status=403)

            if "booking" in request.data:
                booking_data = request.data["booking"]
                
                # Calculate the total number of days for existing bookings
                existing_bookings = GPUBooking.objects.filter(user_email=user_email)
                total_days = sum(
                    (booking.end_date - booking.start_date).days
                    for booking in existing_bookings
                )
                
                # Calculate the number of days for the new booking
                ending_time = datetime.strptime(booking_data["ending_time"], "%Y-%m-%d %H:%M:%S")
                starting_time = datetime.strptime(booking_data["starting_time"], "%Y-%m-%d %H:%M:%S")
                gpu = booking_data["gpu"]
                new_booking_days = (ending_time - starting_time).days
                
                # Verify that the sum of existing bookings and the new booking does not exceed 60 days
                if total_days + new_booking_days > 60:
                    return Response({"error": "Total booking days exceed the limit of 60 days"}, status=400)
                
                # Create the new booking
                GPUBooking.objects.create(
                    user_email=user_email,
                    start_date=booking_data["starting_time"],
                    end_date=booking_data["ending_time"],
                    namespace=namespace,
                    gpu=gpu
                )
                return Response({"message": "Booking created successfully"})

            try:
                user_statuses = GPUBooking.objects.filter(user_email=user_email)
                
                is_schedulable = False
                
                current_time = datetime.now(timezone.utc)
                is_schedulable = any(
                    status.start_date <= current_time and status.end_date >= current_time and status.namespace == namespace
                    for status in user_statuses
                )
                if is_schedulable:
                    status = next(status for status in user_statuses if status.start_date <= current_time and status.end_date >= current_time)
                    return Response({"schedulable": is_schedulable, "until": status.end_date})
                else:
                    return Response({"schedulable": is_schedulable, "until": None})
            except GPUBooking.DoesNotExist:
                return Response({"schedulable": False, "until": None})  # Default to not schedulable if not found

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)


@login_required(login_url="/maia/login/")
def book_gpu(request):
    msg = None
    success = False

    email = request.user.email

    id_token = request.session.get('oidc_id_token')
    groups = request.user.groups.all()

    namespaces = []
    
    if request.user.is_superuser:
        namespaces = get_namespaces(id_token, api_urls=settings.API_URL, private_clusters=settings.PRIVATE_CLUSTERS)

    if namespaces is None or len(namespaces) == 0:
        namespaces = []
        for group in groups:
            if str(group) != "MAIA:users":
                namespaces.append(str(group).split(":")[-1].lower().replace("_", "-"))

    initial_data = {'user_email': email, 'namespace': namespaces[0] if namespaces else None}
    
    if request.method == "POST":
        form = GPUBookingForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            msg = 'Request for GPU Booking submitted successfully.'
            success = True
            return redirect("/maia/gpu-booking/my-bookings/")
        else:
            print(form.errors)
            msg = 'Form is not valid'
    else:
        form = GPUBookingForm(request.POST or None, request.FILES or None, initial=initial_data)
        form.fields['namespace'].choices = [(ns, ns) for ns in namespaces]

    return render(request, "accounts/gpu_booking.html", {"dashboard_version": settings.DASHBOARD_VERSION, "form": form, "msg": msg, "success": success})


@login_required(login_url="/maia/login/")
def gpu_booking_info(request):
    
    id_token = request.session.get('oidc_id_token')
    groups = request.user.groups.all()
    namespaces = []
    if request.user.is_superuser:
        namespaces = get_namespaces(id_token, api_urls=settings.API_URL, private_clusters=settings.PRIVATE_CLUSTERS)

    else:
        for group in groups:
            if str(group) != "MAIA:users":
                namespaces.append(str(group).split(":")[-1].lower().replace("_","-"))
                
    bookings = GPUBooking.objects.filter(user_email=request.user.email)
    
    total_days = 0
    for booking in bookings:
        total_days += (booking.end_date - booking.start_date).days
    
    context = {"namespaces": namespaces, "dashboard_version": settings.DASHBOARD_VERSION, "bookings": bookings, "total_days": total_days}
    if request.user.is_superuser:
        context["username"] = request.user.username + " [ADMIN]"
        context["user"] = ["admin"]
    else:
        context["username"] = request.user.username
    return render(request, "accounts/gpu_booking_info.html", context=context)