from MAIA.dashboard_utils import verify_gpu_availability
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

GPU_SPECS = [
    {
        "name": "RTX 2080 Ti",
        "count": 1,
        "replicas": 1,
        "cluster": "maia-small",
        "vram": "11 GiB"
    }
]

EXISTING_BOOKING = [
    {"id": 0, "gpu": "RTX 2080 Ti", "start_date": "2021-01-01 00:00:00", "end_date": "2021-01-02 00:00:00"},
    {"id": 1, "gpu": "RTX 2080 Ti", "start_date": "2021-01-02 06:00:00", "end_date": "2021-01-04 00:00:00"},
    {"id": 2, "gpu": "RTX 2080 Ti", "start_date": "2021-01-05 00:00:00", "end_date": "2021-01-06 00:00:00"}
]


def plot_gpu_availability(new_booking, overlapping_time_slots, gpu_availability_per_slot, total_replicas=1):
    for booking in EXISTING_BOOKING:
        booking["starting_date"] = datetime.strptime(booking["start_date"], "%Y-%m-%d %H:%M:%S")
        booking["ending_date"] = datetime.strptime(booking["end_date"], "%Y-%m-%d %H:%M:%S")

    new_booking["starting_date"] = datetime.strptime(new_booking["start_date"], "%Y-%m-%d %H:%M:%S")
    new_booking["ending_date"] = datetime.strptime(new_booking["end_date"], "%Y-%m-%d %H:%M:%S")
    # Plotting
    fig, ax = plt.subplots()

    for booking in EXISTING_BOOKING:
        ax.plot([booking["starting_date"], booking["ending_date"]], [booking["id"], booking["id"]], label="Booking ID: "+str(booking["id"]), linewidth=2)

    ax.plot([new_booking["starting_date"], new_booking["ending_date"]], [new_booking["id"], new_booking["id"]], label="New Booking ID: "+str(new_booking["id"]), linewidth=4)

    # Draw vertical lines for overlapping time slots
    for i in range(len(overlapping_time_slots)):
        ax.axvline(overlapping_time_slots[i], color='r', linestyle='--', linewidth=1)
        if i < len(overlapping_time_slots) - 1:
            ax.text((overlapping_time_slots[i] + (overlapping_time_slots[i+1] - overlapping_time_slots[i]) / 2), 3.5, str(gpu_availability_per_slot[i])+'/'+str(total_replicas), verticalalignment='center')



    # Formatting the plot
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.gcf().autofmt_xdate()
    plt.ylim(-1, new_booking["id"]+1)
    plt.yticks([])
    plt.xlabel("Date")
    plt.title("GPU Booking: {}".format(new_booking["gpu"]))
    plt.legend()

    fig.savefig("gpu_booking.png")



def test_booking():
    new_booking = {"id": 3, "gpu": "RTX 2080 Ti", "start_date": "2021-01-02 00:00:00", "end_date": "2021-01-05 00:00:00"}
    overlapping_time_slots, gpu_availability_per_slot, total_replicas = verify_gpu_availability(global_existing_bookings=EXISTING_BOOKING, new_booking=new_booking, gpu_specs=GPU_SPECS)

    for idx, gpu_availability in enumerate(gpu_availability_per_slot):
        if gpu_availability == 0:
            print("GPU not available between the selected time slots: {} - {}".format(overlapping_time_slots[idx], overlapping_time_slots[idx+1]))
            plot_gpu_availability(new_booking, overlapping_time_slots, gpu_availability_per_slot, total_replicas)
            return False
    
    print("GPU available for booking")
    plot_gpu_availability(new_booking, overlapping_time_slots, gpu_availability_per_slot, total_replicas)
    return True



if __name__ == "__main__":
    test_booking()
