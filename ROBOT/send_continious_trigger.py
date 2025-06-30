import socket
import time

# Set up UDP connection
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
endPoint = ("127.0.0.1", 1002)  # Ensure this port matches the listener

def send_trigger(trigger_value):
    """Sends a trigger value over UDP."""
    # Convert the trigger to a bytes object if not already bytes
    if not isinstance(trigger_value, bytes):
        trigger_value = str(trigger_value).encode()
    print(f"Sending trigger: {trigger_value}")
    udp_socket.sendto(trigger_value, endPoint)

while True:
    t = time.time()
    # Get last 4 digits of the integer part
    int_part = int(t) % 10000  
    # Get first 3 digits of the fractional part
    frac_part = int((t - int(t)) * 1000)
    # Format them with leading zeros to ensure proper lengths: 4 digits for int part and 3 digits for frac part
    trigger_str = f"{int_part:04d}{frac_part:03d}"
    send_trigger(trigger_str)
    #time.sleep(1)
    time.sleep(0.01)  # Adjust interval as needed
