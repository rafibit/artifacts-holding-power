





##########################################################################
#this code is before storing the total people count in a single variable.
the new variable is added to optimize the holding power.
##########################################################################
import sensor, image, time, ml, math, uos, gc
from pyb import UART
# from machine import LED

# led = LED("LED_BLUE")

# while True:
#     led.on()
#     time.sleep_ms(150)
#     led.off()
#     time.sleep_ms(100)
#     led.on()
#     time.sleep_ms(150)
#     led.off()
#     time.sleep_ms(600)

gc.collect()
### ML Configuration Section ###
# Initialize sensor
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=2000)

# Load model
try:
    net = ml.Model("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    raise Exception('Failed to load model: ' + str(e))

try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('Failed to load labels: ' + str(e))

# Model settings
min_confidence = 0.8
threshold_list = [(math.ceil(min_confidence * 255), 255)]

# Tracking settings (same as your original)
MATCH_DISTANCE = 100
MAX_MISSED = 700
BASE_PERSON_DISTANCE = 80
CLOSE_PERSON_BUFFER = 30
MAX_OBJECT_HEIGHT = 120
DEBUG_MODE = True

# Tracking data
objects = {}              # Active objects: {id: {data}}
all_time_data = {}        # Permanent time storage: {id: seconds}
next_id = 1               # Next available ID
total_time = 0.0          # Sum of ALL focused time
total_count_302 = 0       # [NEW] Total unique visitors who focused on artifact 302
last_time = time.time()   # For delta time calculation

### LoRaWAN Configuration Section ###
# Initialize UART for Wio E5
uart = UART("LP1", 19200)  # Adjust UART number as needed
uart.init(19200, bits=8, parity=None, stop=1, timeout_char=1000)

# LoRaWAN state
lorawan_joined = False
last_transmission = 0
TRANSMISSION_INTERVAL = 30000  # 30 seconds

# Helper function to send AT commands
def send_at_command(cmd, timeout=1000, expected_response=None):
    print("LoRaWAN:", cmd)
    uart.write(cmd + "\r\n")

    start_time = time.ticks_ms()
    response = ""

    while time.ticks_diff(time.ticks_ms(), start_time) < timeout:
        if uart.any():
            response += uart.read().decode()
            if expected_response and expected_response in response:
                break

    print("LoRaWAN Response:", response)
    return response

# Configure LoRaWAN
def configure_lorawan():
    global lorawan_joined

    # Basic module info
    send_at_command("AT")
    send_at_command("AT+VER")
    send_at_command("AT+ID")

    # Configure LoRaWAN parameters
    send_at_command("AT+MODE=LWOTAA")
    send_at_command("AT+DR=EU868")
    send_at_command("AT+CH=NUM,0-7")

    # Set AppKey (replace with your actual key)
    send_at_command('AT+KEY=APPKEY,"EE67234759A09A9B666F25FA847552F5"')
    send_at_command("AT+DR=DR7")

    # Attempt to join
    response = send_at_command("AT+JOIN", timeout=15000, expected_response="+JOIN: Network joined")
    lorawan_joined = "Network joined" in response
    return lorawan_joined

# Prepare and send data payload
def send_lorawan_data(artefact_id, active_count, total_time, individual_times):
    # Format: artefact_id,active_count,total_time,id1:time1,id2:time2,...
    payload = f"{artefact_id},{active_count},{total_time:.1f}"

    # Add individual times (sorted by ID)
    for obj_id in sorted(individual_times.keys()):
        payload += f",{obj_id}:{individual_times[obj_id]:.1f}"

    # Convert to hex (TTN requirement)
    hex_payload = ''.join('{:02x}'.format(ord(c)) for c in payload)

    # Send via LoRaWAN
    cmd = f'AT+MSGHEX="{hex_payload}"'
    send_at_command(cmd)

### ML Helper Functions (same as your original) ###
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def update_time():
    global total_time, last_time
    current_time = time.time()
    delta = current_time - last_time
    last_time = current_time

    active_count = 0
    for obj_id in list(objects.keys()):
        if objects[obj_id]['active']:
            all_time_data[obj_id] += delta
            active_count += 1
        else:
            objects[obj_id]['missed'] += 1
            if objects[obj_id]['missed'] > MAX_MISSED:
                del objects[obj_id]

    total_time = sum(all_time_data.values())
    return active_count

def get_effective_distance(detection):
    x, y, w, h, score = detection
    if h > MAX_OBJECT_HEIGHT:
        return BASE_PERSON_DISTANCE - CLOSE_PERSON_BUFFER
    return BASE_PERSON_DISTANCE

def group_detections_by_person(detections):
    groups = []
    for det in sorted(detections, key=lambda d: -d[4]):
        x, y, w, h, score = det
        cx, cy = x + w//2, y + h//2
        matched = False
        effective_dist = get_effective_distance(det)

        if DEBUG_MODE and h > MAX_OBJECT_HEIGHT:
            img.draw_rectangle(x, y, w, h, color=(255, 255, 0))

        for group in groups:
            for gx, gy, gw, gh, _ in group:
                g_cx, g_cy = gx + gw//2, gy + gh//2
                if distance((cx, cy), (g_cx, g_cy)) < effective_dist:
                    group.append(det)
                    matched = True
                    break
            if matched:
                break

        if not matched:
            groups.append([det])
    return groups

def get_best_detection_per_person(detections):
    groups = group_detections_by_person(detections)
    return [max(group, key=lambda d: d[4]) for group in groups]

def find_matching_object(cx, cy):
    obj_id = None
    min_dist = float('inf')
    for id, obj in objects.items():
        if not obj['active']:
            dist = distance((cx, cy), obj['last_pos'])
            if dist < min_dist and dist < MATCH_DISTANCE:
                min_dist = dist
                obj_id = id
    return obj_id

def fomo_post_process(model, inputs, outputs):
    ob, oh, ow, oc = model.output_shape[0]
    x_scale = inputs[0].roi[2] / ow
    y_scale = inputs[0].roi[3] / oh
    scale = min(x_scale, y_scale)
    x_offset = ((inputs[0].roi[2] - (ow * scale)) / 2 + inputs[0].roi[0])
    y_offset = ((inputs[0].roi[3] - (ow * scale)) / 2 + inputs[0].roi[1])
    l = [[] for i in range(oc)]
    for i in range(oc):
        img = image.Image(outputs[0][0, :, :, i] * 255)
        blobs = img.find_blobs(
            threshold_list, x_stride=1, y_stride=1, area_threshold=1, pixels_threshold=1
        )
        for b in blobs:
            rect = b.rect()
            x, y, w, h = rect
            score = (
                img.get_statistics(thresholds=threshold_list, roi=rect).l_mean() / 255.0
            )
            x = int((x * scale) + x_offset)
            y = int((y * scale) + y_offset)
            w = int(w * scale)
            h = int(h * scale)
            l[i].append((x, y, w, h, score))
    return l

### Main Loop ###
clock = time.clock()

# Initialize LoRaWAN
if not configure_lorawan():
    print("Warning: Failed to join LoRaWAN network")

while True:
    clock.tick()
    img = sensor.snapshot()
    current_time = time.time()

    # Reset all to inactive at start of frame
    for obj in objects.values():
        obj['active'] = False

    # Process detections
    for i, detection_list in enumerate(net.predict([img], callback=fomo_post_process)):
        if i == 0 or len(detection_list) == 0:
            continue

        label = labels[i]

        if label == "focused":
            filtered_detections = get_best_detection_per_person(detection_list)

            for x, y, w, h, score in filtered_detections:
                cx, cy = x + w//2, y + h//2
                img.draw_circle((cx, cy, 12), color=(255, 0, 0))
                img.draw_string(cx+15, cy-5, "FOCUS", (255, 0, 0))

                obj_id = find_matching_object(cx, cy)

                if obj_id is None:
                    too_close = any(
                        distance((cx, cy), obj['last_pos']) < BASE_PERSON_DISTANCE
                        for obj in objects.values() if obj['active']
                    )

                    if not too_close:
                        obj_id = next_id
                        next_id += 1
                        objects[obj_id] = {
                            'last_pos': (cx, cy),
                            'active': True,
                            'missed': 0
                        }
                        all_time_data[obj_id] = 0.0
                else:
                    objects[obj_id].update({
                        'last_pos': (cx, cy),
                        'active': True,
                        'missed': 0
                    })

                if obj_id is not None:
                    img.draw_string(cx+15, cy+15, f"ID:{obj_id}", (255, 255, 255))
        else:
            for x, y, w, h, score in detection_list:
                cx, cy = x + w//2, y + h//2
                img.draw_circle((cx, cy, 12), color=(0, 255, 0))

    # Update time tracking
    active_count = update_time()

    # Display on screen
    img.draw_string(5, 5, "Artefact ID: 302", (0, 0, 255))
    img.draw_string(5, 20, f"Total Focused: {total_time:.1f}s", (0, 0, 255))
    img.draw_string(5, 30, f"Current Focus: {active_count}", (0, 0, 255))


    y = 45
    for obj_id in sorted(all_time_data.keys()):
        status = "ACTIVE" if obj_id in objects and objects[obj_id]['active'] else "INACTIVE"
        color = (0, 0, 255) if status == "ACTIVE" else (255, 0, 0)
        img.draw_string(5, y, f"ID {obj_id}: {all_time_data[obj_id]:.1f}s - {status}", color)
        y += 10

    # Send data via LoRaWAN periodically
    if lorawan_joined and (time.ticks_diff(time.ticks_ms(), last_transmission) > TRANSMISSION_INTERVAL):
        send_lorawan_data(302, active_count, total_time, all_time_data)
        last_transmission = time.ticks_ms()

    print(f"FPS: {clock.fps():.1f} | Active: {active_count} | Total: {total_time:.1f}s")

