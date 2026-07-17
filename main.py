import cv2
import igus
import json
import threading
import time
import numpy as np
import paho.mqtt.client as mqtt
import random
import string

from vision import BoxDetector, Box

# ============================================================================
# CONFIGURE THESE
# ============================================================================

ROBOT_HOST = "192.168.3.11"
ROBOT_PORT = 3920

MQTT_HOST = "mqtt-dashboard.com"
MQTT_PORT = 1883

ROBOT_NUMBER = 38
TOPIC_BASE   = f"IGUS/robot{ROBOT_NUMBER}"
TOPIC_SORT   = TOPIC_BASE + "/sort"
TOPIC_STATUS = TOPIC_BASE + "/status"

GRASP_Z    = 148.5   # Z where suction cup touches top of box (measured)
TRAVEL_Z   = 280.0   # safe travel height — well above all boxes
BOX_HEIGHT = 50.0    # actual box height = 5cm = 50mm

SUCTION_CHANNEL = 31  # ToolDOut2 (DOut32) — confirmed working

ORIENT = (180.0, 0.0, 180.0)

FACE_AWAY_POSE = igus.Joint(A1=90.0)

# Gripper offset correction (mm)
# If suction cup lands off-center on the box, adjust these:
# Positive X_OFFSET → shifts pick position further from robot base
# Negative X_OFFSET → shifts pick position closer to robot base
# Positive Y_OFFSET → shifts pick position in +Y direction
# Negative Y_OFFSET → shifts pick position in -Y direction
GRIPPER_X_OFFSET = 0.0   # tune if gripper lands off-center
GRIPPER_Y_OFFSET = -17.0   # tune if gripper lands off-center

SETTLE_TIME = 2.0   # seconds to wait after gripper on/off

# ============================================================================
# ROBOT SETUP
# ============================================================================

robot = igus.IGUS(host=ROBOT_HOST, port=ROBOT_PORT, name="IGUS REBEL (physical)")
# robot.wait=True blocks go_to() until the live feedback position is EXACTLY
# bit-equal to the commanded setpoint (Joint/Cart.__eq__ do a plain float ==).
# Real position feedback settles close to the target but essentially never
# becomes bit-exact, so that wait can spin forever mid-sequence. We wait
# ourselves instead, with a tolerance and a timeout.
robot.wait = False
robot.connect()


def wait_until_close(getter, target: dict, tolerance=1.0, timeout=15.0, poll=0.05):
    start = time.time()
    while time.time() - start < timeout:
        current = getter()
        if current is not None:
            current_dict = current.get_dict()
            if all(abs(current_dict[key] - value) <= tolerance for key, value in target.items()):
                return True
        time.sleep(poll)
    return False


def go_to_and_confirm(pos, vel, target: dict, getter, max_retries=3, **wait_kwargs):
    """
    Send a move and block until the robot actually arrives (within tolerance) —
    re-sending the same command and waiting again if it stalls, instead of
    giving up after one timeout and firing the next command onto a robot
    that's still mid-motion. If it never arrives after max_retries, raise so
    the run halts for a human rather than continuing to sort on top of an
    unconfirmed position (e.g. crashing into the storage stack).
    """
    for attempt in range(1, max_retries + 1):
        robot.go_to(pos, vel=vel)
        if wait_until_close(getter, target, **wait_kwargs):
            return
        print(f"[motion] attempt {attempt}/{max_retries} timed out reaching "
              f"{target} (last seen: {getter()}) — retrying")
    raise RuntimeError(f"[motion] FAILED to reach {target} after {max_retries} "
                       f"attempts — halting for safety.")


def _set_dout(channel: int, state: bool):
    robot.send(igus.Command.dout(channel, state))


def gripper_open():
    """Vacuum gripper: suction OFF — release the box."""
    _set_dout(SUCTION_CHANNEL, False)
    print("[Gripper] SUCTION OFF (release)")
    time.sleep(SETTLE_TIME)


def gripper_close():
    """Vacuum gripper: suction ON — grab the box."""
    _set_dout(SUCTION_CHANNEL, True)
    print("[Gripper] SUCTION ON (grab)")
    time.sleep(SETTLE_TIME)


# ============================================================================
# MQTT SETUP
# ============================================================================

_client_id = "app-" + "".join(random.choices(string.ascii_letters + string.digits, k=12))
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1,
                          client_id=_client_id, clean_session=True)

_storage_x: float = 0.0
_storage_y: float = 0.0
_sort_requested = threading.Event()


def publish_status(message):
    print(f"[MQTT OUT] {message}")
    mqtt_client.publish(TOPIC_STATUS, message)


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[MQTT] Connected to {MQTT_HOST}:{MQTT_PORT}")
        client.subscribe(TOPIC_SORT)
        print(f"[MQTT] Subscribed to {TOPIC_SORT}")
    else:
        print(f"[MQTT] Connection failed: rc={rc}")


def on_disconnect(client, userdata, rc):
    print(f"[MQTT] Disconnected: rc={rc}")


def on_message(client, userdata, msg):
    global _storage_x, _storage_y
    topic   = msg.topic
    payload = msg.payload.decode("utf-8")
    print(f"[MQTT IN]  {topic}  →  {payload}")

    if topic == TOPIC_SORT:
        try:
            data = json.loads(payload)
            _storage_x = float(data["X"])
            _storage_y = float(data["Y"])
            print(f"[SORT] Storage spot set to X={_storage_x} Y={_storage_y} mm")
            _sort_requested.set()
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            print(f"[SORT] Bad payload: {e}")


mqtt_client.on_connect    = on_connect
mqtt_client.on_disconnect = on_disconnect
mqtt_client.on_message    = on_message

mqtt_client.connect(MQTT_HOST, MQTT_PORT)
mqtt_client.loop_start()

# ============================================================================
# VISION SETUP
# ============================================================================

detector = BoxDetector()

WIN = "Autonomous Sorting — Live View"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)


def robot_to_pixel(rx: float, ry: float):
    M  = detector.transform
    A2 = M[:, :2]
    b  = np.array([rx - M[0, 2], ry - M[1, 2]], dtype=np.float32)
    uv = np.linalg.solve(A2, b)
    return int(uv[0]), int(uv[1])


def refresh_camera(boxes=None, active_id=None, storage_pixel=None):
    """Update the OpenCV window with a fresh camera frame during robot motion."""
    try:
        ok, frame = detector.cap.read()
        if ok and frame is not None:
            vis = detector.draw(frame,
                                boxes if boxes is not None else [],
                                active_id=active_id,
                                storage_pixel=storage_pixel)
            cv2.imshow(WIN, vis)
            cv2.waitKey(1)
    except Exception:
        pass  # never let camera refresh crash the robot motion


# ============================================================================
# PICK-AND-PLACE MOTION
# ============================================================================

def _move(source: igus.Cart, destination: igus.Cart,
          boxes=None, active_id=None, storage_pixel=None):
    """
    Full pick-and-place motion sequence with live camera updates:
    1. Suction OFF → move above source → lower onto box
    2. Suction ON → lift → travel to storage → lower onto stack
    3. Suction OFF → retreat up
    Camera refreshes between each step so the live view stays active.
    """
    src_above = igus.Cart(source.X,      source.Y,      TRAVEL_Z, *ORIENT)
    dst_above = igus.Cart(destination.X, destination.Y, TRAVEL_Z, *ORIENT)

    print(f"[MOVE] src=({source.X:.0f},{source.Y:.0f},{source.Z:.0f})"
          f"  dst=({destination.X:.0f},{destination.Y:.0f},{destination.Z:.0f})")

    gripper_open()
    refresh_camera(boxes, active_id, storage_pixel)

    print(f"[MOVE] Going above box: X={src_above.X:.0f} Y={src_above.Y:.0f} Z={src_above.Z:.0f}")
    go_to_and_confirm(src_above, 80.0,
                       {"X": src_above.X, "Y": src_above.Y, "Z": src_above.Z},
                       robot.get_current_cart)
    refresh_camera(boxes, active_id, storage_pixel)

    print(f"[MOVE] Lowering onto box: Z={source.Z:.0f}")
    go_to_and_confirm(source, 20.0,
                       {"X": source.X, "Y": source.Y, "Z": source.Z},
                       robot.get_current_cart)
    refresh_camera(boxes, active_id, storage_pixel)

    gripper_close()
    refresh_camera(boxes, active_id, storage_pixel)

    print(f"[MOVE] Lifting box")
    go_to_and_confirm(src_above, 40.0,
                       {"X": src_above.X, "Y": src_above.Y, "Z": src_above.Z},
                       robot.get_current_cart)
    refresh_camera(boxes, active_id, storage_pixel)

    print(f"[MOVE] Travelling to storage")
    go_to_and_confirm(dst_above, 80.0,
                       {"X": dst_above.X, "Y": dst_above.Y, "Z": dst_above.Z},
                       robot.get_current_cart)
    refresh_camera(boxes, active_id, storage_pixel)

    print(f"[MOVE] Lowering onto stack: Z={destination.Z:.0f}")
    go_to_and_confirm(destination, 20.0,
                       {"X": destination.X, "Y": destination.Y, "Z": destination.Z},
                       robot.get_current_cart)
    refresh_camera(boxes, active_id, storage_pixel)

    gripper_open()
    refresh_camera(boxes, active_id, storage_pixel)

    print(f"[MOVE] Retreating")
    go_to_and_confirm(dst_above, 40.0,
                       {"X": dst_above.X, "Y": dst_above.Y, "Z": dst_above.Z},
                       robot.get_current_cart)
    refresh_camera(boxes, active_id, storage_pixel)


def pick_and_place_box(box: Box, storage_x: float, storage_y: float,
                       stack_layer: int, boxes=None, storage_pixel=None):
    pick_z  = GRASP_Z
    place_z = GRASP_Z + stack_layer * BOX_HEIGHT

    # Apply gripper offset correction so suction cup lands centered on box
    source      = igus.Cart(box.robot_x + GRIPPER_X_OFFSET,
                            box.robot_y + GRIPPER_Y_OFFSET,
                            pick_z, *ORIENT)
    destination = igus.Cart(storage_x, storage_y, place_z, *ORIENT)

    old_pos = {"X": round(box.robot_x, 1), "Y": round(box.robot_y, 1), "Z": pick_z}
    new_pos = {"X": storage_x, "Y": storage_y, "Z": place_z}

    print(f"\n[PICK]  Box {box.box_id}  "
          f"robot=({box.robot_x:.1f}, {box.robot_y:.1f})  layer={stack_layer}")

    publish_status(json.dumps({
        "box_id":  box.box_id,
        "pos_old": old_pos,
        "pos_new": new_pos,
    }))

    _move(source, destination,
          boxes=boxes, active_id=box.box_id, storage_pixel=storage_pixel)
    print(f"[PLACE] Box {box.box_id} placed at layer {stack_layer}.")


# ============================================================================
# MAIN SORTING ROUTINE
# ============================================================================

def run_sort(storage_x: float, storage_y: float):
    print("\n" + "=" * 60)
    print(f"SORTING STARTED  |  storage: X={storage_x} Y={storage_y} mm")
    print("=" * 60)

    # Step 1: face away so camera can see boxes
    print("[SORT] Moving robot away from camera field of view...")
    go_to_and_confirm(FACE_AWAY_POSE, 40.0, FACE_AWAY_POSE.get_dict(), robot.get_current_joint)

    # Step 2: detect boxes
    print("[SORT] Detecting boxes...")
    frame = detector.grab_frame()
    boxes = detector.detect(frame)

    n = len(boxes)
    print(f"[SORT] {n} box(es) detected.")
    publish_status(f"{n} have been found!")

    if n == 0:
        print("[SORT] No boxes detected — aborting.")
        return


    # Step 3: sort nearest-first
    boxes_sorted = sorted(boxes, key=lambda b: b.distance_from_origin())
    print("[SORT] Order (nearest first):")
    for i, b in enumerate(boxes_sorted):
        print(f"  {i+1}. Box {b.box_id}  "
              f"robot=({b.robot_x:.1f}, {b.robot_y:.1f})  "
              f"dist={b.distance_from_origin():.1f} mm")

    try:
        storage_px = robot_to_pixel(storage_x, storage_y)
    except Exception:
        storage_px = None

    # Step 4: move to L pose then start picking
    L_POSE = igus.Joint(A3=90.0, A5=90.0)
    go_to_and_confirm(L_POSE, 40.0, {"A3": 90.0, "A5": 90.0}, robot.get_current_joint)
    refresh_camera(boxes, storage_pixel=storage_px)

    for layer, box in enumerate(boxes_sorted):
        # Show green on active box
        vis_frame = detector.draw(frame, boxes,
                                  active_id=box.box_id,
                                  storage_pixel=storage_px)
        cv2.imshow(WIN, vis_frame)
        cv2.waitKey(1)

        pick_and_place_box(box, storage_x, storage_y,
                          stack_layer=layer,
                          boxes=boxes,
                          storage_pixel=storage_px)

        # Refresh detection after each box
        frame = detector.grab_frame()
        boxes = detector.detect(frame)
        refresh_camera(boxes, storage_pixel=storage_px)

    # Final view
    vis_frame = detector.draw(detector.grab_frame(), [],
                              storage_pixel=storage_px)
    cv2.imshow(WIN, vis_frame)
    cv2.waitKey(1)

    publish_status("COMPLETE")
    print("\n[SORT] All boxes placed. COMPLETE.\n")

    robot.go_to_L(vel=40.0)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AUTONOMOUS SORTING SYSTEM — ready")
    print(f"Waiting for start message on  {TOPIC_SORT}")
    print(f"Payload format:  {{\"X\": <mm>, \"Y\": <mm>}}")
    print(f"Storage spot must be INSIDE the table workspace!")
    print("=" * 60 + "\n")

    try:
        while True:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord('q'), 27):
                print("[MAIN] Quit requested.")
                break
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break

            ok, frame = detector.cap.read()
            if ok:
                live_boxes = detector.detect(frame)
                vis = detector.draw(frame, live_boxes)
                cv2.imshow(WIN, vis)

            if _sort_requested.is_set():
                _sort_requested.clear()
                sx, sy = _storage_x, _storage_y
                run_sort(sx, sy)

    finally:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        robot.disconnect()
        detector.release()
        cv2.destroyAllWindows()
        print("[MAIN] Shutdown complete.")