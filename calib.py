import cv2
import numpy as np
import igus

SOURCE = 1
FIXED_Z = 200.0
WIN = "IMG2CART"
TRANSFORM_FILE = "transform.npy"

ROBOT_HOST = "192.168.3.11"
ROBOT_PORT = 3920


class AimMin:
    def __init__(self, source=SOURCE):
        self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open camera source {source!r}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.mode = "calibrate"
        self.pixel_pts = []
        self.robot_pts = []
        self.transform = None
        self.frame = None
        self.aim_marker = None

        self.robot = igus.IGUS(host=ROBOT_HOST, port=ROBOT_PORT, name="IGUS REBEL (physical)")
        self.robot.connect()
        self.robot.go_to_L()

        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN, self.on_mouse)

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self.frame is None:
            return
        if self.mode == "calibrate":
            print(f"\nSelected Pixel ({x}, {y}).")
            try:
                rx = float(input("Robot X (mm): "))
                ry = float(input("Robot Y (mm): "))
            except ValueError:
                print("NaN — Point Canceled.")
                return
            self.pixel_pts.append((x, y))
            self.robot_pts.append((rx, ry))
            print(f"Point Added {len(self.pixel_pts)}. "
                  f"{'Ready to Solve (press s).' if len(self.pixel_pts) >= 6 else ''}")
        else:
            X, Y = self.pixel_to_robot(x, y)
            self.aim_marker = (x, y)
            self.robot.go_to(igus.Cart(X, Y, FIXED_Z, 180.0, 0.0, 180.0))
            print(f"[AIM] pixel=({x},{y}) -> robot X={X:.2f} Y={Y:.2f} Z={FIXED_Z:.2f} mm")

    # ---- solve ----
    def solve(self):
        n = len(self.pixel_pts)
        if n < 6:
            print(f"Need at least 6 Points (have {n}).")
            return
        src = np.array(self.pixel_pts, dtype=np.float32)
        dst = np.array(self.robot_pts, dtype=np.float32)
        M, _ = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC)
        if M is None:
            print("Error: couldn't fit a transform.")
            return
        self.transform = M
        proj = (M @ np.hstack([src, np.ones((n, 1))]).T).T
        err = np.linalg.norm(proj - dst, axis=1)
        print(f"\n[SOLVE] affine on {n} pts | "
              f"residual {err.mean():.2f} mm mean, {err.max():.2f} mm max")
        np.save(TRANSFORM_FILE, M)
        print(f"[SOLVE] Transform saved to '{TRANSFORM_FILE}' - vision.py will pick it up.")
        print("AIM mode: click a position to print its robot target.\n")
        self.mode = "aim"

    def pixel_to_robot(self, u, v):
        X, Y = self.transform @ np.array([u, v, 1.0])
        return float(X), float(Y)

    def undo(self):
        if self.mode == "calibrate" and self.pixel_pts:
            self.pixel_pts.pop();
            self.robot_pts.pop()
            print(f"Removed last point. {len(self.pixel_pts)} left.")

    def draw(self, frame):
        for i, (u, v) in enumerate(self.pixel_pts):
            cv2.drawMarker(frame, (int(u), int(v)), (45, 45, 255),
                           cv2.MARKER_CROSS, 18, 2)
            cv2.circle(frame, (int(u), int(v)), 9, (45, 45, 255), 2)
            cv2.putText(frame, str(i + 1), (int(u) + 12, int(v) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (45, 45, 255), 1, cv2.LINE_AA)
        if self.mode == "aim" and self.aim_marker is not None:
            cv2.circle(frame, self.aim_marker, 13, (106, 191, 95), 2)
            cv2.drawMarker(frame, self.aim_marker, (106, 191, 95),
                           cv2.MARKER_CROSS, 26, 2)
        label = (f"CALIBRATE  {len(self.pixel_pts)} pts  (s=solve  u=undo  q=quit)"
                 if self.mode == "calibrate"
                 else "AIM  click a position  (q=quit)")

        cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)

    def run(self):
        print("CALIBRATE: click a point, then type its robot X/Y here in the "
              "console. Place 4+ points, then press 's'.")
        while True:
            ok, frame = self.cap.read()
            if ok:
                self.frame = frame
            if self.frame is not None:
                shown = self.frame.copy()
                self.draw(shown)
                cv2.imshow(WIN, shown)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), 27):  # q or Esc
                break
            elif key == ord('s') and self.mode == "calibrate":
                self.solve()
            elif key == ord('u'):
                self.undo()
            # window closed with the X button
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
        self.cap.release()
        cv2.destroyAllWindows()

def get_sources():
    for i in range(10):  # check indices 0-9
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(i)
            cap.release()

if __name__ == "__main__":
    # get_sources()  # uncomment to list available camera indices first
    AimMin().run()
