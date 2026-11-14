# Demo venue checklist

One page. Do it in order, then never touch the camera again.

## Setup

- [ ] **Camera on a stand**, fixed, about 20–40 cm from the laptop screen.
- [ ] Camera slightly off-axis (not square-on) to avoid reflections.
- [ ] The whole screen is in view.
- [ ] Camera is never moved after calibration. Tape the stand down.
- [ ] **Screen brightness 50–70%.**
- [ ] Night Shift off, True Tone off.
- [ ] **Grey background** (the slideshow uses bg 128).
- [ ] Demo images full-screen at full screen height: `demo/bee_with_mite_large.png`, `demo/bee_no_mite_large.png`.
- [ ] **Room light steady.** No blue reflections on the screen (they washed out the bees in our first test). Check for stage lights and projector light hitting the laptop.
- [ ] **Camera resolution:** our recordings used 1920×1080. The App Lab app captures 1280×720 (set in `main.py`).
- [ ] **Right camera index:** `make record-cameras` lists the cameras and saves a test picture from each. On the board the app uses `CAMERA` (default `usb:0`).
- [ ] Board connected to the Mac over USB; board at `169.254.100.2`, Mac at `169.254.125.210` in our setup (use your board's and laptop's IPs).
- [ ] Receiver running on the Mac: `demo/start-receiver.sh`.
- [ ] Backup video of the board working, ready to play.

## First-5-minutes smoke test

1. [ ] App page loads: `http://169.254.100.2:7000`. Camera image is live and sharp.
2. [ ] Show `demo/bee_no_mite_large.png`: verdict clean, LED matrix shows the tick.
3. [ ] Show `demo/bee_with_mite_large.png`: verdict MITE, red mite box, LED count goes up.
4. [ ] Reset the count in the page.
5. [ ] Multi-bee slideshow: bees counted, no obvious false mites.
6. [ ] Dashboard at `http://127.0.0.1:8000` shows the new events.
7. [ ] Frame rate looks smooth (check the status line in the app log).

## Troubleshooting

| Problem | Check / fix |
|---|---|
| Camera not opening | Wrong index. The app tries to find the USB camera; set `CAMERA` (e.g. `usb:0`, `/dev/video2`) and restart. Replug the camera; use a powered USB-C hub. |
| Page not loading | Board IP is `169.254.100.2`, port `7000`. Is the app running in App Lab? Is the USB link up? |
| No detections / everything clean | Exposure (screen too bright or dark), focus, distance, bee size on screen. Bees shorter than about 100 px are counted but not checked. Move the screen image, not the camera. |
| False mites | Reflections or glare on the screen. Dim room light, change the camera angle slightly, then recalibrate. |
| Dashboard not updating | Receiver not running (`demo/start-receiver.sh`); clock not trusted on the board; token mismatch between `demo/ardubeeno.env` and `demo/receiver.env`; macOS firewall blocking port 8000. Test from the board: `curl http://169.254.125.210:8000/health`. |
| Slow frame rate | Stop other processes on the board. `THREADS` = 2, never 4. |

## If the board fails

1. Do not debug on stage.
2. Play the backup video of the board working.
3. Show the dashboard with the recorded data.
4. Use the 30-second fallback in [demo-script.md](demo-script.md).
