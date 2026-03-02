from import_clr import *

clr.AddReference("ManagedIR16Filters")

from Lepton import CCI
from IR16Filters import IR16Capture, NewBytesFrameEvent
import numpy as np
import time
import cv2
import pythoncom

pythoncom.CoInitialize()

lep, = (dev.Open() for dev in CCI.GetDevices())
print("Camera uptime:", lep.sys.GetCameraUpTime())

# Shared buffer updated by callback
numpyArr = None

def getFrameRaw(arr, width, height):
    global numpyArr
    # arr is an iterable of uint16 values
    numpyArr = np.fromiter(arr, dtype=np.uint16).reshape(height, width)

capture = IR16Capture()
capture.SetupGraphWithBytesCallback(NewBytesFrameEvent(getFrameRaw))
capture.RunGraph()

win = "Lepton Live"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)

try:
    while True:
        if numpyArr is None:
            time.sleep(0.01)
            continue

        frame16 = numpyArr  # (H, W) uint16

        # Optional: upsample for nicer viewing
        up = cv2.resize(frame16, (240, 240), interpolation=cv2.INTER_LINEAR)

        # Normalize 16-bit -> 8-bit for display
        disp8 = cv2.normalize(up, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # # Optional: apply a colormap (comment out if you want pure grayscale)
        # disp = cv2.applyColorMap(disp8, cv2.COLORMAP_INFERNO)
        # disp = disp8  # use this instead for grayscale

        cv2.imshow(win, disp8)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break

finally:
    capture.StopGraph()
    capture.Dispose()
    cv2.destroyAllWindows()
