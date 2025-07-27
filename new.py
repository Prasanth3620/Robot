from bleak import BleakClient
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Union
import cv2
import asyncio
import platform

app = FastAPI()

# UUIDs for the BLE service and characteristics
SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
CHARACTERISTIC_UUID_RX = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"

# Replace this with your ESP32's MAC address
DEVICE_ADDRESS = "AE3E8C07-584F-76DE-5522-F8ABCAFC0030"

async def send_command(command: int) -> dict:
    try:
        async with BleakClient(DEVICE_ADDRESS) as client:
            if client.is_connected:
                print(f"Connected to {DEVICE_ADDRESS}")
                command_str = str(command).encode()
                await client.write_gatt_char(CHARACTERISTIC_UUID_RX, command_str, response=True)
                print(f"Sent command: {command}")
                return {"status": "success", "command": command}
            else:
                print("Failed to connect to the device")
                return {"status": "failed", "reason": "Failed to connect"}
    except Exception as e:
        print(f"Error: {e}")
        return {"status": "failed", "reason": str(e)}

async def send_command_with_timeout(command: int, timeout: float) -> dict:
    result = await send_command(command)
    if result["status"] == "success":
        await asyncio.sleep(timeout)
        await send_command(-1)  # Stop command
    return result

@app.post("/left")
async def left(request: Request):
    data = await request.json()
    check = data.get("value")
    if check is not None:
        result = await send_command_with_timeout(2, 0.25)  # 0.25 seconds timeout
        return result
    else:
        return {"status": "failed", "reason": "value not provided"}

@app.get("/check")
async def check():
    result = await send_command_with_timeout(2, 0.25)  # 0.25 seconds timeout
    return result

@app.get("/right")
async def right():
    result = await send_command_with_timeout(3, 0.25)  # 0.25 seconds timeout
    return result

@app.get("/front")
async def front():
    result = await send_command_with_timeout(1, 0.25)  # 0.25 seconds timeout
    return result

@app.get("/down")
async def down():
    result = await send_command_with_timeout(4, 0.25)  # 0.25 seconds timeout
    return result

@app.get("/stop")
async def stop():
    result = await send_command(-1)  # Directly send the stop command
    return result

async def open_camera_and_detect_faces():
    # Load the pre-trained face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize variables
    frame_count = 0
    avg_area = 0
    direction = "Unknown"

    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Process each detected face
        for f in faces:
            if f[-1] == max(faces[:, -1]):
                x = f[0]
                y = f[1]
                w = f[2]
                h = f[3]
                break
           
            area = w * h
            avg_area = (avg_area * frame_count + area) / (frame_count + 1)

            if area > 100 * 100:
                frame_count += 1
                if frame_count == 10:
                    if x < frame.shape[1] // 4 or x + w > (3 * frame.shape[1]) // 4:
                        direction = "Left" if x < frame.shape[1] // 4 else "Right"
                        if direction == "Left":
                            await send_command(2)
                            await asyncio.sleep(0.25)
                            await send_command(-1)
                        else:
                            await send_command(3)
                            await asyncio.sleep(0.25)
                            await send_command(-1)
                        print(f"Direction: {direction}")
                    elif avg_area < 80000 and x > frame.shape[1] // 4 and x < (3 * frame.shape[1]) // 4:
                        direction = "Forward"
                        await send_command(1)
                        await asyncio.sleep(0.25)
                        await send_command(-1)
                        print("Direction: Forward")
                    elif avg_area > 80000 and x > frame.shape[1] // 4 and x < (3 * frame.shape[1]) // 4:
                        direction = "Backward"
                        await send_command(4)
                        await asyncio.sleep(0.25)
                        await send_command(-1)
                        print("Direction: Backward")
                    else:
                        direction = "Not finding humans"
                    frame_count = 0
                    avg_area = 0
            else:
                direction = "No significant face detected"
                print("Direction: No significant face detected")

        # Display the flipped frame with detected faces
        cv2.imshow('Live Camera Feed - Press "q" to exit', frame)

        # Break the loop and close the window if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

@app.get("/capture")
async def capture():
    result = await open_camera_and_detect_faces()
    return result
