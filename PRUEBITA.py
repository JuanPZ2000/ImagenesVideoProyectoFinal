import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)
GPIO.setup(24, GPIO.OUT)
rojo = GPIO.PWM(24, 100) 
rojo.start(100)
rojo.ChangeDutyCycle(100)
while True:
    for i in range(100,-1,-1):
        rojo.ChangeDutyCycle(40)
                 

    print("Ciclo completo")