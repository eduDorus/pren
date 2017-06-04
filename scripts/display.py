import RPi.GPIO as GPIO

# GPIO 2 A
# GPIO 3 B
# GPIO 4 C
# GPIO 17 D
# GPIO 27 BI

# RPi.GPIO Layout verwenden (wie Pin-Nummern)
GPIO.setmode(GPIO.BCM)

# Pin 11 (GPIO 17) auf Output setzen
GPIO.setup(2, GPIO.OUT)
GPIO.setup(3, GPIO.OUT)
GPIO.setup(4, GPIO.OUT)
GPIO.setup(17, GPIO.OUT)
GPIO.setup(27, GPIO.OUT)
GPIO.output(27, GPIO.HIGH)


def zero():
    GPIO.output(2, GPIO.LOW)
    GPIO.output(3, GPIO.LOW)
    GPIO.output(4, GPIO.LOW)
    GPIO.output(17, GPIO.LOW)


def one():
    GPIO.output(2, GPIO.HIGH)
    GPIO.output(3, GPIO.LOW)
    GPIO.output(4, GPIO.LOW)
    GPIO.output(17, GPIO.LOW)


def two():
    GPIO.output(2, GPIO.LOW)
    GPIO.output(3, GPIO.HIGH)
    GPIO.output(4, GPIO.LOW)
    GPIO.output(17, GPIO.LOW)


def three():
    GPIO.output(2, GPIO.HIGH)
    GPIO.output(3, GPIO.HIGH)
    GPIO.output(4, GPIO.LOW)
    GPIO.output(17, GPIO.LOW)


def four():
    GPIO.output(2, GPIO.LOW)
    GPIO.output(3, GPIO.LOW)
    GPIO.output(4, GPIO.HIGH)
    GPIO.output(17, GPIO.LOW)


def five():
    GPIO.output(2, GPIO.HIGH)
    GPIO.output(3, GPIO.LOW)
    GPIO.output(4, GPIO.HIGH)
    GPIO.output(17, GPIO.LOW)
