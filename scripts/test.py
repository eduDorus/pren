from time import sleep
import sys
import display

sleep(3)
print("0")
sys.stdout.flush()

sleep(20)
print("4")
display.four()
sys.stdout.flush()