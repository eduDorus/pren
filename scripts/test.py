from time import sleep
import sys
import display

sleep(3)
print("go")
print()
sys.stdout.flush()

sleep(5)
print("4")
display.eight()
sys.stdout.flush()