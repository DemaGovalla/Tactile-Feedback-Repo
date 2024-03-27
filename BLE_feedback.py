<<<<<<< HEAD
"""
UART Service
-------------

An example showing how to write a simple program using the Nordic Semiconductor
(nRF) UART service.

"""

import asyncio, time
from bleak import BleakScanner, BleakClient
from bleak.exc import BleakError
from bleak.backends.scanner import AdvertisementData
from bleak.backends.device import BLEDevice

HAPTIC_SERVICE_UUID = "18424398-7CBC-11E9-8F9E-2A86E4085A59"

PWM1_DUTY_CYCLE_UUID = "2D86686A-53DC-25B3-0C4A-F0E10C8DEE20" # Haptic Motor 1 Duty Cycle; goes between 1 and 99
PWM1_T1_UUID = "2D86686A-53DC-25B3-0C4A-F0E10C8DEE21" # Haptic Motor 1 Active time T1; goes between 1 and 99
PWM1_T2_UUID = "2D86686A-53DC-25B3-0C4A-F0E10C8DEE22" # Haptic Motor 1 Active time T2; goes between 1 and 99

PWM2_DUTY_CYCLE_UUID = "5A87B4EF-3BFA-76A8-E642-92933C31434F" # Haptic Motor 2 Duty Cycle; goes between 1 and 99
PWM2_T1_UUID = "5A87B4EF-3BFA-76A8-E642-92933C314350" # Haptic Motor 2 Active time T1; goes between 1 and 99
PWM2_T2_UUID = "5A87B4EF-3BFA-76A8-E642-92933C314351" # Haptic Motor 2 Active time T2; goes between 1 and 99


UART_SAFE_SIZE = 20



def countdown_timer(seconds):
    for remaining in range(seconds, 0, -1):
        print(f"Time remaining: {remaining} seconds", end='\r')
        time.sleep(1)

def display_menu():
    print('Input the desired PWM signal to control, then press enter. Only PWMs 2 a& 3 are supported right now.')
    print('To exit, type \'disconnect\' or \'exit\' to safely terminate the connection.')
    print('You may also press Ctrl + C to forcefully close the program. This may cause issues when reconnecting.')
    print('To bring this menu back up again, type \'menu\' or \'help\'. Other commands or non-integers are ignored.')
    print('----------')


def edit_duty_cycle():
    print('Input the desired duty cycle, then press enter. Only integers between zero and one hundred are accepted.')


async def uart_terminal():
    """This is a simple "terminal" program that uses the Nordic Semiconductor
    (nRF) UART service. It reads from stdin and sends each line of data to the
    remote device. Any data received from the device is printed to stdout.
    """
    device = None

 
    def match_haptic_service_uuid(device: BLEDevice, adv: AdvertisementData):
        if HAPTIC_SERVICE_UUID.lower() in adv.service_uuids:
            return True

        return False
    
    print('Scanning for device...', end='')
    while device is None:
        device = await BleakScanner.find_device_by_filter(match_haptic_service_uuid)
        print('...', end='', flush=True)
    print('\b...device found. Connecting, please wait.')
    
    while True:
        try:
            async with BleakClient(device) as client:
                # time_connected = time.time()  # XXX for latency
                # file= open('connection_times.csv', 'a')
                # file.write(f'{time_connected - time_start_connect}\n')
                disconnected_event = asyncio.Event()

                def disconnect_callback(client):
                    # f = open("./connection_times.txt", 'a')
                    # f.write(f'{time_connected - time_start_connect}\n')
                    loop.call_soon_threadsafe(disconnected_event.set)
                    print("Connection lost with device.", flush=True)

                client.set_disconnected_callback(disconnect_callback)
                loop = asyncio.get_running_loop()
                display_menu()
                
                signal_and_cycle =[
                    ("1",["1","99", "99"]),
                    ("2",["33","66", "66"]),
                    ("3",["66","33", "33"]),
                    ("4",["99","1","1"]),
                    
                    ("1",["1","99", "99"]),
                    ("2",["33","66", "66"]),
                    ("3",["66","33", "33"]),
                    ("4",["99","1","1"]),
                    
                    ("1",["1","99", "99"]),
                    ("2",["33","66", "66"]),
                    ("3",["66","33", "33"]),
                    ("4",["99","1","1"])    
                ]
                
                while True:
                    
                    for signal, cycle_list in signal_and_cycle:
                        print(f'Signal to program (or command): {signal}', end='', flush=True)
                        data = signal.encode("utf-8")

                        if not data:
                            break
                       
                        data = data.strip()
                        if data.decode() in "1":
                            
                            duty_cycle = cycle_list[0].encode("utf-8")
                            time_one = cycle_list[1].encode("utf-8")
                            time_two = cycle_list[2].encode("utf-8")
                            
                            duty_cycle = int(duty_cycle.strip().decode()).to_bytes(1, 'big')
                            time_one = int(time_one.strip().decode()).to_bytes(1, 'big')
                            time_two = int(time_two.strip().decode()).to_bytes(1, 'big')
                            
                            await client.write_gatt_char(PWM1_DUTY_CYCLE_UUID, duty_cycle, response=False)
                            await client.write_gatt_char(PWM2_DUTY_CYCLE_UUID, duty_cycle, response=False)
                            
                            await client.write_gatt_char(PWM1_T1_UUID, time_one, response=False)
                            await client.write_gatt_char(PWM2_T1_UUID, time_one, response=False)
                            
                            await client.write_gatt_char(PWM1_T2_UUID, time_two, response=False)
                            await client.write_gatt_char(PWM2_T2_UUID, time_two, response=False)
                            
                            print(" Sent to haptic 1 duty cycle:", duty_cycle)
                            print(" Sent to haptic 2 duty cycle:", duty_cycle)

                            countdown_timer(4)
                            
                        elif data.decode() in "2":
                            
                            duty_cycle = cycle_list[0].encode("utf-8")
                            time_one = cycle_list[1].encode("utf-8")
                            time_two = cycle_list[2].encode("utf-8")
                            
                            duty_cycle = int(duty_cycle.strip().decode()).to_bytes(1, 'big')
                            time_one = int(time_one.strip().decode()).to_bytes(1, 'big')
                            time_two = int(time_two.strip().decode()).to_bytes(1, 'big')
                            
                            await client.write_gatt_char(PWM1_DUTY_CYCLE_UUID, duty_cycle, response=False)
                            await client.write_gatt_char(PWM2_DUTY_CYCLE_UUID, duty_cycle, response=False)
                            
                            await client.write_gatt_char(PWM1_T1_UUID, time_one, response=False)
                            await client.write_gatt_char(PWM2_T1_UUID, time_one, response=False)
                            
                            await client.write_gatt_char(PWM1_T2_UUID, time_two, response=False)
                            await client.write_gatt_char(PWM2_T2_UUID, time_two, response=False)
                            
                            print(" Sent to haptic 1 duty cycle:", duty_cycle)
                            print(" Sent to haptic 2 duty cycle:", duty_cycle)
                        
                            countdown_timer(4)
                        
                        elif data.decode() in "3":
                        
                            duty_cycle = cycle_list[0].encode("utf-8")
                            time_one = cycle_list[1].encode("utf-8")
                            time_two = cycle_list[2].encode("utf-8")
                            
                            duty_cycle = int(duty_cycle.strip().decode()).to_bytes(1, 'big')
                            time_one = int(time_one.strip().decode()).to_bytes(1, 'big')
                            time_two = int(time_two.strip().decode()).to_bytes(1, 'big')
                            
                            await client.write_gatt_char(PWM1_DUTY_CYCLE_UUID, duty_cycle, response=False)
                            await client.write_gatt_char(PWM2_DUTY_CYCLE_UUID, duty_cycle, response=False)
                            
                            await client.write_gatt_char(PWM1_T1_UUID, time_one, response=False)
                            await client.write_gatt_char(PWM2_T1_UUID, time_one, response=False)
                            
                            await client.write_gatt_char(PWM1_T2_UUID, time_two, response=False)
                            await client.write_gatt_char(PWM2_T2_UUID, time_two, response=False)
                            
                            print(" Sent to haptic 1 duty cycle:", duty_cycle)
                            print(" Sent to haptic 2 duty cycle:", duty_cycle)
                        
                            countdown_timer(4)
                        
                        elif data.decode() in "4":
                            
                            duty_cycle = cycle_list[0].encode("utf-8")
                            time_one = cycle_list[1].encode("utf-8")
                            time_two = cycle_list[2].encode("utf-8")
                            
                            duty_cycle = int(duty_cycle.strip().decode()).to_bytes(1, 'big')
                            time_one = int(time_one.strip().decode()).to_bytes(1, 'big')
                            time_two = int(time_two.strip().decode()).to_bytes(1, 'big')
                            
                            await client.write_gatt_char(PWM1_DUTY_CYCLE_UUID, duty_cycle, response=False)
                            await client.write_gatt_char(PWM2_DUTY_CYCLE_UUID, duty_cycle, response=False)
                            
                            await client.write_gatt_char(PWM1_T1_UUID, time_one, response=False)
                            await client.write_gatt_char(PWM2_T1_UUID, time_one, response=False)
                            
                            await client.write_gatt_char(PWM1_T2_UUID, time_two, response=False)
                            await client.write_gatt_char(PWM2_T2_UUID, time_two, response=False)
                            
                            print(" Sent to haptic 1 duty cycle:", duty_cycle)
                            print(" Sent to haptic 2 duty cycle:", duty_cycle)
                            
                            countdown_timer(4)

        except asyncio.exceptions.TimeoutError as e:
            print("----")
            print("Timed out when establishing connection. Retrying.")
            print("----")
        except BleakError as e:
            print("----")
            print(e)
            print('----')

if __name__ == "__main__":
    try:
        asyncio.run(uart_terminal())
    except asyncio.CancelledError:
=======
"""
UART Service
-------------

An example showing how to write a simple program using the Nordic Semiconductor
(nRF) UART service.

"""

import asyncio, time
from bleak import BleakScanner, BleakClient
from bleak.exc import BleakError
from bleak.backends.scanner import AdvertisementData
from bleak.backends.device import BLEDevice

HAPTIC_SERVICE_UUID = "18424398-7CBC-11E9-8F9E-2A86E4085A59"

PWM1_DUTY_CYCLE_UUID = "2D86686A-53DC-25B3-0C4A-F0E10C8DEE20" # Haptic Motor 1 Duty Cycle; goes between 1 and 99
PWM1_T1_UUID = "2D86686A-53DC-25B3-0C4A-F0E10C8DEE21" # Haptic Motor 1 Active time T1; goes between 1 and 99
PWM1_T2_UUID = "2D86686A-53DC-25B3-0C4A-F0E10C8DEE22" # Haptic Motor 1 Active time T2; goes between 1 and 99

PWM2_DUTY_CYCLE_UUID = "5A87B4EF-3BFA-76A8-E642-92933C31434F" # Haptic Motor 2 Duty Cycle; goes between 1 and 99
PWM2_T1_UUID = "5A87B4EF-3BFA-76A8-E642-92933C314350" # Haptic Motor 2 Active time T1; goes between 1 and 99
PWM2_T2_UUID = "5A87B4EF-3BFA-76A8-E642-92933C314351" # Haptic Motor 2 Active time T2; goes between 1 and 99


UART_SAFE_SIZE = 20



def countdown_timer(seconds):
    for remaining in range(seconds, 0, -1):
        print(f"Time remaining: {remaining} seconds", end='\r')
        time.sleep(1)

def display_menu():
    print('Input the desired PWM signal to control, then press enter. Only PWMs 2 a& 3 are supported right now.')
    print('To exit, type \'disconnect\' or \'exit\' to safely terminate the connection.')
    print('You may also press Ctrl + C to forcefully close the program. This may cause issues when reconnecting.')
    print('To bring this menu back up again, type \'menu\' or \'help\'. Other commands or non-integers are ignored.')
    print('----------')


def edit_duty_cycle():
    print('Input the desired duty cycle, then press enter. Only integers between zero and one hundred are accepted.')


async def uart_terminal():
    """This is a simple "terminal" program that uses the Nordic Semiconductor
    (nRF) UART service. It reads from stdin and sends each line of data to the
    remote device. Any data received from the device is printed to stdout.
    """
    device = None

 
    def match_haptic_service_uuid(device: BLEDevice, adv: AdvertisementData):
        if HAPTIC_SERVICE_UUID.lower() in adv.service_uuids:
            return True

        return False
    
    print('Scanning for device...', end='')
    while device is None:
        device = await BleakScanner.find_device_by_filter(match_haptic_service_uuid)
        print('...', end='', flush=True)
    print('\b...device found. Connecting, please wait.')
    
    while True:
        try:
            async with BleakClient(device) as client:
                # time_connected = time.time()  # XXX for latency
                # file= open('connection_times.csv', 'a')
                # file.write(f'{time_connected - time_start_connect}\n')
                disconnected_event = asyncio.Event()

                def disconnect_callback(client):
                    # f = open("./connection_times.txt", 'a')
                    # f.write(f'{time_connected - time_start_connect}\n')
                    loop.call_soon_threadsafe(disconnected_event.set)
                    print("Connection lost with device.", flush=True)

                client.set_disconnected_callback(disconnect_callback)
                loop = asyncio.get_running_loop()
                display_menu()
                
                signal_and_cycle =[
                    ("1",["1","99", "99"]),
                    ("2",["33","66", "66"]),
                    ("3",["66","33", "33"]),
                    ("4",["99","1","1"]),
                    
                    ("1",["1","99", "99"]),
                    ("2",["33","66", "66"]),
                    ("3",["66","33", "33"]),
                    ("4",["99","1","1"]),
                    
                    ("1",["1","99", "99"]),
                    ("2",["33","66", "66"]),
                    ("3",["66","33", "33"]),
                    ("4",["99","1","1"])    
                ]
                
                while True:
                    
                    for signal, cycle_list in signal_and_cycle:
                        print(f'Signal to program (or command): {signal}', end='', flush=True)
                        data = signal.encode("utf-8")

                        if not data:
                            break
                       
                        data = data.strip()
                        if data.decode() in "1":
                            
                            duty_cycle = cycle_list[0].encode("utf-8")
                            time_one = cycle_list[1].encode("utf-8")
                            time_two = cycle_list[2].encode("utf-8")
                            
                            duty_cycle = int(duty_cycle.strip().decode()).to_bytes(1, 'big')
                            time_one = int(time_one.strip().decode()).to_bytes(1, 'big')
                            time_two = int(time_two.strip().decode()).to_bytes(1, 'big')
                            
                            await client.write_gatt_char(PWM1_DUTY_CYCLE_UUID, duty_cycle, response=False)
                            await client.write_gatt_char(PWM2_DUTY_CYCLE_UUID, duty_cycle, response=False)
                            
                            await client.write_gatt_char(PWM1_T1_UUID, time_one, response=False)
                            await client.write_gatt_char(PWM2_T1_UUID, time_one, response=False)
                            
                            await client.write_gatt_char(PWM1_T2_UUID, time_two, response=False)
                            await client.write_gatt_char(PWM2_T2_UUID, time_two, response=False)
                            
                            print(" Sent to haptic 1 duty cycle:", duty_cycle)
                            print(" Sent to haptic 2 duty cycle:", duty_cycle)

                            countdown_timer(4)
                            
                        elif data.decode() in "2":
                            
                            duty_cycle = cycle_list[0].encode("utf-8")
                            time_one = cycle_list[1].encode("utf-8")
                            time_two = cycle_list[2].encode("utf-8")
                            
                            duty_cycle = int(duty_cycle.strip().decode()).to_bytes(1, 'big')
                            time_one = int(time_one.strip().decode()).to_bytes(1, 'big')
                            time_two = int(time_two.strip().decode()).to_bytes(1, 'big')
                            
                            await client.write_gatt_char(PWM1_DUTY_CYCLE_UUID, duty_cycle, response=False)
                            await client.write_gatt_char(PWM2_DUTY_CYCLE_UUID, duty_cycle, response=False)
                            
                            await client.write_gatt_char(PWM1_T1_UUID, time_one, response=False)
                            await client.write_gatt_char(PWM2_T1_UUID, time_one, response=False)
                            
                            await client.write_gatt_char(PWM1_T2_UUID, time_two, response=False)
                            await client.write_gatt_char(PWM2_T2_UUID, time_two, response=False)
                            
                            print(" Sent to haptic 1 duty cycle:", duty_cycle)
                            print(" Sent to haptic 2 duty cycle:", duty_cycle)
                        
                            countdown_timer(4)
                        
                        elif data.decode() in "3":
                        
                            duty_cycle = cycle_list[0].encode("utf-8")
                            time_one = cycle_list[1].encode("utf-8")
                            time_two = cycle_list[2].encode("utf-8")
                            
                            duty_cycle = int(duty_cycle.strip().decode()).to_bytes(1, 'big')
                            time_one = int(time_one.strip().decode()).to_bytes(1, 'big')
                            time_two = int(time_two.strip().decode()).to_bytes(1, 'big')
                            
                            await client.write_gatt_char(PWM1_DUTY_CYCLE_UUID, duty_cycle, response=False)
                            await client.write_gatt_char(PWM2_DUTY_CYCLE_UUID, duty_cycle, response=False)
                            
                            await client.write_gatt_char(PWM1_T1_UUID, time_one, response=False)
                            await client.write_gatt_char(PWM2_T1_UUID, time_one, response=False)
                            
                            await client.write_gatt_char(PWM1_T2_UUID, time_two, response=False)
                            await client.write_gatt_char(PWM2_T2_UUID, time_two, response=False)
                            
                            print(" Sent to haptic 1 duty cycle:", duty_cycle)
                            print(" Sent to haptic 2 duty cycle:", duty_cycle)
                        
                            countdown_timer(4)
                        
                        elif data.decode() in "4":
                            
                            duty_cycle = cycle_list[0].encode("utf-8")
                            time_one = cycle_list[1].encode("utf-8")
                            time_two = cycle_list[2].encode("utf-8")
                            
                            duty_cycle = int(duty_cycle.strip().decode()).to_bytes(1, 'big')
                            time_one = int(time_one.strip().decode()).to_bytes(1, 'big')
                            time_two = int(time_two.strip().decode()).to_bytes(1, 'big')
                            
                            await client.write_gatt_char(PWM1_DUTY_CYCLE_UUID, duty_cycle, response=False)
                            await client.write_gatt_char(PWM2_DUTY_CYCLE_UUID, duty_cycle, response=False)
                            
                            await client.write_gatt_char(PWM1_T1_UUID, time_one, response=False)
                            await client.write_gatt_char(PWM2_T1_UUID, time_one, response=False)
                            
                            await client.write_gatt_char(PWM1_T2_UUID, time_two, response=False)
                            await client.write_gatt_char(PWM2_T2_UUID, time_two, response=False)
                            
                            print(" Sent to haptic 1 duty cycle:", duty_cycle)
                            print(" Sent to haptic 2 duty cycle:", duty_cycle)
                            
                            countdown_timer(4)

        except asyncio.exceptions.TimeoutError as e:
            print("----")
            print("Timed out when establishing connection. Retrying.")
            print("----")
        except BleakError as e:
            print("----")
            print(e)
            print('----')

if __name__ == "__main__":
    try:
        asyncio.run(uart_terminal())
    except asyncio.CancelledError:
>>>>>>> b4ac7b874bf450aa87a3cda8c53f33010a2f505b
        pass