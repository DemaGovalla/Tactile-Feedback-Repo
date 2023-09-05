"""
UART Service
-------------

An example showing how to write a simple program using the Nordic Semiconductor
(nRF) UART service.

Adapted from ble_haptic_control.py from Kevin Kasper
"""

import asyncio
import sys
import datetime as dt

import time
import os

from bleak import BleakScanner, BleakClient, discover
from bleak.exc import BleakError
from bleak.backends.scanner import AdvertisementData
from bleak.backends.device import BLEDevice
# HAPTIC_SERVICE_UUID = \
#    "{0x59, 0x5a, 0x08, 0xe4, 0x86, 0x2a, 0x9e, 0x8f, 0xe9, 0x11, 0xbc, 0x7c, 0x98, 0x43, 0x42, 0x18}"

HAPTIC_SERVICE_UUID = "18424398-7CBC-11E9-8F9E-2A86E4085A59"
PWM2_DUTY_CYCLE_UUID = "2D86686A-53DC-25B3-0C4A-F0E10C8DEE20"
PWM2_T1_UUID = "2D86686A-53DC-25B3-0C4A-F0E10C8DEE21"
PWM2_T2_UUID = "2D86686A-53DC-25B3-0C4A-F0E10C8DEE22"

PWM3_DUTY_CYCLE_UUID = "5A87B4EF-3BFA-76A8-E642-92933C31434F"
PWM3_T1_UUID = "5A87B4EF-3BFA-76A8-E642-92933C314350"
PWM3_T2_UUID = "5A87B4EF-3BFA-76A8-E642-92933C314351"

# All BLE devices have MTU of at least 23. Subtracting 3 bytes overhead, we can
# safely send 20 bytes at a time to any device supporting this service.
UART_SAFE_SIZE = 20


def display_menu():
    print('Input the desired PWM signal to control, then press enter. Only PWMs 2 a& 3 are supported right now.')
    print('To exit, type \'disconnect\' or \'exit\' to safely terminate the connection.')
    print('You may also press Ctrl + C to forcefully close the program. This may cause issues when reconnecting.')
    print('To bring this menu back up again, type \'menu\' or \'help\'. Other commands or non-integers are ignored.')
    print('----------')


# def edit_duty_cycle():
#     print('Input the desired duty cycle, then press enter. Only integers between zero and one hundred are accepted.')


def status_disp():
    if not hasattr(status_disp, "stat"):
        status_disp.stat = 0

    chars = ['-', '\\', '|', '/']
    print("\b" + chars[status_disp.stat], end='', flush=True)

    status_disp.stat = (status_disp.stat + 1) % 4


def gait_notification_handler(sender, data):

    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)
    print("##############################################################", flush=True)


async def uart_terminal():
    """This is a simple "terminal" program that uses the Nordic Semiconductor
    (nRF) UART service. It reads from stdin and sends each line of data to the
    remote device. Any data received from the device is printed to stdout.
    """
    device = None

    # def match_nus_uuid(device: BLEDevice, adv: AdvertisementData):
    #     # This assumes that the device includes the UART service UUID in the
    #     # advertising data. This test may need to be adjusted depending on the
    #     # actual advertising data supplied by the device.
    #     if UART_SERVICE_UUID.lower() in adv.service_uuids:
    #         return True
    #
    #     return False

    def match_haptic_service_uuid(device: BLEDevice, adv: AdvertisementData):
        # print(adv.__repr__, flush=True)
        # This assumes that the device includes the UART service UUID in the
        # advertising data. This test may need to be adjusted depending on the
        # actual advertising data supplied by the device.
        if HAPTIC_SERVICE_UUID.lower() in adv.service_uuids:
            return True

        return False

    print('Scanning for device...', end='')

    # time_start_connect = time.time()  # XXX for latency

    while device is None:
        device = await BleakScanner.find_device_by_filter(match_haptic_service_uuid)
        print('...', end='', flush=True)

    print('\b...device found. Connecting, please wait.')
    while True:
        try:
            async with BleakClient(device) as client:
                # time_connected = time.time()  # XXX for latency
                disconnected_event = asyncio.Event()

                def disconnect_callback(client):
                    # f = open("./connection_times.txt", 'a')
                    # f.write(f'{time_connected - time_start_connect}\n')
                    loop.call_soon_threadsafe(disconnected_event.set)
                    print("Connection lost with device.", flush=True)

                client.set_disconnected_callback(disconnect_callback)
                await client.start_notify('2c86686a-53dc-25b3-0c4a-f0e10c8d9e26', gait_notification_handler)

                loop = asyncio.get_running_loop()
                # display_menu()
                services = await client.get_services()
                while True:
                    # This waits until you type a line and press ENTER.
                    # A real terminal program might put stdin in raw mode so that things
                    # like CTRL+C get passed to the remote device.
                    # print('Signal to program (or command): ', end='', flush=True)
                    data = await loop.run_in_executor(None, sys.stdin.buffer.readline)

                    # data will be empty on EOF (e.g. CTRL+D on *nix)
                    if not data:
                        break

                    # for ble_char in services.characteristics:
                    #     print(services.get_characteristic(ble_char))
                    #     # print("{} (Handle: {}): {}", ble_char.service_uuid,
                    #     #       ble_char.service_handle, ble_char.description)

                    # some devices, like devices running MicroPython, expect Windows
                    # line endings (uncomment line below if needed)
                    # data = data.replace(b"\n", b"\r\n")
                    data = data.strip()
                    if data.decode() in "2":
                        # print('Desired duty cycle: ', end='', flush=True)
                        # duty = await loop.run_in_executor(None, sys.stdin.buffer.readline)
                        # duty = int(duty.strip().decode()).to_bytes(1, 'big')
                        # if duty > 100:
                        #     duty = 100
                        # t1 = dt.datetime.now()
                        # print("#", flush=True)
                        # t2 = dt.datetime.now()
                        response = await client.write_gatt_char(PWM2_DUTY_CYCLE_UUID, b"\x46", response=True)
                        # t3 = dt.datetime.now()
                        # print(response)

                        # print(f"start: {t1}. post-log: {t2}, post-write: {t3}")
                        # print(
                        #     f"delta_screen: {t2-t1}. delta_write: {t3-t2}, total: {t3-t1}")
                        # for x in range(1, 30):

                        await client.write_gatt_char(PWM2_T1_UUID, b"\xA1\x00", response=True)

                        await client.write_gatt_char(PWM2_T2_UUID, b"\x01\x01", response=True)

                        # await client.write_gatt_char(PWM2_DUTY_CYCLE_UUID, duty)
                        # print("Sent to haptic 0:", duty)

                    elif data.decode() in "3":
                        # print('Desired duty cycle: ', end='', flush=True)
                        # duty = await loop.run_in_executor(None, sys.stdin.buffer.readline)
                        # duty = int(duty.strip().decode()).to_bytes(1, 'big')
                        # if duty > 100:
                        #     duty = 100
                        await client.write_gatt_char(PWM3_DUTY_CYCLE_UUID, b"\x46", response=True)
                        # for x in range(1, 30):
                        #     print(
                        #         "###############################################################", flush=True)
                        await asyncio.sleep(1.0)
                        await client.write_gatt_char(PWM3_T1_UUID, b"\xA1\x00", response=True)
                        await asyncio.sleep(1.0)
                        await client.write_gatt_char(PWM3_T2_UUID, b"\x01\x01", response=True)
                        await asyncio.sleep(1.0)
                    elif data.decode() in "0":
                        # print('Desired duty cycle: ', end='', flush=True)
                        # duty = await loop.run_in_executor(None, sys.stdin.buffer.readline)
                        # duty = int(duty.strip().decode()).to_bytes(1, 'big')
                        # if duty > 100:
                        #     duty = 100
                        await client.write_gatt_char(PWM2_DUTY_CYCLE_UUID, b"\x00", response=True)
                        await asyncio.sleep(1.0)
                        await client.write_gatt_char(PWM2_T1_UUID, b"\x00\x00", response=True)
                        await asyncio.sleep(1.0)
                        await client.write_gatt_char(PWM2_T2_UUID, b"\x01\x00", response=True)
                        await asyncio.sleep(1.0)
                        # await client.write_gatt_char(PWM3_DUTY_CYCLE_UUID, duty)
                        # print("Sent to haptic 1:", duty)

                    elif data.decode().lower() in {"disconnect", "quit", "exit", "q"}:
                        await client.disconnect()
                        sys.exit('Connection terminated by user.')

                    elif data.decode().lower() in {"?", "menu", "help"}:
                        display_menu()

        except asyncio.exceptions.TimeoutError as e:
            print("----")
            print("Timed out when establishing connection. Retrying.")
            print("----")
        except BleakError as e:
            print("----")
            print(e)
            print('----')


if __name__ == "__main__":
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    print(flush=True)
    try:
        asyncio.run(uart_terminal())
    except asyncio.CancelledError:
        # task is cancelled on disconnect, so we ignore this error
        pass
