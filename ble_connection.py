import asyncio
import bleak

# Define the UUID of the PWM service and characteristic of the target device
# PWM_SERVICE_UUID = "your_pwm_service_uuid"
HAPTIC_SERVICE_UUID = "18424398-7CBC-11E9-8F9E-2A86E4085A59"



# PWM_CHARACTERISTIC_UUID = "your_pwm_characteristic_uuid"

# Haptic Motor 1 Duty Cycle - Not connected right now
PWM1_DUTY_CYCLE_UUID = "5A87B4EF-3BFA-76A8-E642-92933C31434F"
PWM1_T1_UUID = "2D86686A-53DC-25B3-0C4A-F0E10C8DEE21"
PWM1_T2_UUID = "2D86686A-53DC-25B3-0C4A-F0E10C8DEE22"

# Haptic Motor 2 Duty Cycle
PWM2_DUTY_CYCLE_UUID = "5A87B4EF-3BFA-76A8-E642-92933C31434F"
PWM2_T1_UUID = "5A87B4EF-3BFA-76A8-E642-92933C314350" #increases active time: long it beeps for before stoping
PWM2_T2_UUID = "5A87B4EF-3BFA-76A8-E642-92933C314351" #increased frequency: how long it pauses before stoping

# Define the MAC address of the BLE device you want to connect to
# DEVICE_MAC_ADDRESS = "XX:XX:XX:XX:XX:XX"  # Replace with your device's MAC address
DEVICE_MAC_ADDRESS = "80:EA:CA:70:00:0C"  # Replace with your device's MAC address



# Define the PWM duty cycle value to send (replace with your value)
# PWM_DUTY_CYCLE = 50  # Example: 50%
PWM_DUTY_CYCLE = 95  # Example: 10%


async def send_pwm_duty_cycle():
    async with bleak.BleakClient(DEVICE_MAC_ADDRESS) as client:
        await client.connect()
        print(f"Connected to device: {DEVICE_MAC_ADDRESS}")

        # Write the PWM duty cycle value to the characteristic
        duty_cycle_bytes = PWM_DUTY_CYCLE.to_bytes(2, byteorder="little")
        await client.write_gatt_char(PWM2_DUTY_CYCLE_UUID, duty_cycle_bytes)
        print(f"Sent PWM duty cycle: {PWM_DUTY_CYCLE}%")

        # You can send more PWM duty cycle values or perform other operations here if needed

async def main():
    await send_pwm_duty_cycle()

if __name__ == "__main__":
    asyncio.run(main())