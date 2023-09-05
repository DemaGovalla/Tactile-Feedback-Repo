# About

This project is used to send commands to a DA14585-based haptic feedback controller. 
The user is able to interact via the commandline and send duty cycles from 0 - 100 as unsigned integers.


## Install Instructions
1. Ensure you have the C++ build tools for your OS. Download them here under Tools for Visual Studio.
    - https://visualstudio.microsoft.com/downloads/
2. Clone this repository somewhere onto your system, open the new project directory, and create & activate a virtualenv.
    - `pip install virtualenv`
    - Create the environemntt: `virtualenv venv`
        - Activate venv On Windows: `.\venv\Scripts\activate`
        - Activate on Linux: `source /venv/Scripts/activate`
4. Run command `pip install -r requirements.txt` to install all dependencies.
