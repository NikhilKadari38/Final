"""
This module provides a Python interface for controlling the IGUS robot.
It includes classes for representing joint positions, generating commands, and managing the connection to the robot.

It allows users to connect to the robot, send commands to move joints, and receive status updates.

It is just a basic implementation and can be extended with more features as needed.

Author: Lukasz Rojek (lukasz.rojek@srh.de)
"""
import socket
import threading
import time
import json

class Joint():
    __ATTRIBUTES = ["A1", "A2", "A3", "A4", "A5", "A6", "E1", "E2", "E3"]

    def __init__(self, *args, **kwargs):

        for attr, value in zip(Joint.__ATTRIBUTES, args):
            setattr(self, attr, float(value))

        for attr in Joint.__ATTRIBUTES[len(args):]:
            setattr(self, attr, 0.0)

        for attr in Joint.__ATTRIBUTES:
            if attr in kwargs:
                setattr(self, attr, float(kwargs[attr]))
    
    def get_dict(self) -> dict:
        return {
            "A1": self.A1,
            "A2": self.A2,
            "A3": self.A3,
            "A4": self.A4,
            "A5": self.A5,
            "A6": self.A6,
            "E1": self.E1,
            "E2": self.E2,
            "E3": self.E3
        }

    def __eq__(self, other):
        if not isinstance(other, Joint):
            return False
        
        for attr in Joint.__ATTRIBUTES:
            if getattr(self, attr) != getattr(other, attr):
                return False
        
        return True
    
    def __str__(self):
        return f"Joint(A1={self.A1}, A2={self.A2}, A3={self.A3}, A4={self.A4}, A5={self.A5}, A6={self.A6}, E1={self.E1}, E2={self.E2}, E3={self.E3})"    

class Cart():
    __ATTRIBUTES = ["X", "Y", "Z", "A", "B", "C", "E1", "E2", "E3"]

    def __init__(self, *args, **kwargs):

        for attr, value in zip(Cart.__ATTRIBUTES, args):
            setattr(self, attr, float(value))

        for attr in Cart.__ATTRIBUTES[len(args):]:
            setattr(self, attr, 0.0)

        for attr in Cart.__ATTRIBUTES:
            if attr in kwargs:
                setattr(self, attr, float(kwargs[attr]))
    
    def get_dict(self) -> dict:
        return {
            "X": self.X,
            "Y": self.Y,
            "Z": self.Z,
            "A": self.A,
            "B": self.B,
            "C": self.C,
            "E1": self.E1,
            "E2": self.E2,
            "E3": self.E3
        }

    def __eq__(self, other):
        if not isinstance(other, Cart):
            return False
        
        for attr in Cart.__ATTRIBUTES:
            if attr in ["A", "B", "C"]:
                if abs(getattr(self, attr)) != 180.0 and abs(getattr(other, attr)) != 180.0:
                    if getattr(self, attr) != getattr(other, attr):
                        return False
            elif getattr(self, attr) != getattr(other, attr):
                return False
        return True
    
    def __str__(self):
        return f"Cart(X={self.X}, Y={self.Y}, Z={self.Z}, A={self.A}, B={self.B}, C={self.C}, E1={self.E1}, E2={self.E2}, E3={self.E3})"    

class CommandID():
    """
    This class is used to generate unique command IDs for each command sent to the robot.
    It starts from 1 and increments by 1 for each call of the function get_id(). If the ID exceeds 9999, it start again from 1.
    This ensures that each command has a unique identifier, which is important for tracking and managing commands in the robot's control system.
    """

    def __init__(self, start_id: int = 1):
        if not isinstance(start_id, int):
            raise TypeError(f"start_id must be an integer, got {type(start_id).__name__} instead.")
        if start_id < 1 or start_id > 9999:
            raise ValueError(f"start_id must be between 1 and 9999, got {start_id} instead.")
        self.__id = start_id
    
    def get_id(self):
        """
        Returns the current command ID and increments it for the next call.
        """
        self.__increase()
        return self.__id
    
    def __increase(self):
        """
        Increases the command ID by 1, wrapping around to 0 if it exceeds 9999.
        """
        if self.__id >= 9999:
            self.__id = 0
        else:
            self.__id += 1

class Command():
    __command_id = CommandID()
    
    @staticmethod
    def move_joint(A1: float | int = 0.0, A2:float | int = 0.0, A3: float | int = 0.0, A4: float | int = 0.0, A5: float | int = 0.0, A6: float | int = 0.0, E1: float | int = 0.0, E2: float | int = 0.0, E3: float | int = 0.0, vel : float | int = 50.0, joint: Joint | None = None):
        """
        Generates the move joint CRI command with respect to the joint parameters.
        """
        if joint is not None:
            return Command.move_joint(**joint.get_dict())
        if not all(isinstance(arg, (float, int)) for arg in [A1, A2, A3, A4, A5, A6, E1, E2, E3]):
            raise TypeError("All joint parameters must be of type float or int.")
        
        if not isinstance(vel, (float, int)):
            raise TypeError("Velocity must be of type float or int.")
        
        return f"CRISTART {Command.__command_id.get_id()} CMD Move Joint {A1} {A2} {A3} {A4} {A5} {A6} {E1} {E2} {E3} {vel} CRIEND"
    
    @staticmethod
    def move_zero(vel: float | int = 50):
        """
        Generates the move zero CRI command.
        """
        if not isinstance(vel, (float, int)):
            raise TypeError("Velocity must be of type float or int.")
        
        return Command.move_joint(vel=vel)

    @staticmethod
    def move_L(vel: float | int = 50.0):
        """
        Generates the move L CRI command.
        """
        if not isinstance(vel, (float, int)):
            raise TypeError("Velocity must be of type float or int.")
        
        return Command.move_joint(A3=90.0, A5=90.0, vel=vel)
    
    @staticmethod
    def move_cart(X: float | int = 0.0, Y:float | int = 0.0, Z: float | int = 0.0, A: float | int = 0.0, B: float | int = 0.0, C: float | int = 0.0, E1: float | int = 0.0, E2: float | int = 0.0, E3: float | int = 0.0, vel : float | int = 50.0, cart: Cart | None = None):
        """
        Generates the move joint CRI command with respect to the joint parameters.
        """
        if cart is not None:
            return Command.move_cart(**cart.get_dict())
        if not all(isinstance(arg, (float, int)) for arg in [X, Y, Z, A, B, C, E1, E2, E3]):
            raise TypeError("All joint parameters must be of type float or int.")
        
        if not isinstance(vel, (float, int)):
            raise TypeError("Velocity must be of type float or int.")
        
        return f"CRISTART {Command.__command_id.get_id()} CMD Move Cart {X} {Y} {Z} {A} {B} {C} {E1} {E2} {E3} {vel} CRIEND"
    
    @staticmethod
    def alive_jog():
        return f"CRISTART {Command.__command_id.get_id()} ALIVEJOG 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 CRIEND"
    
    @staticmethod
    def connect():
        return f"CRISTART {Command.__command_id.get_id()} CMD Connect CRIEND"

    @staticmethod
    def enable():
        return f"CRISTART {Command.__command_id.get_id()} CMD Enable CRIEND"

    @staticmethod
    def disconnect():
        return f"CRISTART {Command.__command_id.get_id()} CMD Disconnect CRIEND"

    @staticmethod
    def active_multiple_clients():
        return f"CRISTART {Command.__command_id.get_id()} CMD SetActive true CRIEND"

    @staticmethod
    def dout(channel: int, state: bool):
        state_str = "true" if state else "false"
        return f"CRISTART {Command.__command_id.get_id()} CMD DOUT {channel} {state_str} CRIEND"

class IGUS(object):

    def __init__(self, host: str = "localhost", port: int = 3920, name="IGUS"):
        self.__host = host
        self.__port = port
        self.__sock = None
        self.__current_joint = None
        self.__set_joint = None
        self.__current_cart = None
        self.__set_cart = None
        self.name = name
        self.wait = False
        self.callback_read_msg = None
        self.callback_cnt_msg = None
        self.__send_lock = threading.Lock()

    def setopts(self, **kwargs):
        if "host" in kwargs:
            self.__host = kwargs["host"]
        if "port" in kwargs:
            self.__port = kwargs["port"]

    def connect(self):
        """
        Connects to the IGUS robot at the specified host and port.
        """

        self.__sock = socket.create_connection((self.__host, self.__port))

        # initializing communication part
        print(f"{self.name}: initializing communication workers")
        self.__keep_alive()
        self.__keep_reading()
        time.sleep(1)

        # initializing connection
        print(f"{self.name}: initializing connection")
        self.send(Command.active_multiple_clients())
        self.send(Command.connect())
        self.send(Command.enable())
        time.sleep(3)
        print(f"{self.name}: robot is ready")

        while not self.__current_joint and not self.__current_cart:
            time.sleep(0.01)

    def disconnect(self):
        """
        Disconnects from the IGUS robot.
        """
        if self.__keep_alive:
            self.__keep_alive = False

        if self.__keep_reading:
            self.__keep_reading = False

        if self.__sock:
            self.send(Command.disconnect())
            self.__sock.close()
            self.__sock = None
    
    def send(self, command: str):
        """
        Sends a command to the IGUS robot.
        """
        if not self.__sock:
            raise ConnectionError("Socket is not connected. Please connect first.")

        with self.__send_lock:
            self.__sock.sendall(command.encode('utf-8'))
        time.sleep(0.1)

    def go_to(self, pos: Joint | Cart, vel=50.0):
        """
        Moves the IGUS robot to the specified joint position.
        """
        if not isinstance(pos, (Joint, Cart)):
            raise TypeError(f"joint must be an instance of Joint or Cart class, {type(pos)} given instead!")
        
        if isinstance(pos, Joint):
            command = Command.move_joint(**pos.get_dict(), vel=vel)
            self.send(command)
            self.__set_joint = pos
            
            if self.wait:
                while self.__set_joint != self.__current_joint:
                    time.sleep(0.01)
        
        if isinstance(pos, Cart):
            command = Command.move_cart(**pos.get_dict(), vel=vel)
            self.send(command)
            self.__set_cart = pos
            
            if self.wait:
                while self.__set_cart != self.__current_cart:
                    time.sleep(0.01)

    def go_to_zero(self, vel=50.0):
        """
        Moves the IGUS robot to the zero position.
        """
        self.go_to(Joint(), vel=vel)
        #command = Command.move_zero(vel=vel)
        #self.send(command)
        #self.__set_joint = Joint()

    def go_to_L(self, vel=50.0):
        """
        Moves the IGUS robot to the L position.
        """
        self.go_to(Joint(A3=90.0, A5=90.0), vel=vel)
        #command = Command.move_L(vel=vel)
        #self.send(command)
        #self.__set_joint = Joint(A3=90.0, A5=90.0)

    def __update_status(self, status: str):
        """
        Updates the current joint and set joint positions based on the status message received from the IGUS robot.
        """
        data = status.split(" ")

        """
        if "POSJOINTSETPOINT" in data and not self.__set_joint:
            idx = data.index("POSJOINTSETPOINT") + 1
            self.__set_joint = Joint(*data[idx:idx + 9])
        """

        if "POSJOINTCURRENT" in data:
            idx = data.index("POSJOINTCURRENT") + 1
            self.__current_joint = Joint(*data[idx:idx + 9])

            if not self.callback_read_msg is None:
                self.callback_read_msg(self.__current_joint)

        if "POSCARTROBOT" in data:
            idx = data.index("POSCARTROBOT") + 1
            self.__current_cart = Cart(*data[idx:idx + 6])

            if not self.callback_read_msg is None:
                self.callback_read_msg(self.__current_cart)

    def __keep_reading(self):
        """
        Continuously reads messages from the IGUS robot and calls the provided callback function with the response.
        """
        if not self.__sock:
            raise ConnectionError("Socket is not connected. Please connect first.")
        
        def read_thread():
            buff = ""
            while True:
                try:
                    buff = self.__sock.recv(1024).decode("utf-8")
                
                    data_block  = buff[buff.find("CRISTART") : buff.rfind("CRIEND") + 6]
                    #buff = buff[buff.find("CRIEND") + 5 : ]
                
                    self.__update_status(data_block)
                except ConnectionError as e:
                    break
        
        thread = threading.Thread(target=read_thread)
        thread.daemon = True
        thread.start()
    
    def __keep_alive(self, interval: float = 0.1):
        """
        Keeps the connection alive by sending a keep-alive message at regular intervals.
        """
        if not self.__sock:
            raise ConnectionError("Socket is not connected. Please connect first.")
        
        def keep_alive_thread():
            while True:
                try:
                    self.send(Command.alive_jog())
                    time.sleep(interval)
                except ConnectionError as e:
                    break
        thread = threading.Thread(target=keep_alive_thread)
        thread.daemon = True
        thread.start()

    def get_current_joint(self):
        return self.__current_joint

    def get_current_cart(self):
        return self.__current_cart

    def read_file(self, filename: str):
        """
        Reads a file and sends its contents to the IGUS robot.
        """
        if not self.__sock:
            raise ConnectionError("Socket is not connected. Please connect first.")

        with open(filename, "r") as f:
            instructions = json.loads(f.read())["instructions"]
            for idx, instruction in enumerate(instructions):
                type = instruction["type"]
                param = instruction["pos"]
                print(f"executing {idx + 1} step: {param}")
                if type == "joint":
                    self.go_to(Joint(**param))
                if type == "base":
                    self.go_to(Cart(**param))
                


if __name__ == "__main__":
    igus = IGUS()
    igus.wait = True
    try:
        igus.connect()
        #igus.read_file("instructions.json")
        #igus.go_to_zero(vel=100.0)
        igus.go_to_L(vel=100.0)
        igus.go_to(Cart(X=300.0, Z=250.0, A=180.0, B=0.0, C=180.0), vel=100.0)
        igus.go_to(Cart(X=350.0, Z=250.0, A=180.0, B=0.0, C=180.0), vel=100.0)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        igus.disconnect()
        print("Disconnected from IGUS robot.")