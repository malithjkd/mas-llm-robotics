from controller import Robot

# Initialize the robot and time step
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Initialize the wheels
wheel_names = ["front left wheel", "front right wheel", "back left wheel", "back right wheel"]
wheels = [robot.getDevice(wheel_name) for wheel_name in wheel_names]

# Set wheels to velocity mode
for wheel in wheels:
    wheel.setPosition(float('inf'))  # Set wheels to infinite position mode
    wheel.setVelocity(0.0)  # Start with zero velocity

# Function to move the robot forward
def move_forward(speed):
    """Move the robot forward at the specified speed."""
    for wheel in wheels:
        wheel.setVelocity(speed)

# Function to rotate the robot in place
def rotate_in_place(speed):
    """Rotate the robot in place. Positive speed for clockwise, negative for counter-clockwise."""
    wheels[0].setVelocity(-speed)  # Front left wheel
    wheels[1].setVelocity(speed)   # Front right wheel
    wheels[2].setVelocity(-speed)  # Rear left wheel
    wheels[3].setVelocity(speed)   # Rear right wheel

# Main control loop
while robot.step(timestep) != -1:
    # Phase 1: Move forward
    print("Moving forward...")
    move_forward(3.0)  # Adjust speed as needed
    robot.step(3000)  # Move for 3 seconds

    # Phase 2: Rotate in place
    print("Rotating...")
    rotate_in_place(2.0)  # Adjust rotation speed as needed
    robot.step(3000)  # Rotate for 3 seconds

    # Phase 3: Stop the robot
    print("Stopping...")
    move_forward(0.0)  # Stop the robot
    break
