from controller import Robot

# Create a Robot instance
robot = Robot()

# Get the time step of the current world
time_step = int(robot.getBasicTimeStep())

# Get the motor device
motor = robot.getDevice('joint1_motor')

# Set the motor to velocity control mode
motor.setPosition(float('inf'))  # Infinite position for velocity control
motor.setVelocity(0.0)  # Start with zero velocity

# Main control loop
while robot.step(time_step) != -1:
    # Example: Set a sinusoidal velocity
    # You can replace this with your desired control logic
    import math
    time = robot.getTime()
    velocity = 2.0 * math.sin(time)  # Adjust amplitude and frequency as needed
    motor.setVelocity(velocity)

    # Optional: Print the velocity for debugging
    print(f"Time: {time:.2f}, Velocity: {velocity:.2f}")
