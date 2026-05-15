#include <webots/Robot.hpp>

// All the webots classes are defined in the "webots" namespace
using namespace webots;


  // create the Robot instance.
  Robot *robot = new Robot();

  // get the time step of the current world.
  int timeStep = (int)robot->getBasicTimeStep();

  while (robot->step(timeStep) != -1) {

  };

  // Enter here exit cleanup code.

  delete robot;
  return 0;
}
