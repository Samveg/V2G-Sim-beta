from __future__ import division
import os
import v2gsim.model as model
import scipy.io as sio
import scipy.integrate as integrate
import numpy


def assign_EPA_cycle(project, const_grade=0):
    """Create speed versus time profile based on UDDS, HWFT and US06 drive cycles.
    One of the cycle is assigned based on the mean speed of the driving activity. In order
    to fully match the distance traveled specified in the activity, the speed is then adjusted.

    Args:
        project (Project): a project
        const_grade (int): default 0, grade of the terrain in radian
    """
    # Load drive cycle from matlab file !! SPEED MUST BE in SECONDS !!
    data = sio.loadmat(os.path.join(os.path.dirname(__file__), "UDDS.mat"))
    UDDS = data['sch_cycle'][:, 1]
    UDDSDuration = numpy.size(UDDS, 0)

    data = sio.loadmat(os.path.join(os.path.dirname(__file__), 'HWFT.mat'))
    HWFT = data['sch_cycle'][:, 1]
    HWFTDuration = numpy.size(HWFT, 0)

    data = sio.loadmat(os.path.join(os.path.dirname(__file__), 'US06.mat'))
    US06 = data['sch_cycle'][:, 1]
    US06Duration = numpy.size(US06, 0)

    # For every vehicle and every driving cycle in their itineraries
    for vehicle in project.vehicles:
        for activity in vehicle.activities:
            if isinstance(activity, model.Driving):
                # Calculate the duration of the activity
                nb_interval = int((activity.end - activity.start).total_seconds())  # do not divide by project.timestep --> [seconds]
                duration = nb_interval / 3600  # to hours

                if duration > 0:
                    meanSpeed = activity.distance / duration
                else:
                    print('Activity duration is shorter than outputInterval')
                    # Default speed assigned
                    activity.speed = [0] * 100

                # Determine the right cycle (see simple.py / core_simple_driving_consumption for further details)
                cycle = []
                cycleDuration = 0
                if meanSpeed < 31.5:
                    # cycleName = 'UDDS'
                    cycle = UDDS
                    cycleDuration = UDDSDuration  # duration is second
                elif 31.5 <= meanSpeed <= 77.7:
                    # cycleName = 'HWFT'
                    cycle = HWFT
                    cycleDuration = HWFTDuration
                elif meanSpeed > 77.7:
                    # cycleName = 'US06'
                    cycle = US06
                    cycleDuration = US06Duration

                # append the cycle until every interval from nbInterval has speed data
                index = 0
                for i in range(0, nb_interval):
                    if index == cycleDuration - 1:
                        index = 0
                    activity.speed.append(cycle[index])
                    index += 1

                # Get the difference with the integral value
                shift = (activity.distance * 1000) / integrate.cumtrapz(y=activity.speed, dx=1, initial=0.0)[-1]

                # Add the little bit of speed for each time step
                activity.speed = [activity.speed[i] * shift for i in range(0, len(activity.speed))]

                # Specify terrain data (grade for the first and last timestamp)
                activity.terrain = [const_grade, const_grade]
