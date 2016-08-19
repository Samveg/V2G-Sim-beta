from __future__ import division
import numpy
import v2gsim.model


def randomly_assign(activity_index, activity, vehicle):
    """Randomly assign a charging station. Charging station is then kept the
    same for the next times a vehicle park at the same location.

    Args:
        activity_index (int): current activity index from vehicle.activities
        activity (Parked): activity to be assigned with a charging station
        vehicle (Vehicle): needed to lookup charging station used previously at
            the same location

    Return:
        a charging station object to be assign to the parked activity
    """

    # If we have already visited this place then let's keep the same charger
    for previous_activity in vehicle.activities[0:activity_index]:
        if isinstance(previous_activity, v2gsim.model.Parked):
            if activity.location.category in previous_activity.location.category:
                return previous_activity.charging_station

    # Randomly decide which charger will be assigned
    return numpy.random.choice(
        activity.location.available_charging_station['charging_station'].values.tolist(),
        p=activity.location.available_charging_station['probability'].values.tolist(),
        size=1,)[0]
