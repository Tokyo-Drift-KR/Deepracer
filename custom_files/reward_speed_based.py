def reward_function(params):
    """
    reward is accumulated only with speed
    """
    speed = float(params["speed"])
    return speed
