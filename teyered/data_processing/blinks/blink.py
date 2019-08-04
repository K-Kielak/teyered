

class Blink:

    def __init__(self, measurements):
        self.measurements = measurements

    def __eq__(self, other):
        return self.measurements == other.measurements

    def __repr__(self):
        return f'Blink({self.measurements})'

    def get_time_range(self):
        """
        :return Start time of the blink and end time of the blink as a tuple
        """
        return self.measurements[0][0], self.measurements[-1][0]

    def get_duration(self):
        """
        :return: The total duration of the blink
        """
        return self.measurements[-1][0] - self.measurements[0][0]
