

class Blink:

    def __init__(self, measurements):
        self.measurements = measurements

    def __eq__(self, other):
        return self.measurements == other.measurements

    def __repr__(self):
        return f'Blink({self.measurements})'

    def get_time_range(self):
        return self.measurements[0][0], self.measurements[-1][0]

    def get_duration(self):
        return self.measurements[-1][0] - self.measurements[0][0]
