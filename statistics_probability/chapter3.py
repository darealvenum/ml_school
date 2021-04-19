class C3:
    def __init__(self, data):
        self.data = data
        self._len = len(data)

    def mean(self):
        return sum(self.data) // self._len

    # Math: \sigma = \sqrt{\sigma^2}
    # Math: \sigma^2 = \frac{\sum_{i=1}^{n} x_i - \mu}{n}
    def variance(self):
        sd = 0
        mean = self.mean()
        for i in self.data:
            sd += (i - mean) ** 2
        return sd / mean

    def standard_deviation(self):
        return self.variance() ** 0.5


c3 = C3([3, 2, 4, 7])
print(c3.standard_deviation())