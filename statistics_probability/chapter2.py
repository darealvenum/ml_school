class C2:
    def __init__(self, data):
        self.data = data
    
    def median(self):
        return self.data[len(self.data) // 2]
    
    def mean(self):
        return sum(self.data) / len(self.data)
    
    def range(self):
        return self.data[len(self.data) - 1] - self.data[0]
    

