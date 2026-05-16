class PointsBank:
    def __init__(self):

        self.available_points = {
            1:[14, 14, 13, 13, 13, 12, 12, 12, 12],
            2:[17, 16, 16, 14, 14, 13, 13],
            3:[19, 18, 18, 17, 17],          
            4:[22,21,20]
        }

    def claim_points(self, soil_richness):
        if len(self.available_points[soil_richness]) > 0:
            return self.available_points[soil_richness].pop(0)
        else:
            return 0
        
    @property
    def remaining_points(self):
        remaining_points = {}
        for richness_level in self.available_points:
            remaining_points[richness_level] = tuple(self.available_points[richness_level])
        
        return remaining_points