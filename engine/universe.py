class Universe:
    _tick = 0
    id = 0
    tick_to_second = 0.5
    _objects = []
    landscape = None
    graph = None
    graph_pod = None
    deadlock_prevention_manager = None
    warehouse_size = []

    def addObject(self, object):
        object.id = len(self._objects)
        object.setUniverse(self)
        self._objects.append(object)
    
    def set_warehouse_size(self, size):
        self.warehouse_size = size

    def get_warehouse_size(self):
        return self.warehouse_size

    def tick(self):
        for o in self._objects:
            o.move()

    def get_movable_objects(self):
        return self._objects
    
    def generateResult(self):
        result = []
        for o in self.get_movable_objects():
            entry = {
                'id': o.id,                      # Index 0 di NetLogo
                'heading': o.heading,            # Index 1
                'shape': o.shape,                # Index 2
                'velocity': o.velocity,          # Index 3
                'acceleration': o.acceleration,  # Index 4
                'pos_x': o.pos_x,                # Index 5
                'pos_y': o.pos_y,                # Index 6
                'color': o.color,                # Index 7
                # Index 8 di NetLogo (Selalu dikirim, kalau bukan robot, anggap 100)
                'battery_pct': getattr(o, 'battery_pct', 100.0) 
            }
            result.append(entry)

        return result

