from datetime import datetime

class Landscape:
    dimension = 0
    total_objects = 0
    _map = []
    _objects = {}

    def __init__(self, dimension):
        self.dimension = dimension
        self.current_date_string = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        for i in range(self.dimension+1):
            one_row = []
            for j in range(self.dimension+1):
                one_row.append([])
            self._map.append(one_row)
    
    def get_robot_object(self):
        return self._objects
    
    def _setObjectNew(self, label, x, y, speed, acceleration, heading, state, load_mass):
        self.total_objects += 1

        movement = 'vertical'
        if heading == 270 or heading == 90:
            movement = 'horizontal'

        self._objects[label] = {
            'label': label,
            'x': x,
            'y': y,
            'velocity': speed,
            'acceleration': acceleration,
            'heading': heading,
            'movement': movement,
            'state': state,
            'load_mass': load_mass,
        }

        self._map[round(x)][round(y)].append(self._objects[label])

    def setObject(self, label, x, y, speed, acceleration, heading, state, load_mass):
        if label not in self._objects:
            return self._setObjectNew(label, x, y, speed, acceleration, heading, state, load_mass)

        old_x = round(self._objects[label]['x'])
        old_y = round(self._objects[label]['y'])
        nx, ny = round(x), round(y)
        rows = len(self._map)
        cols = len(self._map[0]) if rows else 0

        # check if x or y has changed
        if nx != old_x or ny != old_y:
            # remove from old position (skip if old was already out-of-bounds)
            if 0 <= old_x < rows and 0 <= old_y < cols:
                to_iter = self._map[old_x][old_y]
                for index, e in enumerate(to_iter):
                    if e['label'] == label:
                        del to_iter[index]
                        break

            # add to new position; skip if out-of-bounds (upstream routing
            # occasionally produces invalid coordinates — survive instead of
            # crashing the simulation, log so it's visible in run output).
            if 0 <= nx < rows and 0 <= ny < cols:
                self._map[nx][ny].append(self._objects[label])
            else:
                print(f"[landscape] WARN: out-of-bounds setObject "
                      f"label={label} x={x} y={y} (rounded {nx},{ny}) "
                      f"map={rows}x{cols}; skipping map update")

        movement = 'vertical'
        if heading == 270 or heading == 90:
            movement = 'horizontal'

        self._objects[label] = {
            'label': label,
            'x': x,
            'y': y,
            'velocity': speed,
            'acceleration': acceleration,
            'heading': heading,
            'movement': movement,
            'state': state,
            'load_mass': load_mass,
        }

    def getNeighborObject(self, x, y, radius):
        i = x-radius
        j = y+radius
        check = 2*radius+1
        points_to_check = []
        result = []
        while i < x+check:
            j = y+radius
            while j > y-check:
                if 0 <= i < len(self._map) and 0 <= j < len(self._map[0]):
                    if i != x or j != y:
                        points_to_check.append([i, j])
                j -= 1
            i += 1

        for p in points_to_check:
            s = self._map[p[0]][p[1]]
            if len(s) > 0:
                for obj in s:
                    result.append(self._objects[obj['label']])

        return result

    def get_neighbor_object(self, x, y):
        s = self._map[round(x)][round(y)]
        if len(s) > 0:
            for obj in s:
                return self._objects[obj['label']]
        return None

    @property
    def objects(self):
        return self._objects

        