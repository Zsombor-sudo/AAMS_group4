from irsim.world.object_base import ObjectBase

class Apple(ObjectBase):
    """An apple object that can be collected by robots."""

    def __init__(self, id, x, y, level=1, radius=0.25, color="r"):
        shape = {"name": "rectangle", "length": radius*2, "width": radius*1.75} 
        # shape = {"name": "circle", "radius": radius}
        state = [float(x), float(y), 0.0]

        super().__init__(
            id=id,
            name=f"apple_{id}",
            shape=shape,
            state=state,
            color=color,
            static=True
        )

        self.level = level
        self.collected = False

    def collect(self):
        """Marks the apple as collected."""
        self.collected = True
        self.color = "w"
