"""Temperature schedule for exploration. Accepts config params as **kwargs (sklearn-style)."""


class TemperatureConfig:
    """Temperature schedule: initial value and per-step decay.

    Accepts config params as **kwargs: initial, decay.
    """

    def __init__(
        self,
        initial: float = 1.0,
        decay: float = 1.0,
        **kwargs: object,
    ) -> None:
        self.initial = initial
        self.decay = decay
