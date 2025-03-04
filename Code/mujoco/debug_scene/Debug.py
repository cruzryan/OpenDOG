class Debug:
    COLORS = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "reset": "\033[0m"
    }

    def __init__(self, name="Debug"):
        self.name = name

    def show(self, *args, color="reset"):
        color_code = self.COLORS.get(color, self.COLORS["reset"])
        print(f"{color_code}[{self.name}] Variables Debug:{self.COLORS['reset']}")
        for i, arg in enumerate(args, start=1):
            print(f"{color_code}Var {i}: {arg} -> {repr(arg)}{self.COLORS['reset']}")
