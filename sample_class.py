from logger import Logger

logger = Logger(initial_level="DEBUG",module_name="Test Object")

class AP:
    def __init__(self, parameters):
        self.operation = parameters.get('operation','')
        self.value = parameters["numeric_info"].get('initial_value','')
        self.steps = parameters["numeric_info"].get('steps','')
        self.number = parameters["numeric_info"].get('number','')
        self.do_operation()

    def return_value(self):
        logger.info(self.value)
    
    def do_operation(self):
        if self.operation == "add":
            self._add()
        elif self.operation == "minus":
            self._minus()
        elif self.operation == "multiply":
            self._multiply()
        elif self.operation == "divide":
            self._divide()
        else:
            logger.info("Not a valid input operation type")

    def _add(self):
        for step in range(self.steps):
            self.value += self.number

    def _minus(self):
        for step in range(self.steps):
            self.value -= self.number

    def _multiply(self):
        for step in range(self.steps):
            self.value * self.number    

    def _divide(self):
        for step in range(self.steps):
            self.value / self.number


### Example Expected Usage 
# init = 6
# number = 4
# steps = 7
# operation = "add"

# 6 + (4 + 4 + 4 + 4 + 4 + 4 + 4)

parameters_1 = {
    "numeric_info": {
    "number": 4,
    "steps": 16,
    "initial_value": 6},
    "operation":"add" 
}

parameters_2 = {
    "numeric_info": {
    "number": 8,
    "steps": 20,
    "initial_value": 4},
    "operation":"add" 
}

parameters = [parameters_1, parameters_2]

for parameter in parameters:
    AnalysisPoint = AP(parameters=parameter)
    AnalysisPoint.return_value()


