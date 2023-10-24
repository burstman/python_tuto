import math

# execption classes


class InvalidInputError(Exception):
    pass


class InvalideOperation(Exception):
    pass


class Zero_division(Exception):
    pass

# defining the Calculator class


class Calculator:
    def __init__(self):
        self.operations = {"+": self.addition, "-": self.substract,
                           "*": self.multiply, "/": self.divide}

   # basic methode for computation
    def substract(self, num1, num2):
        return num1 - num2

    def multiply(self, num1, num2):
        return num1*num2

    def addition(self, num1, num2):
        return num1+num2

    def divide(self, num1, num2):
        return num1/num2

   # methode for adding new operation to operations dictionnarie
    def add_operation(self, symbol, func):
        self.operations[symbol] = func

    # calculate methode for cheking inputs and operation.
    # It raises an exception when devide by 0, negative sqrt root
    # and when log base less or equal to zero or 1 and the number to operate is less or equal to 0

    def calculate(self, num1, operation, num2=None):
        try:
            num1 = float(num1)
            if num2 is not None:
                num2 = float(num2)
            if not isinstance(num1, (int, float)) or (num2 is not None and not isinstance(num2, (int, float))):
                raise InvalidInputError(
                    "Invalid input. Please enter valid number. ")

            if operation in self.operations:
                if operation == "sqrt":
                    if num1 < 0:
                        raise InvalidInputError(
                            "Invalid input for the square root!")
                    return self.operations[operation](num1)  # type: ignore
                elif operation == "/":
                    if num2 == 0:
                        raise Zero_division("Error! division by zero")
                    return self.operations[operation](num1, num2)
                elif operation == "log":
                    if num1 <= 0 or (num2 is not None and (num2 <= 0 or num2 == 1)):
                        raise InvalidInputError(
                            "Error! Invalid input for logarithm.")
                    return self.operations[operation](num1, num2)
                elif num2 is not None:
                    return self.operations[operation](num1, num2)
                else:
                    raise InvalideOperation("Error! Invalid operetion symbol")
            else:
                raise InvalideOperation("Error! Invalid opretion symbol")
        except (ValueError, InvalidInputError, InvalideOperation, Zero_division) as e:
            print(f"Error: {e}")
            return None

    def exponentiation(self, num1, num2):
        return num1 ** num2

    def square_root(self, num1):
        return math.sqrt(num1)

    def logarithm(self, num1, base):
        return math.log(num1, base)


# create calc instance of Calculator
calc = Calculator()
# add special operation to the dictionary
calc.add_operation("**", calc.exponentiation)
calc.add_operation("sqrt", calc.square_root)
calc.add_operation("log", calc.logarithm)

# Infinit loop until user exit
while True:
    print(
        "Available operations:\n [+], [-], [*], [/], [**] (exponentiation), [sqrt] (square root), [log] (logarithm), [exit]\n")
    operation = input("Enter operation: ").lower()

    if operation == 'exit':
        print("Exiting calculator. Goodbye!")
        break

    if operation in calc.operations:
        if operation == "sqrt":
            num1 = input("Enter the number: ")
            result = calc.calculate(num1, operation)
        elif operation == "log":
            num1 = input("Enter the number: ")
            num2 = input("Enter the base number: ")
            result = calc.calculate(num1, operation, num2)
        else:
            num1 = input("Enter first number: ")
            num2 = input("Enter second number: ")
            result = calc.calculate(num1, operation, num2)
        if result is not None:
            print("Result: {}".format(result))
