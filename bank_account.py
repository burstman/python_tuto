class Account:
    def __init__(self, account_number, account_holder, account_balance=0.0):
        self.account_number = account_number
        self.account_balance = account_balance
        self.account_holder = account_holder

    def deposit(self, amount):
        if amount > 0:
            self.account_balance += amount
            return f"Deposit of {amount} successfully processed. New balance: {self.account_balance}"
        else:
            return "Invalid deposit amount. Please enter a positive value."

    def withdraw(self, amount):
        if self.account_balance >= amount:
            self.account_balance -= amount
            return f"Withdrow of {amount} successfully processed. New balance: {self.account_balance}"
        else:
            return "Impossible withdrow. Amount is greater than the balance"

    def check_balance(self):
        return f"Current account {self.account_holder} balance: {self.account_balance}"


my_account = Account(account_number=123456789, account_holder="Flissi Hamed")
new_account = Account(account_number=987456123,
                      account_holder="John Doe", account_balance=500)

print(my_account.deposit(200.0))
print(my_account.check_balance())
print(my_account.withdraw(50.0))
print(my_account.check_balance())

print(new_account.deposit(1000.0))
print(new_account.check_balance())
print(new_account.withdraw(700.0))
print(new_account.check_balance())
