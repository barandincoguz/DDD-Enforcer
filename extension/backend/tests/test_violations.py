# test_violations.py
# This code violates multiple DDD rules

class ClientManager:  # ❌ Should use "Customer" instead of "Client", "Manager" is a banned term
    def __init__(self):
        self.clients = []  # ❌ Should use "customers" instead of "clients"
    
    def add_client(self, client):  # ❌ Should use "customer" instead of "client"
        self.clients.append(client)
    
    def process_transaction(self, data):  # ❌ Should use "Order" instead of "Transaction"
        pass


class DataHelper:  # ❌ "Helper" is a banned term, "Data" is a banned term
    pass


class OrderProcessor:  # ❌ "Processor" is a banned term
    def handle_purchase(self, item):  # ❌ Should use "Order" instead of "Purchase"
        pass


class ShoppingBasket:  # ❌ Should use "Cart" instead of "Basket" (if Cart is defined in SRS)
    def __init__(self):
        self.items = []


# Correct usage examples:
class Customer:  # ✅ Correct
    def __init__(self, name):
        self.name = name


class Order:  # ✅ Correct
    def __init__(self, customer):
        self.customer = customer
