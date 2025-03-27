from typing import Any, List


def hello(name: str=None) -> str:
    if name == None or name == '':
        return "Hello!"
    else:
        return 'Hello, ' + name + '!'


def int_to_roman(num: int) -> str:
    roman = ''
    roman += 'M' * (num // 1000)
    num %= 1000
    if num >= 900:
        roman += 'CM'
    elif 900 > num >= 500:
        roman += 'D' + 'C' * ((num - 500) // 100)
    elif 400 <= num < 500:
        roman += 'CD'
    elif 100 <= num < 400:
        roman += 'C' * (num // 100)        
    num %= 100
    if num >= 90:
        roman += 'XC'
    elif 50 <= num < 90:
        roman += 'L' + 'X' * ((num - 50) // 10)
    elif 40 <= num < 50:
        roman += 'XL'
    elif 10 <= num < 40:
        roman += 'X' * (num // 10) 
    num %= 10
    
    if num == 9:
        roman += 'IX'
    elif 5 <= num < 9:
        roman += 'V' + 'I' * (num - 5)
    elif num == 4:
        roman += 'IV'
    elif 1 <= num < 4:
        roman += 'I' * num
    return roman    


def longest_common_prefix(strs_input: List[str]) -> str:
    if strs_input == []:
        return ''
    m_len = len(strs_input[0]) 
    for s in strs_input:
        m_len = min(len(s.strip()), m_len)
    pref = ''
    for i in range(m_len):
        alpha = strs_input[0].strip()[i]
        for s in strs_input:
            if s.strip()[i] != alpha:
                return pref
        pref += alpha
    return pref

def is_prime(num):
    for i in range(2, int(num ** 0.5)+1):
        if num % i == 0:
            return False
    return True

def primes() -> int:
    num = 2
    while True:
        while not is_prime(num):
            num += 1
        yield num
        num += 1


class BankCard:
    def __init__(self, total_sum: int, balance_limit: int=None):
        self.total_sum = total_sum
        self.balance_limit = balance_limit if balance_limit != None else float('inf')

    def __call__(self, sum_spent):
        if self.total_sum >= sum_spent:
            self.total_sum -= sum_spent
            print("You spent %d dollars." % sum_spent)
        else:
            raise ValueError("Not enough money to spend %d dollars." % sum_spent)

    def __str__(self) -> str:
        return "To learn the balance call balance."

    @property
    def balance(self):
        if self.balance_limit > 0:
            self.balance_limit -= 1
            return self.total_sum
        else:
            raise ValueError("Balance check limits exceeded.")

    def put(self, sum_put):
        self.total_sum += sum_put
        print("You put %d dollars." % sum_put)

    def  __add__(self, other):
        return BankCard(self.total_sum + other.total_sum, max(self.balance_limit, other.balance_limit))
