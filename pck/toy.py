#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 15:24:55 2025

@author: niyashao
"""

class animal:
    
    def do_something(func,name,lan,rand):
        
        func[0](name,lan)
        print('the input number is ', rand)
        
        func[1](name,lan)
        
    def eat(name,lan):
        print(name, ' is eating, ', lan, '!')
        
        
    def smile(name,lan):
        print(name, ' is smiling, ', lan, '!')
        

animal.do_something([animal.eat,animal.smile],'dog','hof',3.14159)


def add(a, b):
    return a + b

def operate_on(x, y, func):
    return func(x, y)

result = operate_on(3, 4, add)
print(result)  # Output: 7