# -*- coding: utf-8 -*-
with open('poem_5000.txt', 'w', encoding='utf-8') as f1:
    with open('poem_58k_theme.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(5000):
            f1.write(lines[i])
            
    