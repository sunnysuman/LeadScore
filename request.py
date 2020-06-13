# -*- coding: utf-8 -*-
"""
Created on Mon May 11 17:54:43 2020

@author: Arpita
"""

import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'experience':2, 'test_score':9, 'interview_score':6})

print(r.json())