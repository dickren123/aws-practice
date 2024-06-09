import json
from typing import Optional
import requests
from dataclasses import dataclass


def format_params(parameters):
    result = {}
    for param in parameters:
        result[param['name']]=param['value']
    return result
    
    
def lambda_handler(event,context):
    #Event: {'agent': {'alias': 'TSTALIASID', 'name': 'GetWeather', 'version': 'DRAFT', 'id': '68X4FYOEUN'}, 
    #   'sessionId': '022346938362354', 'sessionAttributes': {}, 'promptSessionAttributes': {}, 'inputText': 'beijing', 
    #   'apiPath': '/get_lat_long', 'httpMethod': 'GET', 'messageVersion': '1.0', 'actionGroup': 'get_weather', 
    #   'parameters': [{'name': 'place', 'type': 'string', 'value': 'beijing'}]}
    params = format_params(event['parameters'])
    print(params)
    if event['apiPath'] == '/get_lat_long':
        
        url = "https://nominatim.openstreetmap.org/search"
        params = {'q': params.get('place'), 'format': 'json', 'limit': 1}
        print(params)
        response = requests.get(url, params=params,headers={'referer': "indie-test"})
        print(response.json())
        response = response.json()

        if response:
            lat = response[0]["lat"]
            lon = response[0]["lon"]
            result = {"latitude": lat, "longitude": lon}
        else:
            result = None
        return {
            "messageVersion": "1.0",
            "response": {
                "actionGroup": event['actionGroup'],
                "apiPath": event['apiPath'],
                "httpMethod": event['httpMethod'],
                "httpStatusCode": 200,
                "responseBody": {
                    "application/json": {
                        "body": json.dumps(result)
                    }
                }
            }
        }
    elif event['apiPath'] == '/get_weather':
        latitude = params.get('latitude')
        longitude = params.get('longitude')
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
        response = requests.get(url)
        return {
            "messageVersion": "1.0",
            "response": {
                "actionGroup": event['actionGroup'],
                "apiPath": event['apiPath'],
                "httpMethod": event['httpMethod'],
                "httpStatusCode": 200,
                "responseBody": {
                    "application/json": {
                        "body": response.json()
                    }
                }
            }
        }
    else:
        return {
            "messageVersion": "1.0",
            "response": {
                "actionGroup": event['actionGroup'],
                "apiPath": event['apiPath'],
                "httpMethod": event['httpMethod'],
                "httpStatusCode": 404,
                "responseBody": {
                    "application/json": {
                        "body": '{"err":"no such function"}'
                    }
                }
            }
        }