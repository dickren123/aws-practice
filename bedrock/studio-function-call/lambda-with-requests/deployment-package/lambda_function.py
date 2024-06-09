import json
from typing import Optional
import requests
from dataclasses import dataclass

weather_codes = {
    0: 'Clear',
    1: 'Mostly Clear',
    2: 'Partly Cloudy',
    3: 'Cloudy',
    45: 'Fog',
    48: 'Freezing Fog',
    51: 'Light Drizzle',
    53: 'Drizzle',
    55: 'Heavy Drizzle',
    56: 'Light Freezing Drizzle',
    57: 'Freezing Drizzle',
    61: 'Light Rain',
    63: 'Rain',
    65: 'Heavy Rain',
    66: 'Light Freezing Rain',
    67: 'Freezing Rain',
    71: 'Light Snow',
    73: 'Snow',
    75: 'Heavy Snow',
    77: 'Snow Grains',
    80: 'Light Rain Shower',
    81: 'Rain Shower',
    82: 'Heavy Rain Shower',
    85: 'Snow Shower',
    86: 'Heavy Snow Shower',
    95: 'Thunderstorm',
    96: 'Hailstorm',
    99: 'Heavy Hailstorm'
}


def format_params(parameters):
    result = {}
    for param in parameters:
        result[param['name']]=param['value']
    return result
    
    
def lambda_handler(event,context):
    #Event: {'agent': {'alias': 'TSTALIASID', 'name': 'GetWeather', 'version': 'DRAFT', 'id': '68X4FYOEUN'}, 
    #   'sessionId': '022346938362354', 'sessionAttributes': {}, 'promptSessionAttributes': {}, 'inputText': 'beijing', 
    #   'path': '/get_lat_long', 'httpMethod': 'GET', 'messageVersion': '1.0', 'actionGroup': 'get_weather', 
    #   'parameters': [{'name': 'place', 'type': 'string', 'value': 'beijing'}]}
    print(json.dumps(event))
    # params = format_params(event['parameters'])
    # print(params)
    if event['path'] == '/get_lat_long':
        
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
                "path": event['path'],
                "httpMethod": event['httpMethod'],
                "httpStatusCode": 200,
                "responseBody": {
                    "application/json": {
                        "body": json.dumps(result)
                    }
                }
            }
        }
    elif event['path'] == '/get_weather':
        latitude = event['queryStringParameters']['latitude']
        longitude = event['queryStringParameters']['longitude']
        print(latitude)
        url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
        response = requests.get(url)
        print(response.json()['current_weather'])
        weather_description = weather_codes.get(response.json()['current_weather']['weathercode'], 0)
        answer = "Now the latest weather data time is {}, the weather is {},  the temperature is {}, windspeed is {} km/h".format(response.json()['current_weather']['time'], weather_description, response.json()['current_weather']['temperature'],response.json()['current_weather']['windspeed'])
        print(answer)
        
        return {
            # "messageVersion": "1.0",
            # "response": {
            #     "path": event['path'],
            #     "httpMethod": event['httpMethod'],
            #     "httpStatusCode": 200,
            #     "responseBody": {
            #         "application/json": {
            #             "body": response.json()['current_weather']
            #         }
            #     }
            # }
            # "isBase64Encoded": false,
            "statusCode": 200,
            # "headers": { "headerName": "headerValue", ... },
            "body": answer
        }
    else:
        return {
            "messageVersion": "1.0",
            "response": {
                "actionGroup": event['actionGroup'],
                "path": event['path'],
                "httpMethod": event['httpMethod'],
                "httpStatusCode": 404,
                "responseBody": {
                    "application/json": {
                        "body": '{"err":"no such function"}'
                    }
                }
            }
        }