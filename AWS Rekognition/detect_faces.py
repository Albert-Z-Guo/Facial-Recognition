import boto3
import json


def detect_faces(event, context):
    '''lambda handler to detect faces'''
    bucket = event['bucket']
    filename = event['filename']

    client = boto3.client('rekognition')
    response = client.detect_faces(Image={'S3Object': {'Bucket': bucket, 'Name': filename}}, Attributes=['ALL'])

    print('Detected faces for ' + filename)
    for faceDetail in response['FaceDetails']:
        print('The detected face is between ' + str(faceDetail['AgeRange']['Low']) + ' and ' + str(faceDetail['AgeRange']['High']) + ' years old')
        print('Here are the other attributes:')
        print(json.dumps(faceDetail, indent=4, sort_keys=True))
    return len(response['FaceDetails'])
