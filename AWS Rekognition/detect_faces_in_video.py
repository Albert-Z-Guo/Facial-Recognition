import boto3


def detect_faces_in_video(event, context):
    '''lambda handler to detect faces in a video'''
    client = boto3.client('rekognition')
    response=client.start_face_detection(Video={'S3Object': {'Bucket': event['bucket'], 'Name': event['filename']}})

    paginationToken = ''
    finished = False
    while finished == False:
        response = client.get_face_detection(JobId=response['JobId'])

        print('Codec: ' + response['VideoMetadata']['Codec'])
        print('Duration: ' + str(response['VideoMetadata']['DurationMillis']))
        print('Format: ' + response['VideoMetadata']['Format'])
        print('Frame rate: ' + str(response['VideoMetadata']['FrameRate']))

        for faceDetection in response['Faces']:
            print('Face: ' + str(faceDetection['Face']))
            print('Confidence: ' + str(faceDetection['Face']['Confidence']))
            print('Timestamp: ' + str(faceDetection['Timestamp']))

        if 'NextToken' in response:
            paginationToken = response['NextToken']
        else:
            finished = True