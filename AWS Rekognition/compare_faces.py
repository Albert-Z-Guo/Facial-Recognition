import boto3


def compare_faces(event, context):
    '''lambda handler to cmopare faces'''
    with open(event['sourceFile'], 'rb') as f:
        image_source = f.read()
    with open(event['targetFile'], 'rb') as f:
        image_target = f.read()

    client = boto3.client('rekognition')
    response = client.compare_faces(SimilarityThreshold=80, SourceImage={'Bytes': image_source}, TargetImage={'Bytes': image_target})

    for faceMatch in response['FaceMatches']:
        position = faceMatch['Face']['BoundingBox']
        similarity = str(faceMatch['Similarity'])
        print('The face at ' + str(position['Left']) + ' ' + str(position['Top']) + ' matches with ' + similarity + '% confidence')

    return len(response['FaceMatches'])
