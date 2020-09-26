# Facial Recognition
### Project Description
This is a project to explore state-of-the-art facial recognition models. 

For cloud deployment, `AWS Rekognition` directory contains [AWS Lambda](https://aws.amazon.com/lambda/) code snippets to detect and compare faces in images and to detect faces in videos using [AWS Rekognition API](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/rekognition.html).

For on-premises deployment, [InsightFace](http://insightface.ai/) models are used extensively. In particular, [RetinaFace](https://arxiv.org/abs/1905.00641) is the model to detect faces and [ArcFace](https://arxiv.org/abs/1801.07698) is the model to extract facial features for comparison.
- `demo_images.ipynb` is a demo to detect and compare faces in images.
  - `faces` directory contains 7 faces to compare during test.
- `demo_videos.py` is a demo to detect faces in videos.
Note that all data used for training and testing are publically available.

### Environment Setup
To install all libraries/dependencies used in this project, run
```bash
brew install ffmpeg
pip3 install -r requirement.txt
```
For Windows or Linux users, it is recommended to use [Conda](https://docs.conda.io/en/latest/) to install [FFmpeg](https://ffmpeg.org/).

### References:
- [InsightFace: a Deep Learning Toolkit for Face Analysis](http://insightface.ai/)
- [A survey on deep learning based face recognition](https://www.sciencedirect.com/science/article/abs/pii/S1077314219301183)
- [test footage](https://www.pexels.com/video/a-crowd-of-travelers-moving-inside-a-transport-terminal-3740034/)