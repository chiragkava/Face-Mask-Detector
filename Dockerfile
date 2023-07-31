FROM public.ecr.aws/lambda/python:3.7

RUN yum install -y gcc

COPY requirements.txt ./
RUN pip3 install -r requirements.txt

COPY lambda_handler.py ./
COPY templates ./templates
COPY models ./models
COPY utils ./utils
COPY constants ./constants
COPY static ./static
COPY weights ./weights
COPY service ./service

CMD ["lambda_handler.lambda_handler"]
