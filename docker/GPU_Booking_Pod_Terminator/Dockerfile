FROM python:3.9
WORKDIR /app
COPY controller.py /app
RUN pip install kubernetes flask
CMD ["python", "-u", "controller.py"]