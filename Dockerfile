FROM python:3.7

# set display port to avoid crash
ENV DISPLAY=:99
ADD ./ /
RUN pip install -r requirements.txt

ENTRYPOINT ["python"]
CMD ["OCBC_Extract.py"]
