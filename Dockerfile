FROM python:3

ADD Helsinki_Outdoor_gym_usage_statistics.py /
ADD test_data_quality.py /
ADD hietaniemi-gym-data.csv /
ADD kaisaniemi-weather-data.csv /
ADD model.pkl /

COPY requirements.txt .
RUN pip install -r requirements.txt

CMD [ "/bin/bash", "-c", "/usr/local/bin/python Helsinki_Outdoor_gym_usage_statistics.py" ]