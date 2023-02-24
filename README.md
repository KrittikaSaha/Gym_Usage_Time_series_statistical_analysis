# Helsinki Outdoor Gym Usage Statistical Analysis

## Quickstart

### Using Docker

To run the analysis using Docker, follow the following 4 steps:

1. git clone the repo:

```
git clone https://github.com/KrittikaSaha/Gym_Usage_Time_series_statistical_analysis.git
```
2. Run the docker daemon. Depending on the OS, for Windows, one may use Docker Desktop, for LInux and Mac one may use Rancher Desktop. For linux, example step to start running docker daemon manually:
```
sudo systemctl start docker
```
3. Once docker daemon is running, build the image using the dockerfile provided in the repo:

```
docker build -t gym_usage .
```
4. Now run the Docker image:
```
docker run gym_usage
```



### Running from local

Pre-requisites: Python3
1. Activate virtualenv

```
virtualenv venv
# For Linux/Mac:
source venv/bin/activate
# For Windows:
. venv/Scripts/activate
```
2. Pip install requirements
```
pip install -r requirements.txt
```
3. cd into /Gym_Usage_Time_series_statistical_analysis directory
```
cd Gym_Usage_Time_series_statistical_analysis
```
4. Run the analysis file and pytest

```
python3 Helsinki_Outdoor_gym_usage_statistics.py
pytest -v
```


