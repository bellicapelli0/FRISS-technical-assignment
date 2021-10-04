# FRISS-technical-assignment
Technical assignment for FRISS

### Docker API instructions

to build and start the container run the following commands: 
```
sudo docker build -t friss-api .
sudo docker run -d -p 5000:5000 friss-api
```

The "score" endpoint takes a claim ID as argument and returns the predicted class for the relative claim.
Example:

```
curl -XGET 0.0.0.0:5000/score/140152636
```

The "predict" endpoint predicts the class using the features passed through a json file and saves the predicted class, so it can be reaccessed through the "score" endpoint

```
curl -H "Content-Type: application/json" -d @test.json 0.0.0.0:5000/predict/
```
