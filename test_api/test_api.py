from fastapi.testclient import TestClient
from main import app
from main import *

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_update_item():
    item_id = 123
    item = {
        "name":"name",
        "description":"description",
        "price":10,
        "tax":0.5
    }
    res = client.put('/items/%s/field' %item_id, json={"item": item})
    assert res.status_code == 200
    assert res.json() == {"item_id": item_id, "item": item}

def test_update_item_validation():
    item_id = "str"
    item = {
        "name":"name",
        "description":"description",
        "price":10,
        "tax":0.5
    }
    res = client.put('/items/%s/field' %item_id, json={"item": item})
    assert res.status_code == 422

def test_signup():
    res = client.post('/user/hashsed/password/201', json={"username":"username", "password":"password", "email":"test@test.com", "full_name":"full_name"})
    assert res.status_code == 201
    assert res.json() == {"username": "username", "email": "test@test.com", "full_name": "full_name"}