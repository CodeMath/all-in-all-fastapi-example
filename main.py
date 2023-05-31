from fastapi import FastAPI, HTTPException
from enum import Enum

class ModelName(str, Enum):

    def _generate_next_value_(name, start, count, last_values):
        return name

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}
    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}

fake_items_db = [{"item_name": "foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]

@app.get('/items/')
async def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip: skip + limit]

# Union[X,Y] either X or Y.
from typing import Union
@app.get('/items/{item_id}')
async def read_item(item_id: str, q: Union[str, None] = None):
    if q:
        return {"item_id": item_id, "q": q}
    return {"item_id": item_id}

@app.get('/items/{item_id}/short')
async def read_item(item_id: str, q: Union[int, None] = None, short: bool = False):
    item = {"item_id": item_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update(
            {"description": "This is an amazing items..."}
        )
    return item

# @app.get("/items/{item_id}/user")
# async def read_user_item(item_id: str, needy: str):
#     item = {"item_id": item_id, "needy": needy}
#     return item

@app.get("/items/{item_id}/user")
async def read_user_item(
    item_id: str, needy: str, skip: int = 0, limit: Union[int, None] = None
):
    item = {"item_id": item_id, "needy": needy, "skip": skip, "limit": limit}
    return item

"""
from pydantic import BaseModel
"""
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None

@app.post('/items/')
async def create_item(item: Item):
    item_dict = item.dict()
    if item.tax:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict


@app.put("/items/{item_id}")
async def create_item(item_id: int, item: Item, q: Union[str, None] = None):
    result = {"item_id": item_id, **item.dict()}
    if q:
        result.update({"q": q})
    return result

@app.put("/items/{item_id}/only/int")
async def create_item(item_id: int, item: Item, q: Union[int, None] = None):
    result = {"item_id": item_id, **item.dict()}
    if q:
        result.update({"q": q})
    return result

# less python 3.9
from fastapi import Query
from typing_extensions import Annotated
@app.get("/items/annotated/limit")
async def read_items(q: Annotated[Union[str, None], Query(min_length=3, max_length=50, regex="^fixedquery$")]):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results


@app.get("/items/annotated/limit/dots/literal/value")
async def read_items(q: Annotated[Union[str, None], Query(min_length=3)] = ...):
    """
    :param q: (required)\n
    :return:
    """
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results

from pydantic import Required
@app.get("/items/annotated/limit/required")
async def read_items(q: Annotated[Union[str, None], Query(min_length=3)] = Required):
    """
    :param q: (required) \n
    :return:
    """
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results


@app.get('/items/annotated/union')
async def read_items(q: Annotated[Union[list[str], None], Query()] = None):
    """
    :param q:(string) ?q=...&q=...&q=... list 로 붙일 수 있다.
    """
    query_items = {"q": q}
    return query_items

@app.get('/items/annotated/union/default')
async def read_items(q: Annotated[list[str], Query()] = ["foo", "bar"]):
    query_items = {"q": q}
    return query_items

@app.get('/items/annotated/union/list')
async def read_items(q: Annotated[list, Query()] = []):
    query_items = {"q": q}
    return query_items

@app.get('/items/annotated/additional/query')
async def read_items(q: Annotated[Union[str, None], Query(
    title="Query string!!!!!",
    description="Query string for the items to search in the database that have a good match",
    min_length=3)] = None):
    results = {"itmes": [{"item_id": "foo"}, {"item_id": "bar"}]}
    if q:
        results.update({"q": q})
    return results

@app.get("/items/annotated/item/query/alias")
async def read_items(q: Annotated[Union[str, None], Query(alias="item-query")] = None):
    """ q alias to item-query"""
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results


@app.get("/items/annotated/item/query/alias/none")
async def read_items(
    q: Annotated[
        Union[str, None],
        Query(
            alias="item-query",
            title="Query string",
            description="Query string for the items to search in the database that have a good match",
            min_length=3,
            max_length=50,
            regex="^fixedquery$",
            deprecated=True,
        ),
    ] = None
):
    """deprecated=True | api is active by specific API version but deprecated on doc"""
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results


@app.get("/items/hidden/query")
async def read_items(
    hidden_query: Annotated[Union[str, None], Query(include_in_schema=False)] = None
):
    """ include_in_schema=False """
    if hidden_query:
        return {"hidden_query": hidden_query}
    else:
        return {"hidden_query": "Not found"}

"""
[Generic validations and metadata]
alias
title
description
deprecated

[Validation specific for string]
min_length
max_length
regex
"""

from fastapi import Path
@app.get("/items/{item_id}/path")
async def read_items(
    item_id: int = Path(title="The ID of the item to get"),
    q: Annotated[Union[str, None], Query(alias="item-query")] = None
    ):
    """item_id is path (required) but q is available string or None"""
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    return results

@app.get("/items/{item_id}/path/int/str")
async def read_items(q: str, item_id: int = Path(title="The ID of the item to get")):
    """q,item_id required"""
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    return results

@app.get("/items/{item_id}/path/order")
async def read_items(*, item_id: int = Path(title="The ID of the item to get"), q: str):
    """
    if q is first and no query set and other path arg is needed then, first arg is *\n
    if not use *, error: non-default parameter follows default parameter
    """
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    return results

@app.get("/items/{item_id}/ge/1")
async def read_items(
    *, item_id: int = Path(title="The ID of the item to get", ge=1), q: str
):
    """ greater then 1: gt=1 / ge=1 / l=1 / le=1"""
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    return results

@app.get("/items/{item_id}/gt/0/l/1000")
async def read_items(
    *,
    item_id: int = Path(title="The ID of the item to get", gt=0, lt=1000),
    q: str,
):
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    return results

@app.get("/items/{item_id}/ge/0/le/1000/size")
async def read_items(
    *,
    item_id: int = Path(title="The ID of the item to get", ge=0, le=1000),
    q: str,
    size: float = Query(gt=0, lt=10.5),
):
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    return results


@app.put("/items/{item_id}/update/item")
async def update_item(
    item_id: Annotated[int, Path(title="The ID of the item to get", ge=0, le=1000)],
    q: Union[str, None] = None,
    item: Union[Item, None] = None,
):
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    if item:
        results.update({"item": item})
    return results

class User(BaseModel):
    username: str
    full_name: Union[str, None] = None

@app.put("/items/{item_id}/update/item/user")
async def update_item(item_id: int, item: Item, user: User):
    results = {"item_id": item_id, "item": item, "user": user}
    return results

from fastapi import Body
@app.put("/items/{item_id}/update/item/user/importance")
async def update_item(
    item_id: int, item: Item, user: User, importance: Annotated[int, Body()]
):
    """additional param in body: Body()"""
    results = {"item_id": item_id, "item": item, "user": user, "importance": importance}
    return results



@app.put("/items/{item_id}/update/item/user/multiple/body")
async def update_item(
    *,
    item_id: int,
    item: Item,
    user: User,
    importance: Annotated[int, Body(gt=0)],
    q: Union[str, None] = None,
):
    """ 명시적 추가 X 그냥 q: Union[str, none] = None """
    results = {"item_id": item_id, "item": item, "user": user, "importance": importance}
    if q:
        results.update({"q": q})
    return results

@app.put("/items/{item_id}/update/item/user/multiple/body/embed")
async def update_item(item_id: int, item: Annotated[Item, Body(embed=True)]):
    """
    Body(embed=True) is nested: {<Item>}
    """
    results = {"item_id": item_id, "item": item}
    return results

from pydantic import Field
class Items(BaseModel):
    name: str
    description: Union[str, None] = Field(
        default=None, title="The description of the item", max_length=300
    )
    price: float = Field(gt=0, description="The price must be greater than zero")
    tax: Union[float, None] = None

@app.put("/items/{item_id}/field")
async def update_item(item_id: int, item: Annotated[Items, Body(embed=True)]):
    results = {"item_id": item_id, "item": item}
    return results

"""
Pydantic's `Field` returns instance of `FiledInfo`.
* from fastapi imoprt Query, Path, Body is also subclass of Pydantic `FieldInfo` class.
* Body also returns objects of a subclass of FieldInfo directly
"""

class ItemTags(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None
    """
    python 3.9++, list | python 3.9--, List[str] and from Pydantic import List
    """
    tags: list[str] = []

@app.put("/items/{item_id}/tags")
async def update_item(item_id: int, item: ItemTags):
    results = {"item_id": item_id, "item": item}
    return results


class ItemTagsSet(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None
    tags: set[str] = set()

@app.put("/items/{item_id}/tags/sets")
async def update_item(item_id: int, item: ItemTagsSet):
    results = {"item_id": item_id, "item": item}
    return results

"""NESTED MODEL"""
class Image(BaseModel):
    url: str
    name: str

class ItemImage(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None
    tags: set[str] = set()
    image: Union[Image, None] = None

@app.put("/items/{item_id}/image")
async def update_item(item_id: int, item: ItemImage):
    results = {"item_id": item_id, "item": item}
    return results

from pydantic import HttpUrl

class ImageHttp(BaseModel):
    url: HttpUrl
    name: str


class ItemImageHttp(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None
    tags: set[str] = set()
    images: Union[list[ImageHttp], None] = None


@app.put("/items/{item_id}/image/http")
async def update_item(item_id: int, item: ItemImageHttp):
    results = {"item_id": item_id, "item": item}
    return results


class ImageHttpUrl(BaseModel):
    url: HttpUrl
    name: str


class ItemHttpUrl(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None
    tags: set[str] = set()
    images: Union[list[ImageHttpUrl], None] = None


class Offer(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    items: list[ItemHttpUrl]


@app.post("/offers/")
async def create_offer(offer: Offer):
    return offer

@app.post("/images/multiple/")
async def create_multiple_images(images: list[ImageHttpUrl]):
    for image in images:
        print(image.url)
    return images

@app.post("/index-weights/")
async def create_index_weights(weights: dict[str, float]):
    return weights


class ItemConfig(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None

    class Config:
        """Default schema and validate"""
        schema_extra = {
            "example": {
                "name": "Foo",
                "description": "A very nice Item",
                "price": 35.4,
                "tax": 3.2,
            }
        }


@app.put("/items/{item_id}/config")
async def update_item(item_id: int, item: ItemConfig):
    results = {"item_id": item_id, "item": item}
    return results



"""
유효성 검사가 아닌 doc 목적, 유효성 검사까지 하려면 위 schema_extra 사용할것
"""
class ItemFiledSchema(BaseModel):
    name: str = Field(example="Foo")
    description: Union[str, None] = Field(default=None, example="A very nice Item")
    price: float = Field(example=35.4)
    tax: Union[float, None] = Field(default=None, example=3.2)


@app.put("/items/{item_id}/filed/schema")
async def update_item(item_id: int, item: ItemFiledSchema):
    results = {"item_id": item_id, "item": item}
    return results


class ItemBodyExample(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: Union[float, None] = None


@app.put("/items/{item_id}/body/example")
async def update_item(
    item_id: int,
    item: Annotated[
        ItemBodyExample,
        Body(
            example={
                "name": "Foo",
                "description": "A very nice Item",
                "price": 35.4,
                "tax": 3.2,
            },
        ),
    ],
):
    results = {"item_id": item_id, "item": item}
    return results


@app.put("/items/{item_id}/body/examples/")
async def update_item(
    *,
    item_id: int,
    item: Annotated[
        Item,
        Body(
            examples={
                "normal": {
                    "summary": "A normal example",
                    "description": "A **normal** item works correctly.",
                    "value": {
                        "name": "Foo",
                        "description": "A very nice Item",
                        "price": 35.4,
                        "tax": 3.2,
                    },
                },
                "converted": {
                    "summary": "An example with converted data",
                    "description": "FastAPI can convert price `strings` to actual `numbers` automatically",
                    "value": {
                        "name": "Bar",
                        "price": "35.4",
                    },
                },
                "invalid": {
                    "summary": "Invalid data is rejected with an error",
                    "value": {
                        "name": "Baz",
                        "price": "thirty five point four",
                    },
                },
            },
        ),
    ],
):
    results = {"item_id": item_id, "item": item}
    return results

from datetime import datetime, time, timedelta
from uuid import UUID
@app.put("/items/{item_id}/complex")
async def read_items(
    item_id: UUID,
    start_datetime: Annotated[Union[datetime, None], Body()] = None,
    end_datetime: Annotated[Union[datetime, None], Body()] = None,
    repeat_at: Annotated[Union[time, None], Body()] = None,
    process_after: Annotated[Union[timedelta, None], Body()] = None,
):
    start_process = start_datetime + process_after
    duration = end_datetime - start_process
    return {
        "item_id": item_id,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "repeat_at": repeat_at,
        "process_after": process_after,
        "start_process": start_process,
        "duration": duration,
    }

from fastapi import Cookie

@app.get('/cookie/items')
async def read_item_cookie(ads_id: Annotated[Union[str, None], Cookie()] = None):
    return {"ads_id": ads_id}

from fastapi import Header
@app.get("/header/items/")
async def read_items_header(user_agent: Union[str, None] = Header(default=None)):
    return {"User-Agent": user_agent}

@app.get("/header/items/undercover")
async def read_items_headers(strange_header: Union[str, None] = Header(default=None, convert_underscores=False)):
    return {"strange_header": strange_header}

@app.get('xtoken/items')
async def xtoken_items(x_token: Union[list[str], None] = Header(default=None)):
    return {"X-Token values": x_token}

"""return models"""
from typing import Any
@app.post('/create/item', response_model=ItemTags)
async def create_itme_obj(item: ItemTags) -> Any:
    return item

@app.get('/get/item', response_model=list[ItemTags])
async def get_item() -> list[ItemTags]:
    return [
        ItemTags(name="Portal Gun", price=42.0),
        ItemTags(name="Plumbus", price=31.0),
    ]


from pydantic import EmailStr
class UserIn(BaseModel):
    username: str
    password: str
    email: EmailStr
    full_name: Union[str, None] = None

class UserOut(BaseModel):
    username: str
    email: EmailStr
    full_name: Union[str, None] = None

@app.post('/user/signup', response_model=UserOut)
async def create_user(user: UserIn) -> Any:
    return user


class BaseUser(BaseModel):
    username: str
    email: EmailStr
    full_name: Union[str, None] = None

class UserIns(BaseUser):
    password: str

@app.post("/user/base/in")
async def create_user(user: UserIns) -> BaseUser:
    return user


"""Directly response"""
from fastapi import Response
from fastapi.responses import JSONResponse, RedirectResponse
@app.get("/portal")
async def get_portal(teleport: bool = False) -> Response:
    if teleport:
        return RedirectResponse(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    return JSONResponse(content={"message": "Here's your interdimensional portal."})

@app.get("/teleport")
async def get_teleport() -> RedirectResponse:
    return RedirectResponse(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")

@app.get("/portal/response/model/none", response_model=None)
async def get_portal(teleport: bool = False) -> Union[Response, dict]:
    if teleport:
        return RedirectResponse(url="https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    return {"message": "Here's your interdimensional portal."}


class ItemObj(BaseModel):
    name: str
    description: Union[str, None] = None
    price: float
    tax: float = 10.5
    tags: list[str] = []

items_obj = {
    "foo": {"name": "Foo", "price": 50.2},
    "bar": {"name": "Bar", "description": "The bartenders", "price": 62, "tax": 20.2},
    "baz": {"name": "Baz", "description": None, "price": 50.2, "tax": 10.5, "tags": []},
}
"""if none of filed at obj, `response_model_exclude_unset=True` active"""
@app.get("/items/obj/{item_id}", response_model=ItemObj, response_model_exclude_unset=True)
async def read_item(item_id: str):
    return items_obj[item_id]

@app.get(
    "/items/obj/{item_id}/name",
    response_model=Item,
    response_model_include={"name", "description"},
)
async def read_item_name(item_id: str):
    return items_obj[item_id]


@app.get("/items/obj/{item_id}/public", response_model=Item, response_model_exclude={"tax"})
async def read_item_public_data(item_id: str):
    return items_obj[item_id]


"""hashed db"""
class UserInHashed(BaseModel):
    username: str
    password: str
    email: EmailStr
    full_name: Union[str, None] = None

class UserOutHashed(BaseModel):
    username: str
    email: EmailStr
    full_name: Union[str, None] = None

class UserInDB(BaseModel):
    username: str
    hashed_password: str
    email: EmailStr
    full_name: Union[str, None] = None

def fake_password_hasher(raw_password: str):
    return "supersecret"+raw_password

def fake_save_user(user_in: UserInHashed):
    hashed_password = fake_password_hasher(user_in.password)
    user_in_db = UserInDB(**user_in.dict(), hashed_password=hashed_password)
    print("User saved! ..not really")
    return user_in_db

@app.post("/user/hashsed/password", response_model=UserOutHashed)
async def create_user(user_in: UserInHashed):
    user_saved = fake_save_user(user_in)
    return user_saved

"""lighting code"""
class UserBaseModel(BaseModel):
    username: str
    email: EmailStr
    full_name: Union[str, None] = None

class UserInModel(UserBaseModel):
    password: str

class UserOutModel(UserBaseModel):
    pass

class UserInDBModel(UserBaseModel):
    hashed_password: str

def fake_password_hasher_model(raw_password: str):
    return "supersecret"+raw_password

def fake_save_user_model(user_in: UserInModel):
    hashed_password = fake_password_hasher(user_in.password)
    user_in_db = UserInDBModel(**user_in.dict(), hashed_password=hashed_password)
    print("User saved! ..not really")
    return user_in_db

from fastapi import status
@app.post("/user/hashsed/password/201", response_model=UserOutModel, status_code=status.HTTP_201_CREATED)
async def create_user(user_in: UserInModel):
    user_saved = fake_save_user_model(user_in)
    return user_saved

class BaseItem(BaseModel):
    description: str
    type: str

class CarItem(BaseItem):
    type = "car"

class PlaneItem(BaseItem):
    type = "plane"
    size: int

items = {
    "item1": {"description": "All my friends drive a low rider", "type": "car"},
    "item2": {
        "description": "Music is my aeroplane, it's my aeroplane",
        "type": "plane",
        "size": 5,
    },
}


@app.get("/anyof/items/{item_id}", response_model=Union[PlaneItem, CarItem])
async def read_item(item_id: str):
    return items[item_id]

@app.get("/keyword-weights/", response_model=dict[str, float])
async def read_keyword_weights():
    return {"foo": 2.3, "bar": 3.4}

"""
Status code

200: success
201: create
204: nobody
3xx: redirection & nobody
304: no modify
4xx: client error
404: not found
5xx: server error
"""
@app.post("/201/items/", status_code=201)
async def create_item(name: str):
    return {"name": name}

from fastapi import status
@app.post('/create/items/new', status_code=status.HTTP_201_CREATED)
async def create_item(name: str):
    return {"name": name}

"""Form"""
from fastapi import Form

@app.post('/login')
async def login(username: Annotated[str, Form()], password: Annotated[str, Form()]):
    return {"username": username}

from fastapi import File, UploadFile
"""
File or UploadFile
> use UploadFile, no bytes: file upload is no more use than enough memory. and read file's metadata.
> UploadFile: filename, content_type, file, `SpooledTemporaryFile`(if overflow memory, auto del)
> async method: write(data), read(size), seek(offset), close()

`media format`: application/x-www-form-urlencoded
`file format`: multipart/form-data
"""
@app.post('/file/')
async def create_file(file: bytes = File()):
    return {"file_size": len(file)}

@app.post('/uploadfile/')
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}


@app.post('/files/')
async def create_files(files: list[bytes] = File()):
    return {"file_sizes": [len(file) for file in files]}

@app.post('/uploadfiles/')
async def create_upload_files(files: list[UploadFile]):
    return {"filename": [file.filename for file in files]}


@app.post("/files/fileb/token")
async def create_file(
    file: bytes = File(), fileb: UploadFile = File(), token: str = Form()
):
    """Using file and Form ,one request"""
    return {
        "file_size": len(file),
        "token": token,
        "fileb_content_type": fileb.content_type,
    }

"""Error handling"""
from fastapi import HTTPException
items_one = {"foo": "The Foo wwwww"}

@app.get('/items/one/404')
async def read_item(item_id: str):
    if item_id not in items_one:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item": items_one[item_id]}


@app.get("/items-header/{item_id}")
async def read_item_header(item_id: str):
    """error with custom header at response"""
    if item_id not in items:
        raise HTTPException(
            status_code=404,
            detail="Item not found",
            headers={"X-Error": "There goes my error"},
        )
    return {"item": items[item_id]}

class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name

from fastapi import Request
@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something. There goes a rainbow..."},
    )

@app.get("/unicorns/{name}")
async def read_unicorn(name: str):
    if name == "yolo":
        raise UnicornException(name=name)
    return {"unicorn_name": name}

"""
Basic Exception
from fastapi.exceptions import RequestValidationError
"""

from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc):
    return PlainTextResponse(str(exc.detail), status_code=exc.status_code)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc):
    return PlainTextResponse(str(exc), status_code=400)

@app.get("/errors/handler/items/{item_id}")
async def read_item(item_id: int):
    if item_id == 3:
        raise HTTPException(status_code=418, detail="Nope! I don't like 3.")
    return {"item_id": item_id}

from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)

@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request, exc):
    print(f"OMG! An HTTP error!: {repr(exc)}")
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    print(f"OMG! The client sent invalid data!: {exc}")
    return await request_validation_exception_handler(request, exc)


@app.get("/custom/handler/error/items/{item_id}")
async def read_item(item_id: int):
    if item_id == 3:
        raise HTTPException(status_code=418, detail="Nope! I don't like 3.")
    return {"item_id": item_id}

class Tags(Enum):
    itmes = "items"
    users = "users"

@app.post('/items/tags', response_model=ItemTagsSet,  tags=[Tags.itmes],
    # summary="Create an item",
    # description="Create an item with all the information, name, description, price, tax and a set of unique tags",
    response_description="The created item",
          )
async def create_item(item: ItemTagsSet):
    """
    Create an item with all the information:

    - **name**: each item must have a name
    - **description**: a long description
    - **price**: required
    - **tax**: if the item doesn't have tax, you can omit this
    - **tags**: a set of unique tag strings for this item
    """
    return item

"""path decorator: add deprecated=True"""
@app.get('/items/tags/read', tags=[Tags.itmes], deprecated=True)
async def read_item():
    return [{"name": "foo", "price": 322}]

@app.get('/users/tags', tags=[Tags.users])
async def read_users():
    return [{"username": "johndoe"}]

from fastapi.encoders import jsonable_encoder
fake_db = {}
@app.put("/jsonable/encoder/items/{id}")
def update_item(id: str, item: Item):
    json_compatible_item_data = jsonable_encoder(item)
    fake_db[id] = json_compatible_item_data

@app.put("/body/update/items/{item_id}", response_model=ItemTags)
async def update_item(item_id: str, item: ItemTags):
    stored_item_data = items_obj[item_id]
    stored_item_model = ItemTags(**stored_item_data)

    update_data = item.dict(exclude_unset=True)
    updated_item = stored_item_model.copy(update=update_data)

    update_item_encoded = jsonable_encoder(updated_item)
    items_obj[item_id] = update_item_encoded
    return update_item_encoded

"""dependency param"""
from fastapi import Depends

async def common_parameters(
        q: Union[str, None] = None, skip: int = 0, limit: int = 100
):
    return {"q": q, "skip": skip, "limit": limit}

CoomonDep = Annotated[dict, Depends(common_parameters)]

@app.get("/common/param/items/")
async def read_items(commons: CoomonDep):
    return commons


@app.get("/common/param/users/")
async def read_users(commons: CoomonDep):
    return commons


def query_extractor(s: Union[str, None] = None):
    return s


def query_or_cookie_extractor(
    s: Annotated[str, Depends(query_extractor)],
    last_query: Annotated[Union[str, None], Cookie()] = None,
):
    if not s:
        return last_query
    return s

queryDefault = Annotated[str, Depends(query_or_cookie_extractor)]
@app.get("/dependency/low/low/items/", dependencies=[Depends(common_parameters), Depends(query_or_cookie_extractor)])
async def read_query(
    query_or_default: queryDefault
):
    return {"q_or_cookie": query_or_default}


fake_items_dbs = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


class CommonQueryParams:
    def __init__(self, q: Union[str, None] = None, skip: int = 0, limit: int = 100):
        self.q = q
        self.skip = skip
        self.limit = limit


@app.get("/common/query/param/items/")
async def read_items(commons: CommonQueryParams = Depends(CommonQueryParams)):
    response = {}
    if commons.q:
        response.update({"q": commons.q})
    items = fake_items_dbs[commons.skip : commons.skip + commons.limit]
    response.update({"items": items})
    return response