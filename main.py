from fastapi import FastAPI
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