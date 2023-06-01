from fastapi import Depends, FastAPI, APIRouter, HTTPException, Header

from typing_extensions import Annotated
async def router_get_token_header(x_token: Annotated[str, Header()]):
    if x_token != "fake-super-secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")

async def router_get_query_token(token: str):
    if token != "jessica":
        raise HTTPException(status_code=400, detail="No Jessica token provided")

router = APIRouter(
    prefix='/router_prefix',
    tags=["item_router"],
    dependencies=[Depends(router_get_token_header)],
    responses={404: {"description": "404 not found"}}
)

router_fake_items_db = {"plumbus": {"name": "Plumbus"}, "gun": {"name": "Portal Gun"}}

@router.get("/router/users/", tags=["users"])
async def read_users_router():
    return [{"username": "Rick"}, {"username": "Morty"}]


@router.get("/router/users/me", tags=["users"])
async def read_user_me_router():
    return {"username": "fakecurrentuser"}


@router.get("/router/users/{username}", tags=["users"])
async def read_user_router(username: str):
    return {"username": username}

from fastapi import BackgroundTasks

def write_notification(email: str, message=""):
    with open("log.txt", mode="w") as email_file:
        content = f"notification for {email}: {message}"
        email_file.write(content)

@router.post("/send-notification/{email}")
async def send_notification(email: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(write_notification, email, message="some notification")
    return {"message": "Notification sent in the background"}