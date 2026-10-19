"""Login helpers that use Supabase Auth."""

import os

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()


def new_client() -> Client:
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


def sign_up(client: Client, email: str, password: str):
    return client.auth.sign_up({"email": email, "password": password})


def sign_in(client: Client, email: str, password: str):
    return client.auth.sign_in_with_password({"email": email, "password": password})


def sign_out(client: Client) -> None:
    client.auth.sign_out()